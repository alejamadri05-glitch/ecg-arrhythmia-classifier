"""Bases externas de PhysioNet para validación fuera de MIT-BIH.

Es la única forma honesta de comparar versiones del modelo después de gastar DS2: pacientes
nuevos, otro hospital, otro equipo de registro.

| | MIT-BIH (entrenamiento) | INCART | SVDB |
|---|---|---|---|
| Derivación | MLII | **II** (no tiene MLII) | **ECG1** (sin identificar) |
| Muestreo | 360 Hz | 257 Hz | 128 Hz |
| Población | EE. UU., años 70–80 | Rusia, isquémica | rica en supraventriculares |

Las señales se remuestrean a 360 Hz y las anotaciones se reescalan, así que entran al mismo
pipeline que MIT-BIH.

Uso:  python -m ecg.external --db incart     # ~820 MB
      python -m ecg.external --db svdb       # ~54 MB
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb

from ecg import config
from ecg.data import dl_database_with_retry
from ecg.preprocess import bandpass, resample
from ecg.segment import beat_windows


@dataclass(frozen=True)
class ExternalDB:
    """Una base de PhysioNet y cómo adaptarla al formato del proyecto."""

    name: str
    physionet: str  # identificador de la base en PhysioNet
    lead: str  # canal a usar (el más parecido a MLII que tenga)
    fs: int  # frecuencia de muestreo original
    sentinel: str  # archivo que indica que la descarga terminó

    @property
    def dir(self) -> Path:
        return config.DATA_DIR / self.name


INCART = ExternalDB("incart", "incartdb", "II", 257, "I75.atr")
SVDB = ExternalDB("svdb", "svdb", "ECG1", 128, "894.atr")
EXTERNAL_DBS = {db.name: db for db in (INCART, SVDB)}

# Compatibilidad con el análisis de INCART ya publicado
INCART_DIR, INCART_LEAD, INCART_FS = INCART.dir, INCART.lead, INCART.fs


def record_ids(db: ExternalDB = INCART) -> list[str]:
    return (db.dir / "RECORDS").read_text().split()


def download(db: ExternalDB = INCART) -> None:
    if (db.dir / db.sentinel).exists():
        return
    db.dir.mkdir(parents=True, exist_ok=True)
    dl_database_with_retry(db.physionet, db.dir)


def verify(db: ExternalDB = INCART) -> list[tuple[str, str]]:
    """Registros que no abren. PhysioNet a veces devuelve una página de error en lugar del
    archivo, y queda una cabecera o una anotación corrupta."""
    malos = []
    for rec_id in record_ids(db):
        try:
            wfdb.rdheader(str(db.dir / rec_id))
            wfdb.rdann(str(db.dir / rec_id), "atr")
        except Exception as e:  # noqa: BLE001 - se reporta el motivo, no se distingue el tipo
            malos.append((rec_id, str(e)[:60]))
    return malos


def repair(db: ExternalDB = INCART) -> list[tuple[str, str]]:
    """Vuelve a descargar los registros rotos y devuelve los que siguen fallando."""
    for rec_id, _ in verify(db):
        for ext in ("hea", "dat", "atr"):
            (db.dir / f"{rec_id}.{ext}").unlink(missing_ok=True)
        wfdb.dl_files(db.physionet, str(db.dir), [f"{rec_id}.{e}" for e in ("hea", "dat", "atr")])
    return verify(db)


def extract_record(rec_id: str, db: ExternalDB = INCART) -> dict[str, np.ndarray]:
    """Latidos etiquetados de un registro, ya en el formato del modelo (360 Hz)."""
    path = str(db.dir / rec_id)
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    names = [s.upper() for s in rec.sig_name]
    if db.lead not in names:
        raise ValueError(f"{rec_id}: no tiene el canal {db.lead} ({rec.sig_name})")
    x = rec.p_signal[:, names.index(db.lead)].astype(np.float64)

    x = bandpass(resample(x, rec.fs, config.FS), config.FS)  # remuestreo + filtro del proyecto
    samples = np.round(np.asarray(ann.sample) * (config.FS / rec.fs)).astype(np.int64)
    symbols = np.asarray(ann.symbol)

    is_beat = np.isin(symbols, list(config.BEAT_SYMBOLS))
    samples, symbols = samples[is_beat], symbols[is_beat]
    idx, X, F = beat_windows(x, config.FS, samples)
    sym = symbols[idx]
    keep = np.isin(sym, list(config.AAMI_MAP))  # descarta la clase Q
    idx, X, F, sym = idx[keep], X[keep], F[keep], sym[keep]
    return {
        "X": X,
        "F": F,
        "y": np.array([config.AAMI_MAP[s] for s in sym]),
        "symbol": sym,
        "r_peak": samples[idx],
        "record": np.full(len(idx), int("".join(c for c in rec_id if c.isdigit())), dtype=np.int32),
    }


def unmapped_symbols(db: ExternalDB = INCART) -> dict[str, int]:
    """Símbolos de latido que la base usa y el proyecto no clasifica (se documentan)."""
    counts: dict[str, int] = {}
    for rec_id in record_ids(db):
        ann = wfdb.rdann(str(db.dir / rec_id), "atr")
        for sym in ann.symbol:
            if sym not in config.AAMI_MAP and sym in config.BEAT_SYMBOLS:
                counts[sym] = counts.get(sym, 0) + 1
    return counts


def build_split(db: ExternalDB = INCART, limit: int | None = None) -> dict[str, np.ndarray]:
    parts = [extract_record(r, db) for r in record_ids(db)[:limit]]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", choices=sorted(EXTERNAL_DBS), default="incart")
    parser.add_argument(
        "--limit", type=int, default=None, help="usar solo los primeros N registros"
    )
    args = parser.parse_args()
    db = EXTERNAL_DBS[args.db]

    download(db)
    if malos := repair(db):
        raise SystemExit(f"Registros que siguen sin abrir tras reintentar: {malos}")
    split = build_split(db, args.limit)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = config.PROCESSED_DIR / f"{db.name}.npz"
    np.savez_compressed(out, **split)
    classes, counts = np.unique(split["y"], return_counts=True)
    resumen = ", ".join(f"{c}={n}" for c, n in zip(classes, counts, strict=True))
    n_rec = len(np.unique(split["record"]))
    print(f"{db.name}: {len(split['y'])} latidos de {n_rec} registros ({resumen}) -> {out}")


if __name__ == "__main__":
    main()
