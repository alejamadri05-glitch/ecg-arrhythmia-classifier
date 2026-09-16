"""Validación externa con INCART (St Petersburg, PhysioNet).

Es la única forma honesta de comparar v1 y v2 después de haber gastado DS2: pacientes nuevos,
otro hospital, otro equipo de registro.

Diferencias con MIT-BIH, que son parte de lo que se quiere medir:

- **Derivación:** INCART no tiene MLII. Se usa la **II** estándar, la más parecida.
- **Frecuencia de muestreo:** 257 Hz; se remuestrea a 360 Hz y se reescalan las anotaciones.
- **Población:** 75 pacientes rusos con cardiopatía isquémica, muchos con frecuencias y
  morfologías distintas a las de MIT-BIH.

Uso:  python -m ecg.external            # descarga (~820 MB) y genera data/processed/incart.npz
"""

import argparse
from pathlib import Path

import numpy as np
import wfdb

from ecg import config
from ecg.preprocess import bandpass, resample
from ecg.segment import beat_windows

INCART_DIR = config.DATA_DIR / "incart"
INCART_LEAD = "II"  # no hay MLII: la II estándar es la derivación más parecida
INCART_FS = 257
BASE_URL = "https://physionet.org/files/incartdb/1.0.0"


def record_ids(db_dir: Path = INCART_DIR) -> list[str]:
    return (Path(db_dir) / "RECORDS").read_text().split()


def download(db_dir: Path = INCART_DIR) -> None:
    db_dir = Path(db_dir)
    if (db_dir / "I75.atr").exists():
        return
    db_dir.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database("incartdb", dl_dir=str(db_dir))


def extract_record(rec_id: str, db_dir: Path = INCART_DIR) -> dict[str, np.ndarray]:
    """Latidos etiquetados de un registro de INCART, ya en el formato del modelo (360 Hz)."""
    path = str(Path(db_dir) / rec_id)
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    names = [s.upper() for s in rec.sig_name]
    if INCART_LEAD not in names:
        raise ValueError(f"{rec_id}: no tiene la derivación {INCART_LEAD} ({rec.sig_name})")
    x = rec.p_signal[:, names.index(INCART_LEAD)].astype(np.float64)

    x = resample(x, rec.fs, config.FS)  # 257 -> 360 Hz
    scale = config.FS / rec.fs
    samples = np.round(np.asarray(ann.sample) * scale).astype(np.int64)
    symbols = np.asarray(ann.symbol)

    is_beat = np.isin(symbols, list(config.BEAT_SYMBOLS))
    samples, symbols = samples[is_beat], symbols[is_beat]
    x = bandpass(x, config.FS)

    idx, X, F = beat_windows(x, config.FS, samples)
    sym = symbols[idx]
    keep = np.isin(sym, list(config.AAMI_MAP))
    idx, X, F, sym = idx[keep], X[keep], F[keep], sym[keep]
    return {
        "X": X,
        "F": F,
        "y": np.array([config.AAMI_MAP[s] for s in sym]),
        "symbol": sym,
        "r_peak": samples[idx],
        "record": np.full(len(idx), int(rec_id.lstrip("I")), dtype=np.int32),
    }


def unmapped_symbols(db_dir: Path = INCART_DIR) -> dict[str, int]:
    """Símbolos de latido que INCART usa y el proyecto no clasifica (se documentan)."""
    counts: dict[str, int] = {}
    for rec_id in record_ids(db_dir):
        ann = wfdb.rdann(str(Path(db_dir) / rec_id), "atr")
        for sym in ann.symbol:
            if sym not in config.AAMI_MAP and sym in config.BEAT_SYMBOLS:
                counts[sym] = counts.get(sym, 0) + 1
    return counts


def build_split(db_dir: Path = INCART_DIR, limit: int | None = None) -> dict[str, np.ndarray]:
    ids = record_ids(db_dir)[:limit]
    parts = [extract_record(r, db_dir) for r in ids]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None, help="usar solo los primeros N registros"
    )
    args = parser.parse_args()

    download()
    split = build_split(limit=args.limit)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = config.PROCESSED_DIR / "incart.npz"
    np.savez_compressed(out, **split)
    classes, counts = np.unique(split["y"], return_counts=True)
    resumen = ", ".join(f"{c}={n}" for c, n in zip(classes, counts, strict=True))
    n_rec = len(np.unique(split["record"]))
    print(f"INCART: {len(split['y'])} latidos de {n_rec} registros ({resumen}) -> {out}")


if __name__ == "__main__":
    main()
