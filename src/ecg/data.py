"""Descarga y lectura de registros MIT-BIH."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb

from ecg import config


@dataclass
class Record:
    rec_id: int
    fs: float
    signal: np.ndarray  # canal MLII, en mV, sin filtrar
    ann_samples: np.ndarray  # posición de cada anotación (muestras)
    ann_symbols: np.ndarray  # símbolo de cada anotación


def download(db_dir: Path = config.DB_DIR) -> None:
    """Descarga MIT-BIH completo (~100 MB) si todavía no está."""
    db_dir = Path(db_dir)
    if (db_dir / "234.atr").exists():
        return
    db_dir.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database("mitdb", dl_dir=str(db_dir))


def lead_index(sig_names: list[str], lead: str = config.LEAD) -> int:
    """Índice del canal por nombre. Nunca asumir índice 0 (el registro 114 está invertido)."""
    if lead not in sig_names:
        raise ValueError(f"Derivación {lead!r} no está en el registro: {sig_names}")
    return sig_names.index(lead)


def rhythm_at(rec_id: int, samples: np.ndarray, db_dir: Path = config.DB_DIR) -> np.ndarray:
    """Ritmo anotado vigente en cada muestra: "(N", "(AFIB", "(SVTA", "(VT", etc.

    MIT-BIH marca los cambios de ritmo con anotaciones "+" cuyo aux_note empieza con "(".
    Sirve para explicar errores (p. ej. latidos N que parecen prematuros durante una FA).
    """
    ann = wfdb.rdann(str(Path(db_dir) / str(rec_id)), "atr")
    idx = [i for i, note in enumerate(ann.aux_note) if note.startswith("(")]
    starts = ann.sample[idx]
    names = np.array([ann.aux_note[i].rstrip("\x00") for i in idx] + ["?"])
    pos = np.searchsorted(starts, np.asarray(samples), side="right") - 1
    return names[np.where(pos >= 0, pos, -1)]


def load_record(rec_id: int, db_dir: Path = config.DB_DIR, lead: str = config.LEAD) -> Record:
    path = str(Path(db_dir) / str(rec_id))
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    ch = lead_index(rec.sig_name, lead)
    return Record(
        rec_id=rec_id,
        fs=float(rec.fs),
        signal=rec.p_signal[:, ch].astype(np.float64),
        ann_samples=np.asarray(ann.sample),
        ann_symbols=np.asarray(ann.symbol),
    )
