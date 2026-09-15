"""Extracción de latidos (ventana alrededor del pico R) y features de ritmo (RR).

`beat_windows` no depende de las anotaciones, así que la API la reutiliza con picos R
detectados automáticamente. `extract_record` agrega las etiquetas AAMI de MIT-BIH.

Uso:  python -m ecg.segment      # genera data/processed/ds1.npz y ds2.npz
"""

from pathlib import Path

import numpy as np

from ecg import config
from ecg.data import load_record
from ecg.preprocess import bandpass, zscore

RR_FEATURE_NAMES = ["pre_rr", "post_rr", "pre_rr_over_local", "post_over_pre_rr"]


def beat_windows(
    x: np.ndarray,
    fs: float,
    r_peaks: np.ndarray,
    pre: int = config.PRE_SAMPLES,
    post: int = config.POST_SAMPLES,
    local_beats: int = config.LOCAL_RR_BEATS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Segmenta cada latido con vecino previo y siguiente y ventana completa.

    Args:
        x: señal ya filtrada.
        r_peaks: posiciones de todos los latidos (incluidos los no clasificables, que
            igual cuentan para el ritmo).

    Returns:
        idx: índice en `r_peaks` de cada latido retenido.
        X: (n, pre+post) segmentos normalizados con z-score.
        F: (n, 4) features RR (ver RR_FEATURE_NAMES). El RR siguiente implica un retraso
            de un latido en la clasificación: aceptable para análisis tipo Holter.
    """
    r = np.asarray(r_peaks, dtype=np.int64)
    rr = np.maximum(np.diff(r) / fs, 1e-3)  # protege ante anotaciones duplicadas
    idx, X, F = [], [], []
    for i in range(1, len(r) - 1):
        s = r[i]
        if s - pre < 0 or s + post > len(x):
            continue
        pre_rr, post_rr = rr[i - 1], rr[i]
        local_rr = rr[max(0, i - local_beats) : i].mean()
        idx.append(i)
        X.append(zscore(x[s - pre : s + post]))
        F.append([pre_rr, post_rr, pre_rr / local_rr, post_rr / pre_rr])
    n = len(idx)
    return (
        np.asarray(idx, dtype=np.int64),
        np.asarray(X, dtype=np.float32).reshape(n, pre + post),
        np.asarray(F, dtype=np.float32).reshape(n, 4),
    )


def extract_record(rec_id: int, db_dir: Path = config.DB_DIR) -> dict[str, np.ndarray]:
    """Latidos etiquetados (N/S/V/F) de un registro de MIT-BIH."""
    rec = load_record(rec_id, db_dir)
    x = bandpass(rec.signal, rec.fs)

    is_beat = np.isin(rec.ann_symbols, list(config.BEAT_SYMBOLS))  # descarta +, ~, |, etc.
    samples, symbols = rec.ann_samples[is_beat], rec.ann_symbols[is_beat]

    idx, X, F = beat_windows(x, rec.fs, samples)
    sym = symbols[idx]
    keep = np.isin(sym, list(config.AAMI_MAP))  # descarta clase Q
    idx, X, F, sym = idx[keep], X[keep], F[keep], sym[keep]
    return {
        "X": X,
        "F": F,
        "y": np.array([config.AAMI_MAP[s] for s in sym]),
        "symbol": sym,
        "r_peak": samples[idx],
        "record": np.full(len(idx), rec_id, dtype=np.int32),
    }


def build_split(records: list[int], db_dir: Path = config.DB_DIR) -> dict[str, np.ndarray]:
    parts = [extract_record(r, db_dir) for r in records]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def load_split(name: str, processed_dir: Path = config.PROCESSED_DIR) -> dict[str, np.ndarray]:
    with np.load(Path(processed_dir) / f"{name}.npz", allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def main() -> None:
    from ecg.data import download

    download()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, records in [("ds1", config.DS1), ("ds2", config.DS2)]:
        split = build_split(records)
        out = config.PROCESSED_DIR / f"{name}.npz"
        np.savez_compressed(out, **split)
        classes, counts = np.unique(split["y"], return_counts=True)
        summary = ", ".join(f"{c}={n}" for c, n in zip(classes, counts, strict=True))
        print(f"{name}: {len(split['y'])} latidos de {len(records)} registros ({summary}) -> {out}")


if __name__ == "__main__":
    main()
