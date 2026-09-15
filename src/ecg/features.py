"""Features para el modelo baseline, calculadas sobre las ventanas de `segment.beat_windows`.

Las ventanas ya vienen normalizadas con z-score, así que las features de morfología son de
forma (ancho, asimetría, energía por región), no de amplitud absoluta en mV.
Todo está vectorizado: ~100 000 latidos en menos de un segundo.
"""

import numpy as np
from scipy.stats import kurtosis, skew

from ecg import config
from ecg.segment import RR_FEATURE_NAMES

FS = config.FS
R = config.PRE_SAMPLES  # índice del pico R dentro de la ventana
WAVE_BINS = 21  # 252 / 12 muestras ≈ 33 ms por bin


def _ms(ms: float) -> int:
    return round(ms * FS / 1000)


def morphology_features(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Features de forma del latido: QRS, onda P, onda T y estadísticos de la ventana."""
    base = np.median(X, axis=1, keepdims=True)  # la mayor parte de la ventana es isoeléctrica
    a = np.abs(X - base)

    qrs = slice(R - _ms(100), R + _ms(100))
    p_region = slice(0, R - _ms(100))  # -250 a -100 ms
    t_region = slice(R + _ms(150), X.shape[1])  # +150 a +450 ms

    # Ancho aproximado del QRS: extensión donde la pendiente supera el 20 % de la máxima.
    d = np.abs(np.diff(X[:, R - _ms(100) : R + _ms(120)], axis=1))
    above = d > 0.2 * d.max(axis=1, keepdims=True)
    first = above.argmax(axis=1)
    last = above.shape[1] - 1 - above[:, ::-1].argmax(axis=1)
    qrs_width_ms = (last - first + 1) * 1000 / FS

    seg_qrs = X[:, qrs]
    offset = np.arange(seg_qrs.shape[1]) - _ms(100)
    feats = {
        "r_amp": X[:, R] - base[:, 0],
        "qrs_max": seg_qrs.max(axis=1),
        "qrs_min": seg_qrs.min(axis=1),
        "qrs_argmax_ms": offset[seg_qrs.argmax(axis=1)] * 1000 / FS,
        "qrs_argmin_ms": offset[seg_qrs.argmin(axis=1)] * 1000 / FS,
        "qrs_width_ms": qrs_width_ms,
        "qrs_area": a[:, qrs].sum(axis=1) / FS,
        "total_area": a.sum(axis=1) / FS,
        "qrs_energy_ratio": (a[:, qrs] ** 2).sum(axis=1) / ((a**2).sum(axis=1) + 1e-8),
        "p_std": X[:, p_region].std(axis=1),
        "p_max": X[:, p_region].max(axis=1) - base[:, 0],
        "t_std": X[:, t_region].std(axis=1),
        "t_max": X[:, t_region].max(axis=1) - base[:, 0],
        "t_min": X[:, t_region].min(axis=1) - base[:, 0],
        "skewness": skew(X, axis=1),
        "kurtosis": kurtosis(X, axis=1),
    }
    return np.column_stack(list(feats.values())).astype(np.float32), list(feats)


def waveform_features(X: np.ndarray, bins: int = WAVE_BINS) -> tuple[np.ndarray, list[str]]:
    """Latido submuestreado (promedio por bloques): la forma completa en pocas dimensiones."""
    n, w = X.shape
    step = w // bins
    wave = X[:, : bins * step].reshape(n, bins, step).mean(axis=2)
    names = [f"wave_{round((i * step + step / 2 - R) * 1000 / FS):+d}ms" for i in range(bins)]
    return wave.astype(np.float32), names


FEATURE_SETS = {
    "rr": ["rr"],
    "rr_ratios": ["rr_ratios"],
    "morph": ["morph"],
    "rr+morph": ["rr", "morph"],
    "rr_ratios+morph": ["rr_ratios", "morph"],
    "rr+morph+wave": ["rr", "morph", "wave"],
    "rr_ratios+morph+wave": ["rr_ratios", "morph", "wave"],
}


def build_features(
    X: np.ndarray, F: np.ndarray, feature_set: str = "rr+morph+wave"
) -> tuple[np.ndarray, list[str]]:
    """Matriz de features para un conjunto con nombre (ver FEATURE_SETS)."""
    blocks, names = [], []
    for part in FEATURE_SETS[feature_set]:
        if part == "rr":
            blocks.append(F), names.extend(RR_FEATURE_NAMES)
        elif part == "rr_ratios":
            # sin RR absolutos: solo cocientes, que generalizan mejor entre pacientes
            blocks.append(F[:, 2:]), names.extend(RR_FEATURE_NAMES[2:])
        elif part == "morph":
            m, n = morphology_features(X)
            blocks.append(m), names.extend(n)
        elif part == "wave":
            w, n = waveform_features(X)
            blocks.append(w), names.extend(n)
    return np.hstack(blocks).astype(np.float32), names
