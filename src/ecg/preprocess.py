"""Filtrado de señal."""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample_poly

from ecg import config


def bandpass(
    x: np.ndarray,
    fs: float,
    lo: float = config.BANDPASS_LO,
    hi: float = config.BANDPASS_HI,
    order: int = config.BANDPASS_ORDER,
) -> np.ndarray:
    """Butterworth pasa-banda con filtfilt (fase cero: no desplaza el pico R)."""
    nyq = fs / 2
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x)


def notch(x: np.ndarray, fs: float, f0: float = 60.0, q: float = 30.0) -> np.ndarray:
    """Notch opcional para interferencia de red eléctrica."""
    b, a = iirnotch(f0, q, fs)
    return filtfilt(b, a, x)


def resample(x: np.ndarray, fs_in: float, fs_out: float = config.FS) -> np.ndarray:
    """Remuestreo polifásico a la frecuencia del modelo."""
    if fs_in == fs_out:
        return x
    from fractions import Fraction

    frac = Fraction(fs_out / fs_in).limit_denominator(1000)
    return resample_poly(x, frac.numerator, frac.denominator)


def zscore(seg: np.ndarray) -> np.ndarray:
    return (seg - seg.mean()) / (seg.std() + 1e-8)
