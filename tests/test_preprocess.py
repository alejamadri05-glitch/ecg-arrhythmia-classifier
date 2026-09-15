import numpy as np
import pytest

from ecg.data import lead_index
from ecg.preprocess import bandpass, resample, zscore

FS = 360


def test_bandpass_preserves_length():
    x = np.random.default_rng(0).normal(size=FS * 10)
    assert bandpass(x, FS).shape == x.shape


def test_bandpass_removes_baseline_wander_and_keeps_qrs_band():
    t = np.arange(FS * 60) / FS
    wander = np.sin(2 * np.pi * 0.1 * t)  # 0.1 Hz: debe desaparecer
    qrs_band = np.sin(2 * np.pi * 10 * t)  # 10 Hz: debe conservarse
    core = slice(FS * 10, -FS * 10)  # ignora el transitorio de los bordes
    assert np.abs(bandpass(wander, FS)[core]).max() < 0.01
    assert np.abs(bandpass(qrs_band, FS)[core]).max() == pytest.approx(1.0, abs=0.01)


def test_bandpass_is_zero_phase():
    """filtfilt no debe desplazar el pico R."""
    x = np.zeros(FS * 4)
    x[FS * 2] = 1.0
    assert np.argmax(bandpass(x, FS)) == FS * 2


def test_resample_changes_length_proportionally():
    x = np.random.default_rng(0).normal(size=257 * 10)  # INCART: 257 Hz
    assert len(resample(x, 257, 360)) == 3600
    assert resample(x, 360, 360) is x


def test_zscore():
    z = zscore(np.array([1.0, 2.0, 3.0, 4.0]))
    assert z.mean() == pytest.approx(0, abs=1e-7)
    assert z.std() == pytest.approx(1, abs=1e-6)


def test_lead_is_found_by_name_not_index():
    assert lead_index(["V5", "MLII"]) == 1  # como en el registro 114
    with pytest.raises(ValueError):
        lead_index(["V1", "V2"])
