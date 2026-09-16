"""Pruebas del adaptador de INCART (validación externa)."""

import numpy as np
import pytest

from ecg import config
from ecg.external import INCART_FS, INCART_LEAD, extract_record
from ecg.preprocess import resample

# La npz solo existe si `python -m ecg.external` terminó: evita archivos a medio bajar
HAS_INCART = (config.PROCESSED_DIR / "incart.npz").exists()


def test_incart_uses_lead_ii_at_257hz():
    assert INCART_LEAD == "II"  # INCART no tiene MLII
    assert INCART_FS == 257


def test_resampled_annotations_still_land_on_the_r_peak():
    """El reescalado de anotaciones 257 -> 360 Hz no debe desplazar el pico R."""
    n = INCART_FS * 20
    x = np.zeros(n)
    peaks = np.arange(1, 20) * INCART_FS  # un pico por segundo
    x[peaks] = 1.0
    y = resample(x, INCART_FS, config.FS)
    moved = np.round(peaks * config.FS / INCART_FS).astype(int)
    for original, nuevo in zip(peaks, moved, strict=True):
        ventana = y[nuevo - 3 : nuevo + 4]
        assert abs(int(np.argmax(ventana)) - 3) <= 1, f"pico {original} desplazado"
    assert len(y) == pytest.approx(n * config.FS / INCART_FS, rel=1e-3)


@pytest.mark.skipif(not HAS_INCART, reason="INCART no descargado")
def test_extract_record_matches_project_format():
    out = extract_record("I01")
    assert out["X"].shape[1] == config.WINDOW  # ya remuestreado a 360 Hz
    assert set(np.unique(out["y"])) <= set(config.CLASSES)
    assert len(out["y"]) == len(out["X"]) == len(out["F"]) == len(out["r_peak"])
    assert (out["record"] == 1).all()
