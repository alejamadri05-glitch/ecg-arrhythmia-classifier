import numpy as np
import pytest

from ecg import config
from ecg.segment import beat_windows, extract_record

FS = 360
HAS_DATA = (config.DB_DIR / "234.atr").exists()


def test_beat_windows_shapes_and_edges():
    x = np.random.default_rng(0).normal(size=FS * 10)
    r = np.array([50, 400, 760, 1120, 1480, 3550])  # el primero y el último no tienen vecinos
    idx, X, F = beat_windows(x, FS, r)
    assert idx.tolist() == [1, 2, 3, 4]
    assert X.shape == (4, config.WINDOW)
    assert F.shape == (4, 4)


def test_beat_windows_skips_incomplete_window():
    x = np.zeros(1000)
    r = np.array([10, 60, 500, 950, 990])  # 60 y 950 no tienen ventana completa
    idx, _, _ = beat_windows(x, FS, r)
    assert idx.tolist() == [2]


def test_rr_features_detect_premature_beat():
    r = np.arange(1, 13) * FS  # ritmo regular a 1 s
    r[6] -= 144  # latido 6 llega 0.4 s antes (prematuro)
    x = np.zeros(r[-1] + FS)
    idx, _, F = beat_windows(x, FS, r)
    f = dict(zip(idx.tolist(), F, strict=True))
    pre_rr, post_rr, pre_over_local, post_over_pre = f[6]
    assert pre_rr == pytest.approx(0.6)
    assert post_rr == pytest.approx(1.4)
    assert pre_over_local < 0.7  # más corto que el ritmo local
    assert post_over_pre > 2  # pausa compensatoria
    assert f[3][2] == pytest.approx(1.0)  # latido regular


@pytest.mark.skipif(not HAS_DATA, reason="MIT-BIH no descargado")
def test_rhythm_annotations_record_201_has_afib():
    from ecg.data import rhythm_at

    out = extract_record(201)
    rhythms = set(rhythm_at(201, out["r_peak"]))
    assert {"(N", "(AFIB"} <= rhythms
    assert "?" not in rhythms  # todo latido cae dentro de un ritmo anotado


@pytest.mark.skipif(not HAS_DATA, reason="MIT-BIH no descargado")
def test_record_114_uses_mlii_and_labels_are_aami():
    out = extract_record(114)
    assert out["X"].shape[1] == config.WINDOW
    assert set(np.unique(out["y"])) <= set(config.CLASSES)
    assert (out["record"] == 114).all()
    assert len(out["y"]) == len(out["X"]) == len(out["F"]) == len(out["r_peak"])
