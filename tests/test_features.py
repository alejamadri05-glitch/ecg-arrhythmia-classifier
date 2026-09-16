import numpy as np
import pytest

from ecg import config
from ecg.features import (
    FEATURE_SETS,
    build_features,
    morphology_features,
    normalized_rr,
    waveform_features,
)
from ecg.preprocess import zscore

R = config.PRE_SAMPLES


def synthetic_beat(qrs_ms: float) -> np.ndarray:
    """Latido gaussiano centrado en R con un ancho de QRS dado."""
    t = (np.arange(config.WINDOW) - R) / config.FS * 1000
    return zscore(np.exp(-0.5 * (t / (qrs_ms / 6)) ** 2))


@pytest.fixture
def batch():
    rng = np.random.default_rng(0)
    X = np.stack([synthetic_beat(w) for w in rng.uniform(60, 160, 20)]).astype(np.float32)
    F = rng.uniform(0.4, 1.5, (20, 4)).astype(np.float32)
    return X, F


@pytest.mark.parametrize("feature_set", list(FEATURE_SETS))
def test_feature_sets_shapes_and_finite(batch, feature_set):
    X, F = batch
    Z, names = build_features(X, F, feature_set)
    assert Z.shape == (len(X), len(names))
    assert len(set(names)) == len(names)
    assert np.isfinite(Z).all()


def test_rr_ratios_excludes_absolute_rr(batch):
    _, names = build_features(*batch, "rr_ratios")
    assert "pre_rr" not in names and "post_rr" not in names


def test_wide_qrs_measures_wider_than_narrow():
    X = np.stack([synthetic_beat(70), synthetic_beat(140)])
    m, names = morphology_features(X)
    width = m[:, names.index("qrs_width_ms")]
    assert width[1] > width[0] * 1.5


def test_waveform_bins():
    X = np.random.default_rng(0).normal(size=(3, config.WINDOW))
    w, names = waveform_features(X)
    assert w.shape == (3, 21) and len(names) == 21
    np.testing.assert_allclose(w[:, 0], X[:, :12].mean(axis=1), rtol=1e-5)


def test_normalized_rr_is_invariant_to_heart_rate(batch):
    """Un paciente con el mismo patrón pero más lento da las mismas features normalizadas."""
    X, F = batch
    slow = F.copy()
    slow[:, :2] *= 2.0  # todos los intervalos al doble (p. ej. 80 lpm -> 40 lpm)
    a, names = build_features(X, F, "rr_norm+morph+wave")
    b, _ = build_features(X, slow, "rr_norm+morph+wave")
    np.testing.assert_allclose(a, b, rtol=1e-5)
    assert "pre_rr" not in names and "pre_rr_over_median" in names


def test_normalized_rr_uses_each_record_separately():
    F = np.array([[1.0, 1.0, 1, 1], [2.0, 2.0, 1, 1], [0.5, 0.5, 1, 1], [1.0, 1.0, 1, 1]])
    records = np.array([1, 1, 2, 2])
    out, _ = normalized_rr(F, records)
    np.testing.assert_allclose(out[:, 0], [1 / 1.5, 2 / 1.5, 0.5 / 0.75, 1 / 0.75], rtol=1e-5)
    todos_juntos, _ = normalized_rr(F)  # sin `records`: una sola grabación (caso de la API)
    np.testing.assert_allclose(todos_juntos[:, 0], F[:, 0] / 1.0, rtol=1e-5)
