import numpy as np
import pytest

from ecg import config
from ecg.features import (
    FEATURE_SETS,
    build_features,
    morphology_features,
    normalized_rr,
    patient_template,
    relative_morphology_features,
    sequence_features,
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


def test_patient_template_is_the_median_beat_per_record():
    X = np.array([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0], [10.0, 10.0], [20.0, 20.0]])
    records = np.array([1, 1, 1, 2, 2])
    tpl = patient_template(X, records)
    np.testing.assert_allclose(tpl[:3], 3.0)  # mediana del registro 1
    np.testing.assert_allclose(tpl[3:], 15.0)  # mediana del registro 2
    np.testing.assert_allclose(patient_template(X), 5.0)  # sin records: una sola grabación


def test_patient_template_options_are_exclusive_and_local():
    X = np.arange(12, dtype=float).reshape(6, 2)
    records = np.zeros(6, dtype=int)
    primeros = patient_template(X, records, max_beats=2)
    np.testing.assert_allclose(primeros[0], [1.0, 2.0])  # mediana de los 2 primeros latidos
    bloques = patient_template(X, records, block=2)
    np.testing.assert_allclose(bloques[0], [1.0, 2.0])
    np.testing.assert_allclose(bloques[4], [9.0, 10.0])  # el último bloque usa su propia mediana
    with pytest.raises(ValueError):
        patient_template(X, records, max_beats=2, block=2)


def test_relative_features_are_invariant_to_the_patients_own_morphology():
    """Dos pacientes con morfologías distintas, pero el mismo latido anómalo *relativo* a cada
    uno, deben dar las mismas features."""
    normal_a, normal_b = synthetic_beat(70), -synthetic_beat(70)  # el B tiene el QRS invertido
    raro_a, raro_b = synthetic_beat(150), -synthetic_beat(150)
    X = np.vstack(
        [np.tile(normal_a, (20, 1)), raro_a[None], np.tile(normal_b, (20, 1)), raro_b[None]]
    )
    records = np.array([1] * 21 + [2] * 21)
    feats, names = relative_morphology_features(X.astype(np.float32), records)
    a, b = feats[20], feats[41]
    firmada = names.index("r_amp_menos_plantilla")  # conserva el signo a propósito
    sin_signo = [i for i in range(len(names)) if i != firmada]
    np.testing.assert_allclose(a[sin_signo], b[sin_signo], rtol=1e-3)
    assert a[firmada] == pytest.approx(-b[firmada], rel=1e-3)  # se invierte con la polaridad
    corr = feats[:, names.index("corr_plantilla")]
    assert corr[:20].min() > 0.99 and corr[20] < corr[0]  # el raro se parece menos a su plantilla


def test_relative_features_detect_a_wider_beat():
    X = np.vstack([np.tile(synthetic_beat(70), (20, 1)), synthetic_beat(160)[None]]).astype(
        np.float32
    )
    feats, names = relative_morphology_features(X, np.zeros(21, dtype=int))
    ancho = feats[:, names.index("qrs_ancho_sobre_plantilla")]
    assert ancho[:20].max() < 1.2 and ancho[20] > 1.5


def synthetic_rr(pattern: list[float]) -> np.ndarray:
    """Matriz F (pre_rr, post_rr, cocientes) a partir de una secuencia de intervalos RR."""
    rr = np.array(pattern, dtype=np.float32)
    post = np.append(rr[1:], rr[-1])
    return np.column_stack([rr, post, np.ones_like(rr), np.ones_like(rr)]).astype(np.float32)


def test_sequence_features_flag_an_isolated_premature_beat():
    F = synthetic_rr([0.8] * 30 + [0.5] + [1.1] + [0.8] * 10)
    X = np.tile(synthetic_beat(70), (len(F), 1)).astype(np.float32)
    ctx, names = sequence_features(X, F, np.zeros(len(F), dtype=int))
    corto = ctx[:, names.index("rr_sobre_referencia_larga")]
    assert corto[30] < 0.7 and abs(corto[10] - 1.0) < 0.05
    assert ctx[30, names.index("latidos_desde_arranque_racha")] == 0


def test_long_reference_still_sees_a_run_as_fast():
    """Dentro de una racha, la referencia local se adapta pero la larga no."""
    F = synthetic_rr([0.8] * 200 + [0.45] * 40 + [0.8] * 20)  # racha larga: la local se adapta
    X = np.tile(synthetic_beat(70), (len(F), 1)).astype(np.float32)
    ctx, names = sequence_features(X, F, np.zeros(len(F), dtype=int))
    dentro = 230  # 30 latidos dentro de la racha
    local = ctx[dentro, names.index("rr_sobre_mediana_movil")]
    larga = ctx[dentro, names.index("rr_sobre_referencia_larga")]
    assert larga < 0.7 < local  # la larga la ve rápida; la local ya se adaptó
    assert ctx[dentro, names.index("frac_rr_cortos_recientes")] > 0.5
    desde = ctx[:, names.index("latidos_desde_arranque_racha")]
    assert desde[200] == 0 and desde[210] == 10  # cuenta desde el arranque de la racha
    assert desde[199] == 20  # en ritmo normal se queda en el tope


def test_sequence_features_do_not_cross_records():
    F = synthetic_rr([0.8] * 20 + [0.4] * 20)
    X = np.tile(synthetic_beat(70), (len(F), 1)).astype(np.float32)
    records = np.array([1] * 20 + [2] * 20)
    ctx, names = sequence_features(X, F, records)
    # el primer latido del registro 2 no puede "ver" el ritmo del registro 1
    largo = ctx[:, names.index("rr_sobre_referencia_larga")]
    assert abs(largo[20] - 1.0) < 0.2
    assert ctx[20, names.index("corr_latido_previo")] == 1.0  # sin vecino previo en su registro


def test_neighbour_correlation_detects_an_odd_beat():
    X = np.tile(synthetic_beat(70), (10, 1)).astype(np.float32)
    X[5] = synthetic_beat(160)  # un latido ancho entre latidos normales
    F = synthetic_rr([0.8] * 10)
    ctx, names = sequence_features(X, F, np.zeros(10, dtype=int))
    prev = ctx[:, names.index("corr_latido_previo")]
    assert prev[5] < 0.9 and prev[4] > 0.99
