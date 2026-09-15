import numpy as np
import pytest

from ecg.models.baseline import BaselineClassifier, class_weights
from ecg.train import cross_validate, patient_folds


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 600
    y = rng.choice(["N", "S", "V", "F"], size=n, p=[0.7, 0.1, 0.15, 0.05])
    Z = rng.normal(size=(n, 5)).astype(np.float32)
    Z[:, 0] += (y == "V") * 3
    Z[:, 1] += (y == "S") * 3
    Z[:, 2] += (y == "F") * 3
    groups = np.repeat(np.arange(12), n // 12)
    return Z, y, groups


def test_patient_folds_never_share_a_record(toy):
    _, _, groups = toy
    seen = []
    for tr, va in patient_folds(groups):
        assert set(groups[tr]).isdisjoint(groups[va])
        seen.extend(np.unique(groups[va]))
    assert sorted(seen) == list(range(12))  # cada paciente se valida exactamente una vez


def test_balanced_weights_equalize_total_weight_per_class():
    y = np.array([0] * 90 + [1] * 10)
    w = class_weights(y, power=1.0)
    assert w[y == 0].sum() == pytest.approx(w[y == 1].sum())
    assert np.allclose(class_weights(y, power=0.0), 1.0)


@pytest.mark.parametrize("kind", ["xgb", "rf"])
def test_missing_class_in_training_still_returns_4_columns(toy, kind):
    Z, y, _ = toy
    keep = y != "F"
    model = BaselineClassifier(kind=kind, params={"n_estimators": 20}).fit(Z[keep], y[keep])
    proba = model.predict_proba(Z)
    assert proba.shape == (len(Z), 4)
    assert np.all(proba[:, 3] == 0)  # F nunca vista
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)


@pytest.mark.parametrize("kind", ["xgb", "rf"])
def test_same_seed_same_predictions(toy, kind):
    Z, y, groups = toy

    def make():
        return BaselineClassifier(kind=kind, params={"n_estimators": 30})

    a = cross_validate(make, Z, y, groups)
    b = cross_validate(make, Z, y, groups)
    np.testing.assert_array_equal(a["proba"], b["proba"])
    assert (a["fold"] >= 0).all()  # todos los latidos recibieron predicción out-of-fold


def test_save_and_load_roundtrip(toy, tmp_path):
    Z, y, _ = toy
    model = BaselineClassifier(kind="xgb", params={"n_estimators": 20}).fit(Z, y)
    model.save(tmp_path / "m.joblib")
    loaded = BaselineClassifier.load(tmp_path / "m.joblib")
    np.testing.assert_array_equal(loaded.predict(Z), model.predict(Z))
