"""Pruebas del módulo de evaluación final.

Usan DS1 a propósito: DS2 no se toca fuera de la evaluación final.
"""

import json

import numpy as np
import pytest

from ecg import config
from ecg.evaluate_ds2 import predict_all, rhythm_labels, risk_checks
from ecg.segment import load_split

HAS_DATA = (config.PROCESSED_DIR / "ds1.npz").exists()
HAS_MODELS = (config.MODELS_DIR / "baseline_xgb.joblib").exists() and (
    config.MODELS_DIR / "cnn.pt"
).exists()
needs_all = pytest.mark.skipif(
    not (HAS_DATA and HAS_MODELS), reason="hacen falta los datos y los modelos entrenados"
)


@pytest.fixture(scope="module")
def sample():
    ds = load_split("ds1")
    idx = np.linspace(0, len(ds["y"]) - 1, 300).astype(int)
    return {k: v[idx] for k, v in ds.items()}


@needs_all
def test_predict_all_returns_valid_classes(sample):
    preds = predict_all(sample)
    assert set(preds) == {"baseline_xgb", "cnn"}
    for p in preds.values():
        assert len(p) == len(sample["y"])
        assert set(np.unique(p)) <= set(config.CLASSES)


@needs_all
def test_predictions_are_deterministic(sample):
    np.testing.assert_array_equal(predict_all(sample)["cnn"], predict_all(sample)["cnn"])


@pytest.mark.skipif(not HAS_DATA, reason="MIT-BIH no descargado")
def test_rhythm_labels_cover_every_beat(sample):
    labels = rhythm_labels(sample)
    assert len(labels) == len(sample["y"])
    assert "?" not in set(labels)  # todo latido cae dentro de un ritmo anotado


@needs_all
def test_risk_checks_are_json_serializable(sample):
    checks = risk_checks(sample, predict_all(sample))
    assert {"lbbb", "V_no_detectados", "por_ritmo"} <= set(checks)
    json.dumps(checks)  # no deben quedar tipos de numpy
