"""Pruebas de la calibración de umbrales por clase."""

import numpy as np

from ecg.calibrate import DEFAULT_GRID, apply_weights, tune_decision_weights
from ecg.evaluate import summary

CLASSES = ["N", "S", "V"]


def test_apply_weights_changes_the_decision():
    proba = np.array([[0.5, 0.4, 0.1]])
    assert apply_weights(proba, {}, CLASSES)[0] == "N"
    assert apply_weights(proba, {"S": 2.0}, CLASSES)[0] == "S"  # 0.4*2 > 0.5
    assert apply_weights(proba, {"N": 1.0, "S": 1.0, "V": 1.0}, CLASSES)[0] == "N"


def test_tuning_corrects_an_over_predicting_model():
    """Un modelo que marca S de más debe recibir un peso de S menor a 1."""
    y = np.array(["N"] * 900 + ["S"] * 100)
    proba = np.zeros((1000, 3))
    proba[:900] = [0.45, 0.50, 0.05]  # latidos N que el modelo llama S
    proba[900:] = [0.30, 0.65, 0.05]
    weights = tune_decision_weights(proba, y, CLASSES)
    assert weights["S"] < 1.0
    antes = summary(y, apply_weights(proba, {}, CLASSES), CLASSES)["macro_f1"]
    despues = summary(y, apply_weights(proba, weights, CLASSES), CLASSES)["macro_f1"]
    assert despues > antes


def test_tuning_leaves_a_good_model_alone():
    """Si el argmax ya es óptimo, los pesos no deben empeorarlo."""
    rng = np.random.default_rng(0)
    y = rng.choice(CLASSES, size=300)
    proba = np.full((300, 3), 0.1)
    proba[np.arange(300), [CLASSES.index(c) for c in y]] = 0.8
    weights = tune_decision_weights(proba, y, CLASSES)
    assert (apply_weights(proba, weights, CLASSES) == y).all()


def test_reference_class_weight_is_fixed_and_grid_is_positive():
    y = np.array(["N", "S", "V"] * 10)
    proba = np.repeat([[0.4, 0.35, 0.25]], 30, axis=0)
    assert tune_decision_weights(proba, y, CLASSES)["N"] == 1.0
    assert (DEFAULT_GRID > 0).all()
