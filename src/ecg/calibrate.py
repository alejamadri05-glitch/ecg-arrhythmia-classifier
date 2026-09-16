"""Calibración del umbral de decisión por clase.

El `argmax` de las probabilidades trata a las cuatro clases por igual, y con pesos de clase
balanceados el modelo termina sobre-prediciendo las raras: en INCART, la v2 predice S 2.3 veces
más de lo que existe y F 4 veces. Multiplicar cada probabilidad por un peso antes del `argmax`
mueve el punto de operación sin reentrenar nada.

Los pesos se ajustan **siempre** con datos que el modelo evaluado no vio (ver
`train.cross_validate_calibrated`, que usa validación anidada).
"""

from itertools import product

import numpy as np

from ecg import config
from ecg.evaluate import summary

DEFAULT_GRID = np.round(np.logspace(-1.0, 0.6, 9), 3)  # de 0.1 a ~4


def apply_weights(proba: np.ndarray, weights: dict[str, float], classes: list[str]) -> np.ndarray:
    """Predicción con umbrales calibrados: argmax de (probabilidad × peso de la clase)."""
    w = np.array([weights.get(c, 1.0) for c in classes], dtype=np.float64)
    return np.asarray(classes)[(proba * w).argmax(axis=1)]


def tune_decision_weights(
    proba: np.ndarray,
    y_true: np.ndarray,
    classes: list[str] | None = None,
    grid: np.ndarray = DEFAULT_GRID,
    metric: str = "macro_f1",
    min_sensitivity: dict[str, float] | None = None,
) -> dict[str, float]:
    """Busca los pesos que maximizan la métrica. La clase de referencia queda fija en 1.0.

    `min_sensitivity` impone un piso de sensibilidad por clase (p. ej. `{"V": 0.90}`): sin esa
    restricción, maximizar el F1 macro puede empeorar el error más grave del análisis de riesgo
    (latidos ventriculares leídos como normales) a cambio de ganar en otra clase.

    `proba` e `y_true` tienen que venir de datos que el modelo evaluado no vio.
    """
    classes = list(classes or config.CLASSES)
    pisos = min_sensitivity or {}

    def evaluar(weights: dict[str, float]) -> float | None:
        s = summary(y_true, apply_weights(proba, weights, classes), classes)
        for cls, minimo in pisos.items():
            se = s.get(f"{cls}_Se")
            if se is None or se < minimo:
                return None
        return s[metric]

    best_weights = dict.fromkeys(classes, 1.0)
    best_score = evaluar(best_weights)
    for combo in product(grid, repeat=len(classes) - 1):
        weights = {classes[0]: 1.0, **dict(zip(classes[1:], combo, strict=True))}
        score = evaluar(weights)
        if score is not None and (best_score is None or score > best_score):
            best_score, best_weights = score, weights
    return best_weights
