"""Modelos clásicos: Random Forest y XGBoost sobre features de `ecg.features`.

Los pesos de clase se calculan dentro de `fit`, con los datos que recibe. Así, dentro de una
validación cruzada, nunca ven la distribución del fold de validación.
"""

from dataclasses import dataclass, field

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from ecg import config

RF_DEFAULTS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
}
XGB_DEFAULTS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
}


def class_weights(y: np.ndarray, power: float = 1.0) -> np.ndarray:
    """Peso por muestra: ("balanced" = n / (n_clases · n_c)) ** power, normalizado a media 1.

    Se calcula únicamente con las etiquetas que recibe (las de entrenamiento).
    """
    w = compute_sample_weight("balanced", y) ** power
    return w / w.mean()


@dataclass
class BaselineClassifier:
    """Envoltorio común para RF y XGBoost con etiquetas de texto (N/S/V/F)."""

    kind: str = "xgb"  # "rf" | "xgb"
    params: dict = field(default_factory=dict)
    feature_set: str = "rr+morph+wave"
    # Tamaño del bloque para la plantilla del paciente (features "rel"/"wave_rel"); None = global.
    # Va en el modelo para que quede autocontenido: quien lo carga sabe cómo construir sus features.
    template_block: int | None = None
    # Exponente de los pesos "balanced": 1 = inverso a la frecuencia, 0.5 = raíz, 0 = sin pesos
    weight_power: float = 1.0
    seed: int = config.SEED
    classes: list[str] = field(default_factory=lambda: list(config.CLASSES))
    # Umbrales calibrados: cada probabilidad se multiplica por el peso de su clase antes del
    # argmax. None = argmax normal. Ver ecg.calibrate.
    decision_weights: dict[str, float] | None = None

    def _make(self):
        if self.kind == "rf":
            p = {**RF_DEFAULTS, **self.params}
            return RandomForestClassifier(**p, n_jobs=-1, random_state=self.seed)
        if self.kind == "xgb":
            p = {**XGB_DEFAULTS, **self.params}
            return XGBClassifier(
                **p,
                objective="multi:softprob",
                tree_method="hist",
                eval_metric="mlogloss",
                n_jobs=-1,
                random_state=self.seed,
            )
        raise ValueError(f"Modelo desconocido: {self.kind}")

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "BaselineClassifier":
        lookup = {c: i for i, c in enumerate(self.classes)}
        y_idx = np.array([lookup[v] for v in y], dtype=np.int64)
        # XGBoost exige etiquetas 0..k-1: si falta una clase en este train, se re-indexa.
        self.seen_ = np.unique(y_idx)
        y_fit = np.searchsorted(self.seen_, y_idx)
        self.model_ = self._make()
        self.model_.fit(Z, y_fit, sample_weight=class_weights(y_fit, self.weight_power))
        return self

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        """Probabilidades (n, 4) en el orden de `classes`, aunque falte una clase en el train."""
        out = np.zeros((len(Z), len(self.classes)), dtype=np.float32)
        out[:, self.seen_] = self.model_.predict_proba(Z)
        return out

    def predict(self, Z: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(Z)
        if self.decision_weights:
            w = np.array([self.decision_weights.get(c, 1.0) for c in self.classes])
            proba = proba * w
        return np.asarray(self.classes)[proba.argmax(axis=1)]

    def feature_importances(self) -> np.ndarray:
        return self.model_.feature_importances_

    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "BaselineClassifier":
        return joblib.load(path)
