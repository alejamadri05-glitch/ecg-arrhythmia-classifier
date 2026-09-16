"""Entrenamiento con validación cruzada inter-paciente dentro de DS1.

DS2 no se usa en ningún lugar de este módulo: se evalúa una sola vez en la fase final.

Uso:  python -m ecg.train baseline [--kind xgb|rf]
      python -m ecg.train baseline-v2
      python -m ecg.train cnn [--skip-cv]
"""

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from ecg import config
from ecg.evaluate import per_class_metrics, summary
from ecg.features import build_features
from ecg.models.baseline import BaselineClassifier
from ecg.models.cnn import MODEL_VERSION, CNNClassifier, CNNConfig
from ecg.segment import load_split

N_FOLDS = 5

# Configuración elegida con la validación cruzada de notebooks/02_baseline.ipynb
# (regla fijada de antemano: mayor F1 macro out-of-fold en DS1)
BASELINE_CHOICE = {
    "kind": "xgb",
    "feature_set": "rr+morph+wave",
    "weight_power": 1.0,
    "params": {},  # hiperparámetros por defecto de models/baseline.py
}

# Versión 2 del baseline (notebooks/05_model_v2.ipynb): RR normalizado por la mediana del
# registro. Validada solo en DS1: DS2 ya se usó y no se vuelve a abrir.
BASELINE_V2_CHOICE = {
    "kind": "xgb",
    "feature_set": "rr_norm+morph+wave",
    "weight_power": 1.0,
    "params": {},
}

# Configuración elegida en notebooks/03_cnn.ipynb (mayor F1 macro media en 2 semillas)
CNN_CHOICE = {"rr": "ratios", "balance": "weights", "augment": True, "invert": False}


def patient_folds(groups: np.ndarray, n_splits: int = N_FOLDS):
    """Folds por registro: todos los latidos de un paciente caen del mismo lado."""
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups))


def cross_validate(
    make_model: Callable[[], object],
    Z: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = N_FOLDS,
) -> dict[str, np.ndarray]:
    """Predicciones out-of-fold: cada latido lo predice un modelo que no vio a su paciente.

    Las métricas se calculan sobre todas las predicciones juntas (estadística "gross" de AAMI),
    porque la clase F está casi entera en un solo registro y un F1 por fold no estaría definido.
    """
    classes = list(config.CLASSES)
    proba = np.zeros((len(y), len(classes)), dtype=np.float32)
    fold = np.full(len(y), -1, dtype=np.int8)
    for k, (tr, va) in enumerate(patient_folds(groups, n_splits)):
        assert set(groups[tr]).isdisjoint(groups[va])
        model = make_model().fit(Z[tr], y[tr])
        proba[va] = model.predict_proba(Z[va])
        fold[va] = k
    return {"proba": proba, "pred": np.asarray(classes)[proba.argmax(axis=1)], "fold": fold}


def train_baseline(
    kind: str,
    feature_set: str,
    params: dict,
    weight_power: float = 1.0,
    model_name: str | None = None,
) -> None:
    ds1 = load_split("ds1")
    y, groups = ds1["y"], ds1["record"]
    # `records` solo lo usan los conjuntos con RR normalizado por registro
    Z, names = build_features(ds1["X"], ds1["F"], feature_set, records=groups)
    model_name = model_name or f"baseline_{kind}"

    def make():
        return BaselineClassifier(
            kind=kind, params=params, feature_set=feature_set, weight_power=weight_power
        )

    t0 = time.time()
    cv = cross_validate(make, Z, y, groups)
    cv_summary = summary(y, cv["pred"])
    print(f"CV {N_FOLDS} folds por paciente ({time.time() - t0:.0f} s)")
    print(per_class_metrics(y, cv["pred"]).round(3).to_string())
    print(f"F1 macro: {cv_summary['macro_f1']:.3f}")

    model = make().fit(Z, y)  # modelo final: todo DS1
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    path = config.MODELS_DIR / f"{model_name}.joblib"
    model.save(path)
    meta = {
        "model": model_name,
        "feature_set": feature_set,
        "feature_names": names,
        "params": params,
        "weight_power": weight_power,
        "train_records": sorted(int(r) for r in np.unique(groups)),
        "seed": config.SEED,
        "cv": {"n_folds": N_FOLDS, "grouping": "record", **cv_summary},
    }
    (config.REPORTS_DIR / f"{model_name}_cv.json").write_text(json.dumps(meta, indent=2))
    print(f"Modelo -> {path}")


def cross_validate_cnn(
    cfg: CNNConfig,
    X: np.ndarray,
    F: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = N_FOLDS,
) -> dict:
    """Igual que `cross_validate`, con los mismos folds externos, para la CNN.

    Dentro de cada fold, la CNN separa pacientes internos para el early stopping: el fold
    externo solo se usa para predecir.
    """
    classes = list(config.CLASSES)
    proba = np.zeros((len(y), len(classes)), dtype=np.float32)
    fold = np.full(len(y), -1, dtype=np.int8)
    best_epochs, histories, inner_train_records = [], [], []
    for k, (tr, va) in enumerate(patient_folds(groups, n_splits)):
        assert set(groups[tr]).isdisjoint(groups[va])
        model = CNNClassifier(cfg).fit(X[tr], F[tr], y[tr], groups=groups[tr])
        proba[va] = model.predict_proba(X[va], F[va])
        fold[va] = k
        best_epochs.append(model.best_epoch_)
        histories.append(model.history_)
        inner_train_records.append(sorted(int(r) for r in np.unique(groups[tr][model.train_idx_])))
    return {
        "proba": proba,
        "pred": np.asarray(classes)[proba.argmax(axis=1)],
        "fold": fold,
        "best_epochs": best_epochs,
        "histories": histories,
        "inner_train_records": inner_train_records,
    }


def fit_final_cnn(cfg: CNNConfig, ds1: dict, cv_summary: dict | None = None) -> Path:
    """Modelo final con el mismo protocolo que la validación: early stopping en pacientes
    internos de DS1. Lo que se valida es lo que se entrega."""
    model = CNNClassifier(cfg).fit(ds1["X"], ds1["F"], ds1["y"], groups=ds1["record"])
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    path = config.MODELS_DIR / "cnn.pt"
    model.save(path)
    g = ds1["record"]
    meta = {
        "model": "cnn",
        "model_version": MODEL_VERSION,
        "config": asdict(cfg),
        "best_epoch": model.best_epoch_,
        "train_records": sorted(int(r) for r in np.unique(g[model.train_idx_])),
        "early_stopping_records": sorted(int(r) for r in np.unique(g[model.val_idx_])),
        "history": model.history_,
        "cv": cv_summary,
    }
    (config.REPORTS_DIR / "cnn_cv.json").write_text(json.dumps(meta, indent=2))
    return path


def train_cnn(cfg: CNNConfig, skip_cv: bool = False) -> None:
    ds1 = load_split("ds1")
    cv_summary = None
    if not skip_cv:
        t0 = time.time()
        cv = cross_validate_cnn(cfg, ds1["X"], ds1["F"], ds1["y"], ds1["record"])
        cv_summary = {"n_folds": N_FOLDS, "grouping": "record", "seeds": [cfg.seed],
                      **summary(ds1["y"], cv["pred"])}  # fmt: skip
        print(f"CV {N_FOLDS} folds por paciente ({time.time() - t0:.0f} s)")
        print(per_class_metrics(ds1["y"], cv["pred"]).round(3).to_string())
        print(f"F1 macro: {cv_summary['macro_f1']:.3f} | mejores épocas: {cv['best_epochs']}")
    print(f"Modelo -> {fit_final_cnn(cfg, ds1, cv_summary)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["baseline", "baseline-v2", "cnn"])
    parser.add_argument("--kind", choices=["xgb", "rf"], default=BASELINE_CHOICE["kind"])
    parser.add_argument("--feature-set", default=BASELINE_CHOICE["feature_set"])
    parser.add_argument("--skip-cv", action="store_true", help="CNN: solo el modelo final")
    args = parser.parse_args()
    if args.model == "cnn":
        train_cnn(CNNConfig(**CNN_CHOICE), skip_cv=args.skip_cv)
        return
    if args.model == "baseline-v2":
        c = BASELINE_V2_CHOICE
        train_baseline(c["kind"], c["feature_set"], c["params"], c["weight_power"], "baseline_v2")
        return
    params = BASELINE_CHOICE["params"] if args.kind == BASELINE_CHOICE["kind"] else {}
    train_baseline(args.kind, args.feature_set, params, BASELINE_CHOICE["weight_power"])


if __name__ == "__main__":
    main()
