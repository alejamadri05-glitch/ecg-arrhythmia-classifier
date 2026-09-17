"""Entrenamiento con validación cruzada inter-paciente dentro de DS1.

DS2 no se usa en ningún lugar de este módulo: se evalúa una sola vez en la fase final.

Uso:  python -m ecg.train baseline [--kind xgb|rf]
      python -m ecg.train baseline-v2
      python -m ecg.train baseline-v3   # 3 clases
      python -m ecg.train baseline-v4   # 3 clases + morfología relativa al paciente
      python -m ecg.train baseline-v5   # v4 + contexto de la secuencia de latidos
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
from ecg.calibrate import DEFAULT_GRID, apply_weights, tune_decision_weights
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

# Versión 3 (notebooks/07_calibration.ipynb): 3 clases (F fuera de alcance) + umbrales
# calibrados. Validada solo en DS1.
CLASSES_V3 = ["N", "S", "V"]
BASELINE_V3_CHOICE = {
    "kind": "xgb",
    "feature_set": "rr_norm+morph+wave",
    "weight_power": 1.0,
    "params": {},
    # Sin calibración: en notebooks/07_calibration.ipynb la calibración gana 0.005 de F1 macro
    # pero deja 79 latidos V sin detectar (+26 %), y sus pesos son inestables entre folds.
    "calibrate": False,
}

# Versión 4 (notebooks/08_patient_relative.ipynb): morfología relativa al latido dominante del
# paciente, con plantilla local recalculada cada 100 latidos. Validada solo en DS1.
BASELINE_V4_CHOICE = {
    "kind": "xgb",
    "feature_set": "rr_norm+morph+rel+wave",
    "weight_power": 1.0,
    "params": {},
    "template_block": 100,
}

# Versión 5 (notebooks/10_sequence_context.ipynb): v4 + contexto de la secuencia de latidos,
# para los latidos S que vienen en racha. Validada solo en DS1.
BASELINE_V5_CHOICE = {
    "kind": "xgb",
    "feature_set": "rr_norm+morph+rel+wave+ctx",
    "weight_power": 1.0,
    "params": {},
    "template_block": 100,
}

# Configuración elegida en notebooks/03_cnn.ipynb (mayor F1 macro media en 2 semillas)
CNN_CHOICE = {"rr": "ratios", "balance": "weights", "augment": True, "invert": False}


def patient_folds(
    groups: np.ndarray, n_splits: int = N_FOLDS, fold_map: dict[int, int] | None = None
):
    """Folds por registro: todos los latidos de un paciente caen del mismo lado.

    `fold_map` (p. ej. `config.DS1_FOLD_MAP`) fija a qué fold va cada paciente. Es necesario al
    comparar experimentos sobre subconjuntos distintos de latidos: GroupKFold equilibra por
    cantidad de latidos, así que filtrar una clase cambiaría el reparto de pacientes.
    """
    if fold_map is None:
        return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups))
    faltantes = {int(r) for r in np.unique(groups)} - set(fold_map)
    if faltantes:
        raise ValueError(f"Registros sin fold asignado: {sorted(faltantes)}")
    fold_of = np.array([fold_map[int(r)] for r in groups])
    folds = sorted(set(fold_map.values()))
    return [(np.flatnonzero(fold_of != k), np.flatnonzero(fold_of == k)) for k in folds]


def cross_validate(
    make_model: Callable[[], object],
    Z: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = N_FOLDS,
    fold_map: dict[int, int] | None = None,
    classes: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Predicciones out-of-fold: cada latido lo predice un modelo que no vio a su paciente.

    Las métricas se calculan sobre todas las predicciones juntas (estadística "gross" de AAMI),
    porque la clase F está casi entera en un solo registro y un F1 por fold no estaría definido.
    """
    classes = list(classes or config.CLASSES)
    proba = np.zeros((len(y), len(classes)), dtype=np.float32)
    fold = np.full(len(y), -1, dtype=np.int8)
    for k, (tr, va) in enumerate(patient_folds(groups, n_splits, fold_map)):
        assert set(groups[tr]).isdisjoint(groups[va])
        model = make_model().fit(Z[tr], y[tr])
        proba[va] = model.predict_proba(Z[va])
        fold[va] = k
    return {"proba": proba, "pred": np.asarray(classes)[proba.argmax(axis=1)], "fold": fold}


def cross_validate_calibrated(
    make_model: Callable[[], object],
    Z: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    classes: list[str],
    n_splits: int = N_FOLDS,
    inner_splits: int = 4,
    grid: np.ndarray = DEFAULT_GRID,
    fold_map: dict[int, int] | None = None,
) -> dict:
    """Out-of-fold con umbrales calibrados, en **validación anidada**.

    Dentro de cada fold de entrenamiento se hace otra validación por paciente; los umbrales se
    ajustan con esas predicciones internas y recién después se aplican al fold externo, que no
    participó ni del entrenamiento ni de la calibración.
    """
    proba = np.zeros((len(y), len(classes)), dtype=np.float32)
    pred = np.empty(len(y), dtype="<U1")
    fold_weights = []
    for tr, va in patient_folds(groups, n_splits, fold_map):
        inner_proba = np.zeros((len(tr), len(classes)), dtype=np.float32)
        for itr, iva in patient_folds(groups[tr], inner_splits):
            inner_proba[iva] = make_model().fit(Z[tr][itr], y[tr][itr]).predict_proba(Z[tr][iva])
        weights = tune_decision_weights(inner_proba, y[tr], classes, grid)
        model = make_model().fit(Z[tr], y[tr])
        proba[va] = model.predict_proba(Z[va])
        pred[va] = apply_weights(proba[va], weights, classes)
        fold_weights.append(weights)
    return {"proba": proba, "pred": pred, "weights": fold_weights}


def train_baseline_v3(calibrate: bool = BASELINE_V3_CHOICE["calibrate"]) -> None:
    """3 clases (N/S/V): la clase F queda fuera del alcance. Todo se decide dentro de DS1.

    `calibrate` activa los umbrales por clase (ver notebooks/07_calibration.ipynb). Está
    desactivado por defecto: mejora el F1 macro en 0.005 y empeora el error más grave.
    """
    c = BASELINE_V3_CHOICE
    ds1 = load_split("ds1")
    keep = ds1["y"] != "F"  # la clase F queda fuera del alcance (ver docs/intended_use.md)
    y, groups = ds1["y"][keep], ds1["record"][keep]
    Z, names = build_features(ds1["X"][keep], ds1["F"][keep], c["feature_set"], records=groups)

    def make():
        return BaselineClassifier(
            kind=c["kind"], params=c["params"], feature_set=c["feature_set"],
            weight_power=c["weight_power"], classes=list(CLASSES_V3),
        )  # fmt: skip

    t0 = time.time()
    cv = cross_validate_calibrated(make, Z, y, groups, CLASSES_V3, fold_map=config.DS1_FOLD_MAP)
    pred = cv["pred"] if calibrate else np.asarray(CLASSES_V3)[cv["proba"].argmax(axis=1)]
    print(f"CV anidada, {N_FOLDS} folds por paciente ({time.time() - t0:.0f} s)")
    print(per_class_metrics(y, pred, CLASSES_V3).round(3).to_string())
    cv_summary = summary(y, pred, CLASSES_V3)
    print(f"F1 macro: {cv_summary['macro_f1']:.3f} | "
          f"latidos V leídos como N: {int(((y == 'V') & (pred == 'N')).sum())}")  # fmt: skip

    weights = None
    if calibrate:  # umbrales del modelo final, ajustados con validación interna sobre todo DS1
        inner_proba = np.zeros((len(y), len(CLASSES_V3)), dtype=np.float32)
        for itr, iva in patient_folds(groups, 4):
            inner_proba[iva] = make().fit(Z[itr], y[itr]).predict_proba(Z[iva])
        weights = tune_decision_weights(inner_proba, y, CLASSES_V3)
    model = make().fit(Z, y)
    model.decision_weights = weights

    config.MODELS_DIR.mkdir(exist_ok=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    path = config.MODELS_DIR / "baseline_v3.joblib"
    model.save(path)
    meta = {
        "model": "baseline_v3",
        "classes": CLASSES_V3,
        "fuera_de_alcance": ["F"],
        "feature_set": c["feature_set"],
        "feature_names": names,
        "decision_weights": weights,
        "calibrado": calibrate,
        "weight_power": c["weight_power"],
        "train_records": sorted(int(r) for r in np.unique(groups)),
        "seed": config.SEED,
        "cv": {
            "n_folds": N_FOLDS,
            "grouping": "record",
            "calibracion": "anidada (4 folds internos)" if calibrate else "sin calibrar",
            "calibrado_referencia": summary(y, cv["pred"], CLASSES_V3),
            **cv_summary,
        },
    }
    (config.REPORTS_DIR / "baseline_v3_cv.json").write_text(json.dumps(meta, indent=2))
    print(f"Umbrales: {weights or 'sin calibrar'}\nModelo -> {path}")


def train_baseline(
    kind: str,
    feature_set: str,
    params: dict,
    weight_power: float = 1.0,
    model_name: str | None = None,
    classes: list[str] | None = None,
    template_block: int | None = None,
) -> None:
    ds1 = load_split("ds1")
    classes = list(classes or config.CLASSES)
    keep = np.isin(ds1["y"], classes)  # las clases fuera de alcance no se entrenan ni se miden
    y, groups = ds1["y"][keep], ds1["record"][keep]
    # `records` lo usan los conjuntos con RR normalizado y con morfología relativa al paciente
    Z, names = build_features(
        ds1["X"][keep], ds1["F"][keep], feature_set, records=groups, template_block=template_block
    )
    model_name = model_name or f"baseline_{kind}"

    def make():
        return BaselineClassifier(
            kind=kind, params=params, feature_set=feature_set, weight_power=weight_power,
            classes=list(classes), template_block=template_block,
        )  # fmt: skip

    t0 = time.time()
    cv = cross_validate(make, Z, y, groups, fold_map=config.DS1_FOLD_MAP, classes=list(classes))
    cv_summary = summary(y, cv["pred"], classes)
    print(f"CV {N_FOLDS} folds por paciente ({time.time() - t0:.0f} s)")
    print(per_class_metrics(y, cv["pred"], classes).round(3).to_string())
    print(f"F1 macro: {cv_summary['macro_f1']:.3f} | "
          f"latidos V leídos como N: {int(((y == 'V') & (cv['pred'] == 'N')).sum())}")  # fmt: skip

    model = make().fit(Z, y)  # modelo final: todo DS1
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    path = config.MODELS_DIR / f"{model_name}.joblib"
    model.save(path)
    meta = {
        "model": model_name,
        "classes": classes,
        "feature_set": feature_set,
        "feature_names": names,
        "template_block": template_block,
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
    parser.add_argument(
        "model",
        choices=["baseline", "baseline-v2", "baseline-v3", "baseline-v4", "baseline-v5", "cnn"],
    )
    parser.add_argument("--kind", choices=["xgb", "rf"], default=BASELINE_CHOICE["kind"])
    parser.add_argument("--feature-set", default=BASELINE_CHOICE["feature_set"])
    parser.add_argument("--skip-cv", action="store_true", help="CNN: solo el modelo final")
    parser.add_argument(
        "--calibrate", action="store_true", help="v3: activar los umbrales por clase"
    )
    args = parser.parse_args()
    if args.model == "cnn":
        train_cnn(CNNConfig(**CNN_CHOICE), skip_cv=args.skip_cv)
        return
    if args.model == "baseline-v5":
        c = BASELINE_V5_CHOICE
        train_baseline(c["kind"], c["feature_set"], c["params"], c["weight_power"],
                       "baseline_v5", CLASSES_V3, c["template_block"])  # fmt: skip
        return
    if args.model == "baseline-v4":
        c = BASELINE_V4_CHOICE
        train_baseline(c["kind"], c["feature_set"], c["params"], c["weight_power"],
                       "baseline_v4", CLASSES_V3, c["template_block"])  # fmt: skip
        return
    if args.model == "baseline-v3":
        train_baseline_v3(calibrate=args.calibrate)
        return
    if args.model == "baseline-v2":
        c = BASELINE_V2_CHOICE
        train_baseline(c["kind"], c["feature_set"], c["params"], c["weight_power"], "baseline_v2")
        return
    params = BASELINE_CHOICE["params"] if args.kind == BASELINE_CHOICE["kind"] else {}
    train_baseline(args.kind, args.feature_set, params, BASELINE_CHOICE["weight_power"])


if __name__ == "__main__":
    main()
