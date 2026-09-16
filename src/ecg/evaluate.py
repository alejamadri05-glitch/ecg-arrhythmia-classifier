"""Métricas estilo AAMI EC57: sensibilidad (Se), predictividad positiva (+P) y F1 por clase.

La exactitud se reporta, pero no es la métrica principal: con N ≈ 90 %, predecir todo N ya
da ~90 % de exactitud.
"""

import numpy as np
import pandas as pd

from ecg import config


def confusion(y_true, y_pred, classes=config.CLASSES, normalize: bool = False) -> pd.DataFrame:
    """Matriz de confusión (filas = real, columnas = predicho). `normalize` normaliza por fila."""
    classes = list(classes)
    order = {c: i for i, c in enumerate(classes)}
    counts = np.zeros((len(classes), len(classes)), dtype=np.int64)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true):  # con una entrada vacía la matriz queda en ceros, no vacía
        rows = np.array([order[v] for v in y_true])
        cols = np.array([order[v] for v in y_pred])
        np.add.at(counts, (rows, cols), 1)
    cm = pd.DataFrame(
        counts.astype(float if normalize else int),
        index=pd.Index(classes, name="real"),
        columns=pd.Index(classes, name="predicho"),
    )
    if normalize:
        cm = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0)
    return cm


def per_class_metrics(y_true, y_pred, classes=config.CLASSES) -> pd.DataFrame:
    """Se, +P y F1 por clase. NaN cuando no hay casos reales (Se) o predichos (+P)."""
    cm = confusion(y_true, y_pred, classes).to_numpy()
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        se = tp / (tp + fn)
        ppv = tp / (tp + fp)
        f1 = 2 * tp / (2 * tp + fn + fp)
    return pd.DataFrame(
        {"Se": se, "+P": ppv, "F1": f1, "TP": tp, "FN": fn, "FP": fp, "n": tp + fn},
        index=pd.Index(classes, name="clase"),
    )


def summary(y_true, y_pred, classes=config.CLASSES) -> dict:
    """Resumen plano (útil para tablas de experimentos y JSON)."""
    m = per_class_metrics(y_true, y_pred, classes)
    out = {
        "macro_f1": float(m["F1"].fillna(0).mean()),
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
    }
    for cls in classes:
        for k in ["Se", "+P", "F1"]:
            v = m.loc[cls, k]
            out[f"{cls}_{k}"] = None if np.isnan(v) else float(v)
    return out


def per_record_metrics(y_true, y_pred, records, classes=config.CLASSES) -> pd.DataFrame:
    """Rendimiento por paciente: dónde falla el modelo y con qué clases."""
    rows = []
    y_true, y_pred, records = map(np.asarray, (y_true, y_pred, records))
    for rec in np.unique(records):
        m = records == rec
        pc = per_class_metrics(y_true[m], y_pred[m], classes)
        row = {
            "record": int(rec),
            "latidos": int(m.sum()),
            "errores": int((y_true[m] != y_pred[m]).sum()),
        }
        for cls in classes:
            row[f"n_{cls}"] = int(pc.loc[cls, "n"])
        for cls in classes[1:]:
            row[f"Se_{cls}"] = pc.loc[cls, "Se"]
            row[f"+P_{cls}"] = pc.loc[cls, "+P"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("record")


def plot_confusion(cm_norm: pd.DataFrame, ax, title: str = "", counts: pd.DataFrame | None = None):
    """Heatmap de la matriz normalizada por fila, con conteos opcionales."""
    im = ax.imshow(cm_norm.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    classes = list(cm_norm.index)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicho")
    ax.set_ylabel("real")
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cm_norm.iat[i, j]
            label = f"{v:.2f}" if counts is None else f"{v:.2f}\n({counts.iat[i, j]})"
            ax.text(j, i, label, ha="center", va="center", fontsize=8,
                    color="white" if v > 0.6 else "black")  # fmt: skip
    ax.set_title(title)
    return im
