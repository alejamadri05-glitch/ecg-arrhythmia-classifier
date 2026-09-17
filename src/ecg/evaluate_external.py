"""Validación externa: v1 contra v2 en INCART, con modelos entrenados solo con DS1.

DS2 ya se usó una vez y la v2 se diseñó a partir de lo que se vio ahí, así que DS2 no puede
comparar las dos versiones. INCART sí: ninguna de las dos la vio nunca.

Análisis fijado antes de mirar los resultados:

1. Métricas por clase (Se, +P, F1) de v1 y v2, y F1 macro.
2. Hipótesis: **v2 detecta más S que v1**, sobre todo en registros cuya frecuencia cardíaca cae
   fuera del rango de DS1 (mediana de RR entre 0.55 y 1.12 s).
3. Se espera que ambas versiones rindan peor que en MIT-BIH: cambia la derivación (II en vez de
   MLII), el equipo y la población.

Uso:  python -m ecg.evaluate_external                                   # INCART, v1 vs v2
      python -m ecg.evaluate_external --db svdb --classes NSV \
          --models v2_rr_normalizado,v3_3clases,v4_morfologia_relativa
"""

import argparse
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ecg import config  # noqa: E402
from ecg.evaluate import (  # noqa: E402
    confusion,
    per_class_metrics,
    per_record_metrics,
    plot_confusion,
    summary,
)
from ecg.external import EXTERNAL_DBS, INCART  # noqa: E402
from ecg.features import build_features  # noqa: E402
from ecg.models.baseline import BaselineClassifier  # noqa: E402
from ecg.models.cnn import CNNClassifier  # noqa: E402
from ecg.segment import load_split  # noqa: E402

MODELS = {
    "v1_rr_absoluto": "baseline_xgb.joblib",
    "v2_rr_normalizado": "baseline_v2.joblib",
    "v3_3clases": "baseline_v3.joblib",
    "v4_morfologia_relativa": "baseline_v4.joblib",
}
DEFAULT_MODELS = ["v1_rr_absoluto", "v2_rr_normalizado", "cnn"]  # lo publicado para INCART
DS1_RR_RANGE = (0.55, 1.12)  # rango de medianas de RR por paciente en DS1


def predict_all(
    data: dict,
    with_cnn: bool = True,
    models: list[str] | None = None,
    classes: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Predicciones de cada modelo entrenado solo con DS1.

    `classes` define la tarea sobre la que se compara. Si es más chica que las clases del modelo
    (p. ej. N/S/V contra un modelo de 4 clases), se le restringe la salida a esas clases, que es
    la única forma de comparar modelos de 3 y 4 clases sobre los mismos latidos.
    """
    classes = list(classes or config.CLASSES)
    preds = {}
    for name in models or list(MODELS):
        model = BaselineClassifier.load(config.MODELS_DIR / MODELS[name])
        Z, _ = build_features(
            data["X"], data["F"], model.feature_set, records=data["record"],
            template_block=model.template_block,
        )  # fmt: skip
        columnas = [model.classes.index(c) for c in classes if c in model.classes]
        proba = model.predict_proba(Z)[:, columnas]
        preds[name] = np.asarray([c for c in classes if c in model.classes])[proba.argmax(axis=1)]
    if with_cnn and (config.MODELS_DIR / "cnn.pt").exists():
        cnn = CNNClassifier.load(config.MODELS_DIR / "cnn.pt", device="cpu")
        proba = cnn.predict_proba(data["X"], data["F"])
        columnas = [cnn.classes.index(c) for c in classes]
        preds["cnn"] = np.asarray(classes)[proba[:, columnas].argmax(axis=1)]
    return preds


def per_record_heart_rate(data: dict) -> pd.DataFrame:
    """Mediana de RR por registro y si cae dentro del rango visto en DS1."""
    rows = []
    for rec in np.unique(data["record"]):
        m = data["record"] == rec
        median_rr = float(np.median(data["F"][m, 0]))
        rows.append(
            {
                "record": int(rec),
                "mediana_rr": round(median_rr, 3),
                "lpm": round(60 / median_rr, 1),
                "dentro_rango_ds1": bool(DS1_RR_RANGE[0] <= median_rr <= DS1_RR_RANGE[1]),
                "latidos": int(m.sum()),
                "latidos_S": int(((data["y"] == "S") & m).sum()),
            }
        )
    return pd.DataFrame(rows).set_index("record")


def s_detection_by_heart_rate(
    data: dict, preds: dict[str, np.ndarray], classes: list[str] | None = None
) -> pd.DataFrame:
    """Se y +P de S según si el paciente tiene una frecuencia como las de DS1 o no."""
    hr = per_record_heart_rate(data)
    dentro = np.isin(data["record"], hr.index[hr.dentro_rango_ds1])
    rows = []
    grupos = [
        ("frecuencia como DS1", dentro, int(hr.dentro_rango_ds1.sum())),
        ("fuera del rango de DS1", ~dentro, int((~hr.dentro_rango_ds1).sum())),
    ]
    for grupo, mask, n_pacientes in grupos:
        fila = {
            "grupo": grupo,
            "pacientes": n_pacientes,
            "latidos_S": int((data["y"][mask] == "S").sum()),
        }
        for name, p in preds.items():
            m = per_class_metrics(data["y"][mask], p[mask], classes or config.CLASSES)
            fila[f"{name} · S_Se"] = round(float(m.loc["S", "Se"]), 3)
            fila[f"{name} · S_+P"] = round(float(m.loc["S", "+P"]), 3)
        rows.append(fila)
    return pd.DataFrame(rows).set_index("grupo")


def make_figures(
    data: dict, preds: dict[str, np.ndarray], fig_dir, db=INCART, classes: list[str] | None = None
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    classes = list(classes or config.CLASSES)
    y = data["y"]
    fig, axes = plt.subplots(1, len(preds), figsize=(4.6 * len(preds), 4.2))
    for ax, (name, p) in zip(np.atleast_1d(axes), preds.items(), strict=True):
        plot_confusion(
            confusion(y, p, classes, normalize=True), ax, name, counts=confusion(y, p, classes)
        )
    fig.suptitle(f"{db.name.upper()} · validación externa (canal {db.lead})", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{db.name}_confusion.png", bbox_inches="tight")
    plt.close(fig)

    hr = per_record_heart_rate(data)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for name, color in zip(list(preds)[:2], ["#C44E52", "#55A868"], strict=False):
        se = [
            per_class_metrics(y[data["record"] == rec], preds[name][data["record"] == rec],
                              classes).loc["S", "Se"]
            for rec in hr.index
        ]  # fmt: skip
        ax.scatter(hr.lpm, se, label=name, color=color, alpha=0.75)
    ax.axvspan(60 / DS1_RR_RANGE[1], 60 / DS1_RR_RANGE[0], color="grey", alpha=0.12,
               label="frecuencias vistas en DS1")  # fmt: skip
    ax.set_xlabel("frecuencia cardíaca mediana del paciente (lpm)")
    ax.set_ylabel("Se de S")
    ax.set_title(f"{db.name.upper()}: detección de S por paciente")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{db.name}_s_by_heart_rate.png")
    plt.close(fig)


def evaluate_external(
    db=INCART, models: list[str] | None = None, classes: list[str] | None = None
) -> dict:
    """Evalúa los modelos elegidos en una base externa. Se corre **una sola vez** por base."""
    classes = list(classes or config.CLASSES)
    models = models or DEFAULT_MODELS
    data = load_split(db.name)
    dentro = np.isin(data["y"], classes)  # las clases fuera de la tarea no se miden
    data = {k: v[dentro] for k, v in data.items()}
    y = data["y"]
    preds = predict_all(
        data, with_cnn="cnn" in models, models=[m for m in models if m != "cnn"],
        classes=classes,
    )  # fmt: skip
    hr = per_record_heart_rate(data)

    results = {
        "aviso": "Proyecto educativo. No es un dispositivo médico.",
        "dataset": {
            "nombre": db.name,
            "canal": db.lead,
            "remuestreo": f"{db.fs} -> {config.FS} Hz",
            "clases_evaluadas": classes,
            "registros": int(len(hr)),
            "latidos": int(len(y)),
            "por_clase": {c: int((y == c).sum()) for c in classes},
            "latidos_excluidos_por_clase_fuera_de_la_tarea": int((~dentro).sum()),
            "pacientes_fuera_del_rango_de_frecuencia_de_ds1": int((~hr.dentro_rango_ds1).sum()),
        },
        "accuracy_prediciendo_siempre_N": round(float((y == "N").mean()), 4),
        "modelos": {
            name: {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in summary(y, p, classes).items()
            }
            | {
                "V_como_N": int(((y == "V") & (p == "N")).sum()),
                "falsas_alarmas_sobre_N": int(((y == "N") & (p != "N")).sum()),
                "errores": int((y != p).sum()),
            }
            for name, p in preds.items()
        },
        "S_por_frecuencia": json.loads(s_detection_by_heart_rate(data, preds, classes).to_json()),
    }
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / f"{db.name}_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )
    pd.concat(
        {n: per_class_metrics(y, p, classes) for n, p in preds.items()}, names=["modelo"]
    ).round(4).to_csv(config.REPORTS_DIR / f"{db.name}_per_class.csv")
    pd.concat(
        {n: per_record_metrics(y, p, data["record"], classes) for n, p in preds.items()},
        names=["modelo"],
    ).round(4).to_csv(config.REPORTS_DIR / f"{db.name}_per_record.csv")
    hr.to_csv(config.REPORTS_DIR / f"{db.name}_heart_rate.csv")
    make_figures(data, preds, config.REPORTS_DIR / "figures", db, classes)
    return {"results": results, "preds": preds, "data": data, "hr": hr, "classes": classes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", choices=sorted(EXTERNAL_DBS), default="incart")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument(
        "--classes", default="".join(config.CLASSES), help="tarea a evaluar, p. ej. NSV"
    )
    args = parser.parse_args()
    out = evaluate_external(EXTERNAL_DBS[args.db], args.models.split(","), list(args.classes))
    r, y, classes = out["results"], out["data"]["y"], out["classes"]
    print(f"{r['dataset']['nombre']}: {r['dataset']['latidos']} latidos de "
          f"{r['dataset']['registros']} pacientes | clases {classes}")  # fmt: skip
    print(f"Exactitud prediciendo siempre N: {r['accuracy_prediciendo_siempre_N']:.3f}\n")
    for name, p in out["preds"].items():
        print(f"=== {name}")
        print(per_class_metrics(y, p, classes).round(3).to_string())
        m = r["modelos"][name]
        print(f"F1 macro: {m['macro_f1']:.3f} | V leídos como N: {m['V_como_N']} | "
              f"errores: {m['errores']}\n")  # fmt: skip
    print(s_detection_by_heart_rate(out["data"], out["preds"], classes).to_string())


if __name__ == "__main__":
    main()
