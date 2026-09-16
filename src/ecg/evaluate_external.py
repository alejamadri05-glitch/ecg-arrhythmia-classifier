"""Validación externa: v1 contra v2 en INCART, con modelos entrenados solo con DS1.

DS2 ya se usó una vez y la v2 se diseñó a partir de lo que se vio ahí, así que DS2 no puede
comparar las dos versiones. INCART sí: ninguna de las dos la vio nunca.

Análisis fijado antes de mirar los resultados:

1. Métricas por clase (Se, +P, F1) de v1 y v2, y F1 macro.
2. Hipótesis: **v2 detecta más S que v1**, sobre todo en registros cuya frecuencia cardíaca cae
   fuera del rango de DS1 (mediana de RR entre 0.55 y 1.12 s).
3. Se espera que ambas versiones rindan peor que en MIT-BIH: cambia la derivación (II en vez de
   MLII), el equipo y la población.

Uso:  python -m ecg.evaluate_external
"""

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
from ecg.external import INCART_LEAD  # noqa: E402
from ecg.features import build_features  # noqa: E402
from ecg.models.baseline import BaselineClassifier  # noqa: E402
from ecg.models.cnn import CNNClassifier  # noqa: E402
from ecg.segment import load_split  # noqa: E402

MODELS = {
    "v1_rr_absoluto": "baseline_xgb.joblib",
    "v2_rr_normalizado": "baseline_v2.joblib",
}
DS1_RR_RANGE = (0.55, 1.12)  # rango de medianas de RR por paciente en DS1


def predict_all(data: dict, with_cnn: bool = True) -> dict[str, np.ndarray]:
    preds = {}
    for name, filename in MODELS.items():
        model = BaselineClassifier.load(config.MODELS_DIR / filename)
        Z, _ = build_features(data["X"], data["F"], model.feature_set, records=data["record"])
        preds[name] = model.predict(Z)
    if with_cnn and (config.MODELS_DIR / "cnn.pt").exists():
        cnn = CNNClassifier.load(config.MODELS_DIR / "cnn.pt", device="cpu")
        preds["cnn"] = cnn.predict(data["X"], data["F"])
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


def s_detection_by_heart_rate(data: dict, preds: dict[str, np.ndarray]) -> pd.DataFrame:
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
            m = per_class_metrics(data["y"][mask], p[mask])
            fila[f"{name} · S_Se"] = round(float(m.loc["S", "Se"]), 3)
            fila[f"{name} · S_+P"] = round(float(m.loc["S", "+P"]), 3)
        rows.append(fila)
    return pd.DataFrame(rows).set_index("grupo")


def make_figures(data: dict, preds: dict[str, np.ndarray], fig_dir) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    y = data["y"]
    fig, axes = plt.subplots(1, len(preds), figsize=(4.6 * len(preds), 4.2))
    for ax, (name, p) in zip(np.atleast_1d(axes), preds.items(), strict=True):
        plot_confusion(confusion(y, p, normalize=True), ax, name, counts=confusion(y, p))
    fig.suptitle(f"INCART · validación externa (derivación {INCART_LEAD})", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "incart_confusion.png", bbox_inches="tight")
    plt.close(fig)

    hr = per_record_heart_rate(data)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for name, color in [("v1_rr_absoluto", "#C44E52"), ("v2_rr_normalizado", "#55A868")]:
        se = []
        for rec in hr.index:
            m = data["record"] == rec
            se.append(per_class_metrics(y[m], preds[name][m]).loc["S", "Se"])
        ax.scatter(hr.lpm, se, label=name, color=color, alpha=0.75)
    ax.axvspan(60 / DS1_RR_RANGE[1], 60 / DS1_RR_RANGE[0], color="grey", alpha=0.12,
               label="frecuencias vistas en DS1")  # fmt: skip
    ax.set_xlabel("frecuencia cardíaca mediana del paciente (lpm)")
    ax.set_ylabel("Se de S")
    ax.set_title("INCART: detección de S por paciente")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "incart_s_by_heart_rate.png")
    plt.close(fig)


def evaluate_external() -> dict:
    data = load_split("incart")
    y = data["y"]
    preds = predict_all(data)
    hr = per_record_heart_rate(data)

    results = {
        "aviso": "Proyecto educativo. No es un dispositivo médico.",
        "dataset": {
            "nombre": "INCART (St Petersburg, PhysioNet)",
            "derivacion": INCART_LEAD,
            "remuestreo": "257 -> 360 Hz",
            "registros": int(len(hr)),
            "latidos": int(len(y)),
            "por_clase": {c: int((y == c).sum()) for c in config.CLASSES},
            "pacientes_fuera_del_rango_de_frecuencia_de_ds1": int((~hr.dentro_rango_ds1).sum()),
        },
        "accuracy_prediciendo_siempre_N": round(float((y == "N").mean()), 4),
        "modelos": {
            name: {
                k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary(y, p).items()
            }
            for name, p in preds.items()
        },
        "S_por_frecuencia": json.loads(s_detection_by_heart_rate(data, preds).to_json()),
    }
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "incart_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )
    pd.concat({n: per_class_metrics(y, p) for n, p in preds.items()}, names=["modelo"]).round(
        4
    ).to_csv(config.REPORTS_DIR / "incart_per_class.csv")
    pd.concat(
        {n: per_record_metrics(y, p, data["record"]) for n, p in preds.items()}, names=["modelo"]
    ).round(4).to_csv(config.REPORTS_DIR / "incart_per_record.csv")
    hr.to_csv(config.REPORTS_DIR / "incart_heart_rate.csv")
    make_figures(data, preds, config.REPORTS_DIR / "figures")
    return {"results": results, "preds": preds, "data": data, "hr": hr}


def main() -> None:
    out = evaluate_external()
    r, y = out["results"], out["data"]["y"]
    print(f"INCART: {r['dataset']['latidos']} latidos de {r['dataset']['registros']} pacientes")
    print(f"Exactitud prediciendo siempre N: {r['accuracy_prediciendo_siempre_N']:.3f}\n")
    for name, p in out["preds"].items():
        print(f"=== {name}")
        print(per_class_metrics(y, p).round(3).to_string())
        print(f"F1 macro: {r['modelos'][name]['macro_f1']:.3f}\n")
    print(s_detection_by_heart_rate(out["data"], out["preds"]).to_string())


if __name__ == "__main__":
    main()
