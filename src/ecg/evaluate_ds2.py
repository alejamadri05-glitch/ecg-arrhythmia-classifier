"""Evaluación final en DS2: se ejecuta **una sola vez**, con los modelos ya elegidos.

Los modelos y la configuración salieron de la validación cruzada dentro de DS1
(`reports/model_selection.json`). Acá no se ajusta nada: solo se mide.

Uso:  python -m ecg.evaluate_ds2
"""

import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ecg import config  # noqa: E402
from ecg.data import rhythm_at  # noqa: E402
from ecg.evaluate import (  # noqa: E402
    confusion,
    per_class_metrics,
    per_record_metrics,
    plot_confusion,
    summary,
)
from ecg.features import build_features  # noqa: E402
from ecg.models.baseline import BaselineClassifier  # noqa: E402
from ecg.models.cnn import CNNClassifier  # noqa: E402
from ecg.segment import load_split  # noqa: E402

# Riesgos anotados en las fases 1–4, a verificar ahora en DS2
LBBB_RECORDS = [111, 214]  # bloqueo de rama izquierda: el baseline falló en el 207 de DS1
BRADY_RECORD = 232  # aporta el 75 % de los S de DS2 y tiene RR absoluto distinto
FUSION_RECORD = 213  # aporta el 93 % de los F de DS2
NOISY_RECORD = 105  # el registro con más ruido anotado


def predict_all(ds2: dict) -> dict[str, np.ndarray]:
    """Predicciones de los dos modelos entrenados solo con DS1."""
    out = {}
    baseline = BaselineClassifier.load(config.MODELS_DIR / "baseline_xgb.joblib")
    Z, _ = build_features(ds2["X"], ds2["F"], baseline.feature_set)
    out["baseline_xgb"] = baseline.predict(Z)
    cnn = CNNClassifier.load(config.MODELS_DIR / "cnn.pt", device="cpu")
    out["cnn"] = cnn.predict(ds2["X"], ds2["F"])
    return out


def rhythm_labels(ds2: dict) -> np.ndarray:
    """Ritmo anotado en cada latido de DS2 (por registro)."""
    out = np.empty(len(ds2["y"]), dtype=object)
    for rec in np.unique(ds2["record"]):
        m = ds2["record"] == rec
        out[m] = rhythm_at(int(rec), ds2["r_peak"][m])
    return out


def risk_checks(ds2: dict, preds: dict[str, np.ndarray]) -> dict:
    """Verifica en DS2 los riesgos concretos detectados en DS1."""
    y, rec, sym = ds2["y"], ds2["record"], ds2["symbol"]
    rhythms = rhythm_labels(ds2)
    checks: dict = {}

    checks["lbbb"] = {}
    for r in LBBB_RECORDS:
        m = (rec == r) & (sym == "L")
        checks["lbbb"][str(r)] = {
            "latidos_L": int(m.sum()),
            **{f"{k}_pct_N": round(100 * float((p[m] == "N").mean()), 1) for k, p in preds.items()},
        }

    m232 = rec == BRADY_RECORD
    checks[f"record_{BRADY_RECORD}_S"] = {
        "latidos_S": int(((y == "S") & m232).sum()),
        "pct_de_los_S_de_DS2": round(100 * float(((y == "S") & m232).sum() / (y == "S").sum()), 1),
        **{
            k: {
                "Se": round(float(per_class_metrics(y[m232], p[m232]).loc["S", "Se"]), 3),
                "+P": round(float(per_class_metrics(y[m232], p[m232]).loc["S", "+P"]), 3),
            }
            for k, p in preds.items()
        },
    }

    mf = rec == FUSION_RECORD
    checks[f"record_{FUSION_RECORD}_F"] = {
        "latidos_F": int(((y == "F") & mf).sum()),
        **{k: round(float((p[(y == "F") & mf] == "F").mean()), 3) for k, p in preds.items()},
    }

    checks["V_no_detectados"] = {  # RISK-01: el error más grave
        k: {
            "V_totales": int((y == "V").sum()),
            "V_como_N": int(((y == "V") & (p == "N")).sum()),
            "peores_registros": pd.Series(rec[(y == "V") & (p == "N")])
            .value_counts()
            .head(3)
            .to_dict(),
        }
        for k, p in preds.items()
    }

    mn = rec == NOISY_RECORD
    checks[f"record_{NOISY_RECORD}_ruido"] = {
        "latidos": int(mn.sum()),
        **{k: {"errores": int((y[mn] != p[mn]).sum())} for k, p in preds.items()},
    }

    top_rhythms = pd.Series(rhythms).value_counts().head(4).index.tolist()
    checks["por_ritmo"] = {}
    for rh in top_rhythms:
        m = rhythms == rh
        checks["por_ritmo"][rh] = {
            "latidos": int(m.sum()),
            **{
                k: {
                    "N_Se": round(float(per_class_metrics(y[m], p[m]).loc["N", "Se"]), 3),
                    "S_Se": round(float(per_class_metrics(y[m], p[m]).loc["S", "Se"]), 3),
                    "V_Se": round(float(per_class_metrics(y[m], p[m]).loc["V", "Se"]), 3),
                }
                for k, p in preds.items()
            },
        }
    return checks


def make_figures(ds2: dict, preds: dict[str, np.ndarray], fig_dir) -> None:
    y, rec = ds2["y"], ds2["record"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    titles = {"baseline_xgb": "Baseline XGBoost (primario)", "cnn": "CNN 1D (comparación)"}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, (name, p) in zip(axes, preds.items(), strict=True):
        plot_confusion(confusion(y, p, normalize=True), ax, titles[name], counts=confusion(y, p))
    fig.suptitle("DS2 (22 pacientes no vistos)", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "ds2_confusion.png", bbox_inches="tight")
    plt.close(fig)

    err = pd.DataFrame({k: per_record_metrics(y, p, rec)["errores"] for k, p in preds.items()})
    fig, ax = plt.subplots(figsize=(10, 3.2))
    xs = np.arange(len(err))
    ax.bar(xs - 0.2, err["baseline_xgb"], 0.4, label="baseline XGBoost", color="#4C72B0")
    ax.bar(xs + 0.2, err["cnn"], 0.4, label="CNN 1D", color="#DD8452")
    ax.set_xticks(xs, err.index.astype(str), rotation=90)
    ax.set_ylabel("errores")
    ax.set_title("DS2: errores por paciente")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "ds2_errors_by_record.png")
    plt.close(fig)

    # Ejemplos del error más grave: latidos V clasificados como N (RISK-01)
    primary = preds["baseline_xgb"]
    idx = np.flatnonzero((y == "V") & (primary == "N"))[:6]
    t_ms = (np.arange(config.WINDOW) - config.PRE_SAMPLES) / config.FS * 1000
    fig, axes = plt.subplots(2, 3, figsize=(11, 4.5), sharex=True)
    for ax, i in zip(axes.ravel(), idx, strict=False):
        ax.plot(t_ms, ds2["X"][i], color="#C44E52", lw=1.2)
        ax.axvline(0, color="grey", ls=":", lw=0.8)
        ax.set_title(f"registro {ds2['record'][i]} · RR previo {ds2['F'][i, 0]:.2f} s", fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("ms desde el pico R")
    fig.suptitle("DS2: latidos V clasificados como N por el modelo primario (RISK-01)", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "ds2_error_examples.png", bbox_inches="tight")
    plt.close(fig)


def evaluate_ds2() -> dict:
    ds2 = load_split("ds2")
    y, rec = ds2["y"], ds2["record"]
    preds = predict_all(ds2)
    selection = json.loads((config.REPORTS_DIR / "model_selection.json").read_text())

    results = {
        "aviso": "Proyecto educativo. No es un dispositivo médico.",
        "seleccion": selection,
        "dataset": {
            "split": "DS2 (de Chazal et al., 2004)",
            "registros": sorted(int(r) for r in np.unique(rec)),
            "latidos": int(len(y)),
            "por_clase": {c: int((y == c).sum()) for c in config.CLASSES},
        },
        "accuracy_prediciendo_siempre_N": round(float((y == "N").mean()), 4),
        "modelos": {},
        "riesgos": risk_checks(ds2, preds),
    }
    for name, p in preds.items():
        results["modelos"][name] = {
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary(y, p).items()},
            "matriz_confusion": confusion(y, p).to_dict(),
        }

    config.REPORTS_DIR.mkdir(exist_ok=True)
    per_class = pd.concat(
        {name: per_class_metrics(y, p) for name, p in preds.items()}, names=["modelo"]
    )
    per_class.round(4).to_csv(config.REPORTS_DIR / "ds2_per_class.csv")
    per_record = pd.concat(
        {name: per_record_metrics(y, p, rec) for name, p in preds.items()}, names=["modelo"]
    )
    per_record.round(4).to_csv(config.REPORTS_DIR / "ds2_per_record.csv")
    (config.REPORTS_DIR / "ds2_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )
    make_figures(ds2, preds, config.REPORTS_DIR / "figures")
    return {"results": results, "preds": preds, "per_class": per_class, "ds2": ds2}


def main() -> None:
    out = evaluate_ds2()
    r, y = out["results"], out["ds2"]["y"]
    print(f"DS2: {r['dataset']['latidos']} latidos, {len(r['dataset']['registros'])} pacientes")
    print(f"Exactitud prediciendo siempre N: {r['accuracy_prediciendo_siempre_N']:.3f}\n")
    for name, p in out["preds"].items():
        marca = " (primario)" if name == r["seleccion"]["primary_model"] else ""
        print(f"=== {name}{marca}")
        print(per_class_metrics(y, p).round(3).to_string())
        m = r["modelos"][name]
        print(f"F1 macro: {m['macro_f1']:.3f} | exactitud: {m['accuracy']:.3f}\n")
    print(json.dumps(r["riesgos"], indent=2, ensure_ascii=False))
    print(f"\nReportes -> {config.REPORTS_DIR}/ds2_*.{{json,csv}} y figures/ds2_*.png")


if __name__ == "__main__":
    main()
