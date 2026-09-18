"""Genera la figura del README: anotación del cardiólogo contra predicción del modelo.

Es la misma vista de la demo de Streamlit, pero reproducible desde la línea de comandos para que
la imagen del README no sea una captura de pantalla que nadie puede volver a generar.

    python scripts/readme_figure.py --registro 214 --inicio 1265   # la del README
    python scripts/readme_figure.py --registro 232                 # el caso que no se resolvió

Requiere MIT-BIH descargado y el modelo entrenado:
    python -m ecg.segment && python -m ecg.train baseline-v5
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ecg import config  # noqa: E402
from ecg.data import load_record  # noqa: E402
from ecg.features import build_features  # noqa: E402
from ecg.models.baseline import BaselineClassifier  # noqa: E402
from ecg.preprocess import bandpass  # noqa: E402
from ecg.segment import extract_record  # noqa: E402

COLORES = {"N": "#4C72B0", "S": "#DD8452", "V": "#C44E52", "F": "#8172B3"}
SALIDA = config.REPORTS_DIR / "figures" / "readme_demo.png"


def mejor_ventana(picos: np.ndarray, etiquetas: np.ndarray, fs: float, segundos: float) -> int:
    """Primera ventana con al menos dos latidos anormales: una tira de puro N no muestra nada."""
    ancho = int(segundos * fs)
    for inicio in range(0, int(picos.max()) - ancho, ancho // 2):
        dentro = (picos >= inicio) & (picos < inicio + ancho)
        if dentro.sum() >= 6 and (etiquetas[dentro] != "N").sum() >= 2:
            return inicio
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registro", type=int, default=214)
    parser.add_argument("--modelo", default="baseline_v5.joblib")
    parser.add_argument("--segundos", type=float, default=10.0)
    parser.add_argument(
        "--inicio",
        type=float,
        default=None,
        help="Segundo en que empieza la ventana. Por defecto, la primera con latidos anormales.",
    )
    parser.add_argument("--salida", type=Path, default=SALIDA)
    args = parser.parse_args()

    modelo = BaselineClassifier.load(config.MODELS_DIR / args.modelo)
    latidos = extract_record(args.registro)
    Z, _ = build_features(
        latidos["X"], latidos["F"], modelo.feature_set, template_block=modelo.template_block
    )
    pred = np.asarray(modelo.classes)[modelo.predict_proba(Z).argmax(axis=1)]
    real = latidos["y"]

    rec = load_record(args.registro)
    señal = bandpass(rec.signal, rec.fs)
    if args.inicio is None:
        desde = mejor_ventana(latidos["r_peak"], real, rec.fs, args.segundos)
    else:
        desde = int(args.inicio * rec.fs)
    hasta = desde + int(args.segundos * rec.fs)
    visible = (latidos["r_peak"] >= desde) & (latidos["r_peak"] < hasta)

    fig, ejes = plt.subplots(2, 1, figsize=(12, 4.6), sharex=True, sharey=True)
    tiempo = np.arange(desde, hasta) / rec.fs
    for eje, etiquetas, titulo in [
        (ejes[0], real, "Anotación del cardiólogo"),
        (ejes[1], pred, f"Predicción · {args.modelo.replace('.joblib', '')}"),
    ]:
        eje.plot(tiempo, señal[desde:hasta], color="black", lw=0.9)
        for pico, etiqueta in zip(latidos["r_peak"][visible], etiquetas[visible], strict=True):
            eje.axvline(pico / rec.fs, color=COLORES.get(etiqueta, "grey"), alpha=0.28, lw=7)
            eje.annotate(
                etiqueta,
                (pico / rec.fs, eje.get_ylim()[1]),
                ha="center",
                va="top",
                fontsize=9,
                color=COLORES.get(etiqueta, "grey"),
                weight="bold",
            )
        eje.set_title(titulo, loc="left", fontsize=11)
        eje.set_ylabel("mV")
        eje.spines[["top", "right"]].set_visible(False)

    coinciden = int((real[visible] == pred[visible]).sum())
    ejes[1].set_xlabel(
        f"segundos · registro {args.registro} (DS2, paciente que el modelo nunca vio) · "
        f"{coinciden} de {int(visible.sum())} latidos coinciden con la anotación"
    )
    fig.tight_layout()
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.salida, dpi=130, bbox_inches="tight")
    print(f"{args.salida} · {coinciden}/{int(visible.sum())} latidos coinciden")


if __name__ == "__main__":
    main()
