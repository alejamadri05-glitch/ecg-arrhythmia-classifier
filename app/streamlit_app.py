"""Demo interactiva: clasificación de latidos sobre registros de MIT-BIH.

Muestra la anotación del cardiólogo y la predicción del modelo lado a lado, sobre pacientes que
el modelo nunca vio (DS2).

> Proyecto educativo y de investigación. **No es un dispositivo médico.**

Uso:  streamlit run app/streamlit_app.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ecg import config
from ecg.data import load_record
from ecg.evaluate import confusion, per_class_metrics
from ecg.features import build_features
from ecg.models.baseline import BaselineClassifier
from ecg.preprocess import bandpass
from ecg.segment import extract_record

COLORES = {"N": "#4C72B0", "S": "#DD8452", "V": "#C44E52", "F": "#8172B3"}
MODELOS = {
    "v5 · contexto de secuencia (recomendado)": "baseline_v5.joblib",
    "v4 · morfología relativa al paciente": "baseline_v4.joblib",
    "v3 · 3 clases": "baseline_v3.joblib",
    "v1 · primera versión (RR absoluto)": "baseline_xgb.joblib",
}

st.set_page_config(page_title="Clasificador de arritmias ECG", layout="wide")


@st.cache_resource
def cargar_modelo(archivo: str) -> BaselineClassifier:
    return BaselineClassifier.load(config.MODELS_DIR / archivo)


@st.cache_data
def cargar_latidos(registro: int) -> dict:
    return extract_record(registro)


@st.cache_data
def cargar_senal(registro: int) -> tuple[np.ndarray, float]:
    rec = load_record(registro)
    return bandpass(rec.signal, rec.fs), rec.fs


def predecir(archivo: str, registro: int) -> tuple[np.ndarray, dict]:
    modelo = cargar_modelo(archivo)
    latidos = cargar_latidos(registro)
    Z, _ = build_features(
        latidos["X"], latidos["F"], modelo.feature_set, template_block=modelo.template_block
    )
    return np.asarray(modelo.classes)[modelo.predict_proba(Z).argmax(axis=1)], latidos


st.title("Clasificador de arritmias ECG")
st.caption(
    "Proyecto educativo y de investigación. **No es un dispositivo médico** ni debe usarse para "
    "decisiones clínicas."
)

with st.sidebar:
    st.header("Qué mirar")
    registro = st.selectbox(
        "Registro (DS2: pacientes que el modelo nunca vio)", config.DS2, index=config.DS2.index(232)
    )
    nombre_modelo = st.selectbox("Modelo", list(MODELOS))
    inicio_min = st.slider("Minuto de inicio", 0.0, 29.0, 0.0, 0.5)
    ventana_s = st.slider("Ventana (segundos)", 5, 30, 10)
    st.markdown(
        "**Sugerencias**\n\n"
        "- **232**: paciente bradicárdico con latidos supraventriculares sostenidos, donde el "
        "modelo falla. Comparar v1 con v5.\n"
        "- **111** o **214**: bloqueo de rama izquierda, el error que resolvió la v4.\n"
        "- **105**: el registro más ruidoso."
    )

archivo = MODELOS[nombre_modelo]
try:
    pred, latidos = predecir(archivo, registro)
except FileNotFoundError:
    st.error(
        f"No encuentro `models/{archivo}`. Entrenalo con `python -m ecg.train baseline-v5` "
        "(y `python -m ecg.segment` si todavía no descargaste MIT-BIH)."
    )
    st.stop()

real = latidos["y"]
clases = list(cargar_modelo(archivo).classes)
fuera_de_alcance = [c for c in config.CLASSES if c not in clases]
en_alcance = np.isin(real, clases)

senal, fs = cargar_senal(registro)
desde, hasta = int(inicio_min * 60 * fs), int((inicio_min * 60 + ventana_s) * fs)
visible = (latidos["r_peak"] >= desde) & (latidos["r_peak"] < hasta)

st.subheader(f"Registro {registro} · minuto {inicio_min:.1f} a {inicio_min + ventana_s / 60:.1f}")
fig, ejes = plt.subplots(2, 1, figsize=(13, 5), sharex=True, sharey=True)
tiempo = np.arange(desde, hasta) / fs
for eje, etiquetas, titulo in [
    (ejes[0], real, "Anotación del cardiólogo"),
    (ejes[1], pred, f"Predicción · {nombre_modelo.split(' · ')[0]}"),
]:
    eje.plot(tiempo, senal[desde:hasta], color="black", lw=0.8)
    for pico, etiqueta in zip(latidos["r_peak"][visible], etiquetas[visible], strict=True):
        eje.axvline(pico / fs, color=COLORES.get(etiqueta, "grey"), alpha=0.28, lw=6)
        eje.annotate(
            etiqueta,
            (pico / fs, eje.get_ylim()[1]),
            ha="center",
            va="top",
            fontsize=9,
            color=COLORES.get(etiqueta, "grey"),
        )
    eje.set_title(titulo, loc="left", fontsize=10)
    eje.set_ylabel("mV")
ejes[1].set_xlabel("tiempo (s)")
fig.tight_layout()
st.pyplot(fig)

diferencias = int((real[visible] != pred[visible]).sum())
if diferencias:
    st.warning(f"En esta ventana hay **{diferencias}** latidos donde la predicción no coincide.")
else:
    st.success("En esta ventana todas las predicciones coinciden con la anotación.")

st.subheader(f"Registro completo ({len(real)} latidos, 30 minutos)")
col_izq, col_der = st.columns([2, 3])
with col_izq:
    st.markdown("**Métricas por clase**")
    st.dataframe(
        per_class_metrics(real[en_alcance], pred[en_alcance], clases).round(3),
        use_container_width=True,
    )
    if fuera_de_alcance:
        n_fuera = int((~en_alcance).sum())
        st.caption(
            f"La clase {', '.join(fuera_de_alcance)} está fuera del alcance de este modelo "
            f"({n_fuera} latidos del registro no se evalúan)."
        )
with col_der:
    st.markdown("**Matriz de confusión** (filas: anotación, columnas: predicción)")
    st.dataframe(confusion(real[en_alcance], pred[en_alcance], clases), use_container_width=True)

rr = np.diff(latidos["r_peak"]) / fs
mediana_rr = float(np.median(rr))
irregulares = float(np.mean(np.abs(rr - mediana_rr) / mediana_rr > 0.30))
avisos = []
if irregulares > 0.15:
    avisos.append(
        f"Ritmo irregular: el {irregulares:.0%} de los latidos se aparta más del 30 % del "
        "intervalo mediano. El modelo supone ritmo de base regular."
    )
if not (0.55 <= mediana_rr <= 1.12):
    avisos.append(
        f"Frecuencia cardíaca mediana de {60 / mediana_rr:.0f} lpm, fuera del rango visto en "
        "entrenamiento (54–109 lpm)."
    )
if avisos:
    st.error("**Avisos del sistema** (los mismos que devuelve la API)\n\n- " + "\n- ".join(avisos))

with st.expander("Errores por tipo"):
    tipos = pd.Series(
        [f"{a} → {b}" for a, b in zip(real[en_alcance], pred[en_alcance], strict=True) if a != b]
    ).value_counts()
    st.dataframe(tipos.to_frame("latidos"), use_container_width=True)
