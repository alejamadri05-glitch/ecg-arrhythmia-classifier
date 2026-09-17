"""API de clasificación de latidos (FastAPI).

Sirve el modelo v5. Las validaciones de entrada no son cortesía: son los controles de riesgo
del análisis (ver `docs/risk_analysis.md` en la Fase 7).

> Proyecto educativo y de investigación. **No es un dispositivo médico** y no debe usarse para
> decisiones clínicas.

Uso:  uvicorn api.main:app --reload
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import numpy as np
import wfdb.processing
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field

from ecg import config
from ecg.features import build_features
from ecg.models.baseline import BaselineClassifier
from ecg.preprocess import bandpass, resample
from ecg.segment import beat_windows

MODEL_PATH = Path(os.getenv("ECG_MODEL", config.MODELS_DIR / "baseline_v5.joblib"))
MODEL_VERSION = os.getenv("ECG_MODEL_VERSION", "5.0.0")

# Límites de entrada (controles de riesgo)
MIN_FS, MAX_FS = 100.0, 2000.0  # REQ-002: fuera de este rango se rechaza
MIN_SECONDS = 10.0  # señal mínima para que el contexto tenga sentido
MIN_BEATS = 3  # cada latido necesita un vecino previo y uno siguiente
FEW_BEATS = 20  # por debajo, la plantilla del paciente es poco confiable
MAX_SAMPLES = 20_000_000  # ~15 h a 360 Hz
ABNORMAL_FRACTION = 0.5  # por encima, el supuesto "lo dominante es lo normal" se cae
# Rango de frecuencia cardíaca mediana por paciente visto en entrenamiento (DS1)
TRAINED_RR_RANGE = (0.55, 1.12)
# Ritmo irregular: fracción de latidos cuyo RR se aparta más del 30 % de la mediana.
# Umbral calibrado contra los casos de falla conocidos (registro 232: 0.22; registro 865 de SVDB:
# 0.36) y registros de ritmo regular (0.00–0.01).
RR_DEVIATION = 0.30
IRREGULAR_FRACTION = 0.15

DISCLAIMER = (
    "Proyecto educativo y de investigación. No es un dispositivo médico; "
    "requiere revisión humana de un profesional."
)

app = FastAPI(
    title="Clasificador de arritmias ECG",
    description=DISCLAIMER,
    version=MODEL_VERSION,
)


@lru_cache(maxsize=1)
def get_model() -> BaselineClassifier:
    """Carga perezosa: las validaciones de entrada responden aunque no haya modelo."""
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"No hay modelo en {MODEL_PATH}. Entrenalo con: python -m ecg.train baseline-v5",
        )
    return BaselineClassifier.load(MODEL_PATH)


class PredictRequest(BaseModel):
    fs: float = Field(..., description="Frecuencia de muestreo de la señal enviada (Hz)")
    signal: list[float] = Field(..., description="Una derivación, preferentemente tipo MLII (mV)")
    r_peaks: list[int] | None = Field(
        None, description="Posiciones de los picos R. Si no se envían, se detectan."
    )


class BeatPrediction(BaseModel):
    r_peak: int = Field(..., description="Posición del pico R en la señal enviada")
    time_s: float
    beat_class: str = Field(..., alias="class")
    probabilities: dict[str, float]

    model_config = {"populate_by_name": True}


class PredictResponse(BaseModel):
    model_version: str
    model_name: str
    classes: list[str]
    out_of_scope: list[str]
    fs_input: float
    fs_model: int
    n_beats: int
    beats: list[BeatPrediction]
    warnings: list[str]
    disclaimer: str = DISCLAIMER


def validate_input(request: PredictRequest) -> np.ndarray:
    """Controles de riesgo sobre la entrada. Devuelve la señal como array."""
    if not (MIN_FS <= request.fs <= MAX_FS):  # REQ-002
        raise HTTPException(
            status_code=422,
            detail=f"Frecuencia de muestreo fuera de rango: {request.fs} Hz "
            f"(admitido: {MIN_FS:.0f}–{MAX_FS:.0f} Hz)",
        )
    if len(request.signal) > MAX_SAMPLES:
        raise HTTPException(status_code=413, detail="Señal demasiado larga")
    signal = np.asarray(request.signal, dtype=np.float64)
    if signal.size == 0 or not np.isfinite(signal).all():
        raise HTTPException(
            status_code=422, detail="La señal tiene valores no numéricos (NaN o infinito)"
        )
    if len(signal) < MIN_SECONDS * request.fs:
        raise HTTPException(
            status_code=422,
            detail=f"Señal demasiado corta: {len(signal) / request.fs:.1f} s "
            f"(mínimo {MIN_SECONDS:.0f} s)",
        )
    if request.r_peaks is not None:
        peaks = np.asarray(request.r_peaks)
        if peaks.size and (peaks.min() < 0 or peaks.max() >= len(signal)):
            raise HTTPException(
                status_code=422, detail="Hay picos R fuera de los límites de la señal"
            )
        if peaks.size and np.any(np.diff(peaks) <= 0):
            raise HTTPException(
                status_code=422, detail="Los picos R deben venir ordenados y sin repetir"
            )
    return signal


def signal_quality(x: np.ndarray, r_peaks: np.ndarray) -> float | None:
    """Calidad media de la señal (REQ-005). Devuelve None si no se puede calcular."""
    try:
        import neurokit2 as nk

        quality = nk.ecg_quality(x, rpeaks=r_peaks, sampling_rate=config.FS)
        return float(np.nanmean(quality))
    except Exception:  # noqa: BLE001 - la calidad es informativa, nunca debe tumbar la respuesta
        return None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_present": MODEL_PATH.exists()}


@app.get("/model")
def model_info() -> dict:
    model = get_model()
    return {
        "model_version": MODEL_VERSION,
        "model_name": MODEL_PATH.stem,
        "classes": model.classes,
        "out_of_scope": [c for c in config.CLASSES if c not in model.classes],
        "feature_set": model.feature_set,
        "template_block": model.template_block,
        "fs_model": config.FS,
        "lead": config.LEAD,
        "disclaimer": DISCLAIMER,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: Annotated[PredictRequest, Body()]) -> PredictResponse:
    signal = validate_input(request)
    model = get_model()
    warnings: list[str] = []

    escala = config.FS / request.fs
    x = bandpass(resample(signal, request.fs, config.FS), config.FS)
    if request.fs != config.FS:
        warnings.append(f"La señal se remuestreó de {request.fs:g} Hz a {config.FS} Hz")

    if request.r_peaks is None:
        peaks = wfdb.processing.xqrs_detect(x, fs=config.FS, verbose=False)
        warnings.append(f"Los picos R se detectaron automáticamente ({len(peaks)} latidos)")
    else:
        peaks = np.round(np.asarray(request.r_peaks) * escala).astype(np.int64)

    if len(peaks) < MIN_BEATS:
        raise HTTPException(
            status_code=422,
            detail=f"Hacen falta al menos {MIN_BEATS} latidos y se encontraron {len(peaks)}",
        )

    idx, X, F = beat_windows(x, config.FS, peaks)
    if len(idx) == 0:
        raise HTTPException(
            status_code=422, detail="Ningún latido tiene una ventana completa alrededor del pico R"
        )
    if len(idx) < len(peaks):
        warnings.append(
            f"{len(peaks) - len(idx)} latidos quedaron sin clasificar por estar en los bordes "
            "de la señal o no tener vecinos"
        )
    if len(idx) < FEW_BEATS:
        warnings.append(
            f"Solo {len(idx)} latidos: la plantilla del paciente y el contexto de ritmo son poco "
            f"confiables con menos de {FEW_BEATS} latidos"
        )

    Z, _ = build_features(X, F, model.feature_set, template_block=model.template_block)
    proba = model.predict_proba(Z)
    clases = np.asarray(model.classes)
    pred = clases[proba.argmax(axis=1)]

    # Aviso desde la señal, no desde las predicciones: si el paciente está fuera del rango de
    # frecuencias de entrenamiento, el modelo puede equivocarse "en silencio" (ver limitaciones).
    rr = np.diff(peaks) / config.FS
    mediana_rr = float(np.median(rr))
    irregulares = float(np.mean(np.abs(rr - mediana_rr) / mediana_rr > RR_DEVIATION))
    if irregulares > IRREGULAR_FRACTION:
        warnings.append(
            f"Ritmo muy irregular: el {irregulares:.0%} de los latidos se aparta más del "
            f"{RR_DEVIATION:.0%} del intervalo mediano. El modelo supone un ritmo de base regular; "
            "con arritmia sostenida puede clasificar como normales latidos que no lo son."
        )
    if not (TRAINED_RR_RANGE[0] <= mediana_rr <= TRAINED_RR_RANGE[1]):
        warnings.append(
            f"Frecuencia cardíaca mediana de {60 / mediana_rr:.0f} lpm, fuera del rango visto en "
            f"entrenamiento ({60 / TRAINED_RR_RANGE[1]:.0f}–{60 / TRAINED_RR_RANGE[0]:.0f} lpm). "
            "En estos pacientes el modelo detecta muchos menos latidos supraventriculares."
        )

    calidad = signal_quality(x, peaks)
    if calidad is not None and calidad < 0.6:  # REQ-005
        warnings.append(f"Calidad de señal baja (índice {calidad:.2f} de 1.0)")

    anormales = float(np.mean(pred != "N"))
    if anormales > ABNORMAL_FRACTION:
        warnings.append(
            f"El {anormales:.0%} de los latidos se clasificó como anormal. El modelo supone que el "
            "ritmo dominante del paciente es normal; con arritmia sostenida su rendimiento cae "
            "mucho (ver limitaciones)."
        )

    beats = [
        BeatPrediction(
            r_peak=int(round(peaks[i] / escala)),
            time_s=round(float(peaks[i]) / config.FS, 4),
            **{"class": str(c)},
            probabilities={k: round(float(v), 4) for k, v in zip(model.classes, p, strict=True)},
        )
        for i, c, p in zip(idx, pred, proba, strict=True)
    ]
    return PredictResponse(
        model_version=MODEL_VERSION,
        model_name=MODEL_PATH.stem,
        classes=list(model.classes),
        out_of_scope=[c for c in config.CLASSES if c not in model.classes],
        fs_input=request.fs,
        fs_model=config.FS,
        n_beats=len(beats),
        beats=beats,
        warnings=warnings,
    )
