"""Pruebas de la API. Las validaciones de entrada son controles de riesgo (ver REQ-002/005)."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import MIN_SECONDS, MODEL_PATH, app
from ecg import config

client = TestClient(app)
HAY_MODELO = MODEL_PATH.exists()
needs_model = pytest.mark.skipif(not HAY_MODELO, reason="hace falta el modelo entrenado")


def ecg_sintetico(segundos: float = 30.0, fs: float = 360.0, lpm: float = 75.0):
    """ECG de juguete: picos gaussianos regulares. Alcanza para ejercitar la API."""
    n = int(segundos * fs)
    t = np.arange(n) / fs
    periodo = 60.0 / lpm
    picos = np.arange(periodo / 2, segundos, periodo)
    señal = sum(np.exp(-0.5 * ((t - p) / 0.012) ** 2) for p in picos)
    return señal.tolist(), [int(p * fs) for p in picos]


def test_health_no_necesita_modelo():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


@pytest.mark.parametrize("fs", [50.0, 99.0, 2001.0, 5000.0])
def test_rejects_bad_fs(fs):
    """REQ-002: frecuencia de muestreo fuera del rango permitido."""
    señal, _ = ecg_sintetico(segundos=30, fs=360)
    r = client.post("/predict", json={"fs": fs, "signal": señal})
    assert r.status_code == 422
    assert "Frecuencia de muestreo fuera de rango" in r.json()["detail"]


def test_rejects_short_signal():
    señal, _ = ecg_sintetico(segundos=MIN_SECONDS / 2)
    r = client.post("/predict", json={"fs": 360.0, "signal": señal})
    assert r.status_code == 422 and "demasiado corta" in r.json()["detail"]


def test_rejects_non_numeric_signal():
    cuerpo = json.dumps({"fs": 360.0, "signal": [float("nan")] * 4000})
    r = client.post("/predict", content=cuerpo, headers={"content-type": "application/json"})
    assert r.status_code == 422 and "no numéricos" in r.json()["detail"]


def test_rejects_unsorted_or_out_of_range_peaks():
    señal, _ = ecg_sintetico()
    desordenados = client.post(
        "/predict", json={"fs": 360.0, "signal": señal, "r_peaks": [500, 200, 900]}
    )
    assert desordenados.status_code == 422 and "ordenados" in desordenados.json()["detail"]
    fuera = client.post(
        "/predict", json={"fs": 360.0, "signal": señal, "r_peaks": [10, len(señal) + 5]}
    )
    assert fuera.status_code == 422 and "fuera de los límites" in fuera.json()["detail"]


def test_rejects_too_few_beats():
    señal, _ = ecg_sintetico()
    r = client.post("/predict", json={"fs": 360.0, "signal": señal, "r_peaks": [1000, 2000]})
    assert r.status_code == 422 and "al menos" in r.json()["detail"]


@needs_model
def test_response_carries_model_version_and_scope():
    """REQ-004: cada respuesta incluye la versión del modelo."""
    señal, picos = ecg_sintetico()
    r = client.post("/predict", json={"fs": 360.0, "signal": señal, "r_peaks": picos})
    assert r.status_code == 200
    d = r.json()
    assert d["model_version"] and d["model_name"]
    assert d["classes"] == ["N", "S", "V"] and d["out_of_scope"] == ["F"]
    assert "no es un dispositivo médico" in d["disclaimer"].lower()
    assert d["n_beats"] == len(d["beats"]) > 0


@needs_model
def test_predictions_are_well_formed():
    señal, picos = ecg_sintetico()
    d = client.post("/predict", json={"fs": 360.0, "signal": señal, "r_peaks": picos}).json()
    for beat in d["beats"]:
        assert beat["class"] in config.CLASSES
        assert set(beat["probabilities"]) == {"N", "S", "V"}
        assert sum(beat["probabilities"].values()) == pytest.approx(1.0, abs=0.01)
        assert 0 <= beat["r_peak"] < len(señal)
    posiciones = [b["r_peak"] for b in d["beats"]]
    assert posiciones == sorted(posiciones)


@needs_model
def test_resamples_and_warns_when_fs_differs():
    señal, picos = ecg_sintetico(fs=250.0)
    d = client.post("/predict", json={"fs": 250.0, "signal": señal, "r_peaks": picos}).json()
    assert d["fs_input"] == 250.0 and d["fs_model"] == config.FS
    assert any("remuestre" in w for w in d["warnings"])
    # las posiciones vuelven en el espacio de la señal enviada, no en el remuestreado
    assert max(b["r_peak"] for b in d["beats"]) < len(señal)


@needs_model
def test_detects_r_peaks_when_missing():
    señal, picos = ecg_sintetico()
    d = client.post("/predict", json={"fs": 360.0, "signal": señal}).json()
    assert any("detectaron automáticamente" in w for w in d["warnings"])
    assert abs(d["n_beats"] - len(picos)) <= 3


@needs_model
def test_warns_when_rhythm_is_irregular():
    """El caso que falla en silencio: arritmia sostenida (registro 232 de DS2)."""
    fs, lpm = 360.0, 75.0
    señal, picos = ecg_sintetico(segundos=60, fs=fs, lpm=lpm)
    irregulares = [p for i, p in enumerate(picos) if i % 2 == 0 or i % 3 == 0]
    d = client.post("/predict", json={"fs": fs, "signal": señal, "r_peaks": irregulares}).json()
    assert any("irregular" in w for w in d["warnings"])


@needs_model
def test_warns_when_there_are_few_beats():
    señal, picos = ecg_sintetico(segundos=12)
    d = client.post("/predict", json={"fs": 360.0, "signal": señal, "r_peaks": picos}).json()
    assert any("poco confiables" in w for w in d["warnings"])


def test_input_validation_does_not_need_the_model():
    """En CI no hay modelo entrenado: una entrada inválida debe fallar por su motivo real (422),
    no por falta de modelo (503)."""
    señal, _ = ecg_sintetico()
    casos = [
        ({"fs": 50.0, "signal": señal}, "Frecuencia de muestreo"),
        ({"fs": 360.0, "signal": señal[:100]}, "demasiado corta"),
        ({"fs": 360.0, "signal": señal, "r_peaks": [1000, 2000]}, "al menos"),
        ({"fs": 360.0, "signal": señal, "r_peaks": [900, 500]}, "ordenados"),
    ]
    for payload, esperado in casos:
        r = client.post("/predict", json=payload)
        assert r.status_code == 422, (payload.get("r_peaks"), r.status_code, r.json())
        assert esperado in r.json()["detail"]


@needs_model
def test_warns_when_heart_rate_is_outside_training_range():
    """REQ-009: en pacientes con frecuencia atípica el modelo detecta muchos menos S (RISK-05)."""
    señal, picos = ecg_sintetico(segundos=60, lpm=40.0)  # RR 1.5 s; DS1 llega hasta 1.12 s
    d = client.post("/predict", json={"fs": 360.0, "signal": señal, "r_peaks": picos}).json()
    assert any("fuera del rango visto en entrenamiento" in w for w in d["warnings"])


@needs_model
def test_warns_when_signal_quality_is_low(monkeypatch):
    """REQ-005: la calidad la calcula neurokit2; acá se fuerza un valor bajo."""
    monkeypatch.setattr("api.main.signal_quality", lambda x, r_peaks: 0.2)
    señal, picos = ecg_sintetico()
    d = client.post("/predict", json={"fs": 360.0, "signal": señal, "r_peaks": picos}).json()
    assert any("Calidad de señal baja" in w for w in d["warnings"])


def test_quality_failure_does_not_break_the_prediction():
    """El índice de calidad es informativo: si no se puede calcular, devuelve None en vez de
    tumbar la respuesta (la predicción no depende de él)."""
    from api.main import signal_quality

    assert signal_quality(np.zeros(5), np.array([1, 2])) is None


def test_disclaimer_is_in_every_response():
    """REQ-012: el aviso de que no es un dispositivo médico viaja con cada respuesta."""
    señal, picos = ecg_sintetico()
    r = client.post("/predict", json={"fs": 360.0, "signal": señal, "r_peaks": picos})
    if r.status_code == 200:  # sin modelo entrenado (CI) solo se comprueba /model
        assert "no es un dispositivo médico" in r.json()["disclaimer"].lower()
    if HAY_MODELO:
        assert "no es un dispositivo médico" in client.get("/model").json()["disclaimer"].lower()


def test_landing_page_works_without_model():
    """La página de inicio no debe depender de que haya un modelo entrenado."""
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "No es un dispositivo médico" in r.text  # REQ-012
    for ruta in ("/docs", "/health", "/model"):
        assert f'href="{ruta}"' in r.text


def test_landing_page_is_not_in_the_openapi_schema():
    """Es una página, no un endpoint de la API: no debe ensuciar /docs."""
    esquema = client.get("/openapi.json").json()
    assert "/" not in esquema["paths"]
    assert set(esquema["paths"]) == {"/health", "/model", "/predict"}
