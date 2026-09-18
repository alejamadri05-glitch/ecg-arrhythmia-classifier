# SOUP (software de origen no conocido)

*SOUP* es el término de IEC 62304 para el software de terceros que se incorpora al producto sin
haberlo desarrollado ni validado uno mismo. La norma pide inventariarlo: qué es, qué versión, para
qué se usa y qué pasa si falla.

Las versiones son las **verificadas en la imagen de Docker** publicada el 2026-09-17 (Python
3.12-slim, arm64). `tests/test_requirements.py::test_soup_lists_every_dependency` comprueba que no
falte ninguna dependencia en esta tabla; las versiones exactas se revisan al publicar.

## En la imagen de la API

Es lo que corre en producción. PyTorch **no** está: la API sirve el modelo de XGBoost.

| Paquete | Versión | Para qué se usa | Si falla o cambia |
|---|---|---|---|
| `xgboost` | 3.4.1 | El clasificador que se sirve | Sin predicciones. Un cambio de formato del modelo guardado lo haría incargable; se mitiga fijando la versión y con `test_save_and_load_roundtrip` |
| `scikit-learn` | 1.9.1 | Métricas, particiones por paciente, utilidades del modelo | Entrenamiento y evaluación incorrectos; una partición mal hecha rompería la separación inter-paciente (REQ-010, verificado por pruebas) |
| `numpy` | 2.5.3 | Todo el cálculo numérico | El sistema no funciona |
| `scipy` | 1.18.1 | Filtro Butterworth y remuestreo | Señal mal filtrada: predicciones degradadas de forma silenciosa (RISK-04) |
| `wfdb` | 4.3.1 | Lectura de registros de PhysioNet y detección de picos R (`xqrs_detect`) | Sin detección automática de picos R (REQ-007); la lectura de datos es solo de entrenamiento |
| `neurokit2` | 0.2.13 | Índice de calidad de señal (REQ-005) | El aviso de calidad deja de emitirse. **Control implementado:** si falla, se captura y se devuelve `None`; la predicción no se interrumpe |
| `pandas` | 2.3.3 | Tablas de métricas y reportes | Solo afecta reportes, no la predicción |
| `matplotlib` | 3.11.2 | Figuras de los notebooks y reportes | Solo afecta figuras |
| `fastapi` | 0.141.1 | Framework de la API | La API no levanta |
| `pydantic` | 2.13.5 | Validación del esquema de entrada | Primera línea de defensa de REQ-008; un fallo dejaría pasar entradas mal formadas |
| `uvicorn` | 0.53.0 | Servidor ASGI | La API no levanta |
| `starlette` | 1.6.0 | Base de FastAPI (dependencia indirecta) | La API no levanta |
| `joblib` | 1.6.0 | Serialización del modelo (dependencia de scikit-learn) | El modelo no carga |
| `pywavelets`, `soundfile` | 1.10.0, 0.14.0 | Dependencias indirectas de neurokit2 | Igual que neurokit2 |

## Solo en desarrollo

No se distribuyen en la imagen.

| Paquete | Versión | Para qué se usa | Si falla |
|---|---|---|---|
| `torch` | ≥ 2.2 (extra `dl`) | CNN 1D: entrenamiento y comparación. **No se sirve**, quedó descartada frente al baseline | Nada en producción |
| `streamlit` | ≥ 1.33 (extra `app`) | Demo interactiva | Solo la demo |
| `pytest` | ≥ 8 | Verificación de todos los requisitos | Sin verificación automática: es el riesgo mayor de esta tabla |
| `ruff` | ≥ 0.4 | Formato y linter | Deterioro de calidad, sin efecto funcional |
| `httpx` | ≥ 0.27 | Cliente de prueba de la API | Solo pruebas |
| `jupyter`, `nbconvert`, `ipykernel` | — | Ejecución de los notebooks | Solo reproducción de notebooks |

## Sistema base

| Componente | Versión | Nota |
|---|---|---|
| Imagen base | `python:3.12-slim` | Debian slim; se actualiza al reconstruir |
| `libgomp1` | del sistema | OpenMP, requerido por XGBoost. Sin ella el modelo no carga |
| Python | 3.12 en la imagen; 3.11 en CI | El paquete declara `requires-python >= 3.11` |

## Datos de terceros

| Fuente | Uso | Control |
|---|---|---|
| MIT-BIH Arrhythmia Database (PhysioNet) | Entrenamiento y evaluación | Descarga verificada por SHA-256 contra el manifiesto oficial (REQ-013) |
| INCART, SVDB (PhysioNet) | Validación externa | Descarga con reintentos; `external.verify()` detecta archivos corruptos |

Ambas bases son de acceso abierto (Open Data Commons Attribution) y se citan en el README.
