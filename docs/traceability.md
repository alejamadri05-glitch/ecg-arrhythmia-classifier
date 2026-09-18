# Matriz de trazabilidad

Conecta cada requisito de [`requirements.md`](requirements.md) con el código que lo implementa, la
prueba que lo verifica y el riesgo de [`risk_analysis.md`](risk_analysis.md) que controla.

Esta matriz **se verifica sola**: `tests/test_requirements.py` comprueba que cada requisito esté
trazado, que cada archivo y cada función citados existan, y que no se cite un riesgo inexistente.
Si alguien renombra una prueba y no actualiza esta tabla, el CI falla.

| Requisito | Implementación | Verificación | Riesgo |
|---|---|---|---|
| REQ-001 | `api/main.py::predict`, `src/ecg/models/baseline.py::BaselineClassifier` | `tests/test_api.py::test_predictions_are_well_formed` | RISK-07 |
| REQ-002 | `api/main.py::validate_input` | `tests/test_api.py::test_rejects_bad_fs` | RISK-03 |
| REQ-003 | `src/ecg/train.py::train_baseline`, `reports/baseline_v5_cv.json` | `tests/test_requirements.py::test_v_sensitivity_meets_the_declared_threshold` | RISK-01 |
| REQ-004 | `api/main.py::model_info`, `api/main.py::predict` | `tests/test_api.py::test_response_carries_model_version_and_scope` | RISK-08 |
| REQ-005 | `api/main.py::signal_quality` | `tests/test_api.py::test_warns_when_signal_quality_is_low`, `tests/test_api.py::test_quality_failure_does_not_break_the_prediction` | RISK-04 |
| REQ-006 | `src/ecg/preprocess.py::resample` | `tests/test_api.py::test_resamples_and_warns_when_fs_differs` | RISK-03 |
| REQ-007 | `api/main.py::predict` | `tests/test_api.py::test_detects_r_peaks_when_missing` | RISK-04 |
| REQ-008 | `api/main.py::validate_input` | `tests/test_api.py::test_rejects_short_signal`, `tests/test_api.py::test_rejects_non_numeric_signal`, `tests/test_api.py::test_rejects_unsorted_or_out_of_range_peaks`, `tests/test_api.py::test_rejects_too_few_beats` | RISK-03 |
| REQ-009 | `api/main.py::predict` | `tests/test_api.py::test_warns_when_rhythm_is_irregular`, `tests/test_api.py::test_warns_when_heart_rate_is_outside_training_range` | RISK-05, RISK-06 |
| REQ-010 | `src/ecg/config.py`, `src/ecg/train.py::patient_folds` | `tests/test_config.py::test_no_record_shared_between_ds1_and_ds2`, `tests/test_config.py::test_paced_records_excluded_from_both_splits`, `tests/test_baseline.py::test_patient_folds_never_share_a_record` | RISK-05 |
| REQ-011 | `src/ecg/models/baseline.py::BaselineClassifier`, `src/ecg/models/cnn.py::CNNClassifier` | `tests/test_baseline.py::test_same_seed_same_predictions`, `tests/test_cnn.py::test_same_seed_same_output_on_cpu`, `tests/test_evaluate_ds2.py::test_predictions_are_deterministic` | RISK-08 |
| REQ-012 | `api/main.py::model_info`, `api/main.py::predict` | `tests/test_api.py::test_disclaimer_is_in_every_response` | RISK-10 |
| REQ-013 | `src/ecg/data.py::extract_verified`, `src/ecg/data.py::with_retry` | `tests/test_data.py::test_zip_adulterado_no_escribe_nada`, `tests/test_data.py::test_falta_un_archivo_de_registro`, `tests/test_data.py::test_sin_sha256sums_no_se_acepta`, `tests/test_data.py::test_ignora_rutas_que_escapan_de_la_carpeta` | RISK-09 |
| REQ-014 | `Dockerfile`, `.github/workflows/docker.yml` | Workflow `docker`: construye la imagen, la levanta y le exige `/health`, `/model`, una predicción sobre el registro 100 y un rechazo 422 | RISK-03 |
| REQ-015 | `.github/workflows/ci.yml`, `pyproject.toml` | Workflow `ci`: `ruff check`, `ruff format --check` y `pytest` | — |

## Cobertura de riesgos

Cada riesgo con al menos un control verificado por una prueba:

| Riesgo | Requisitos que lo controlan |
|---|---|
| RISK-01 (V no detectado) | REQ-003 |
| RISK-02 (falsas alarmas) | Decisiones de diseño documentadas en el análisis de riesgo; se vigila con las métricas de +P, no con una prueba automática |
| RISK-03 (entrada inválida) | REQ-002, REQ-006, REQ-008, REQ-014 |
| RISK-04 (ruido o derivación distinta) | REQ-005, REQ-007 |
| RISK-05 (paciente distinto al entrenamiento) | REQ-009, REQ-010 |
| RISK-06 (la arritmia es el ritmo de base) | REQ-009 |
| RISK-07 (clase F fuera de alcance) | REQ-001 |
| RISK-08 (modelo servido sin validación externa) | REQ-004, REQ-011 |
| RISK-09 (datos corruptos) | REQ-013 |
| RISK-10 (uso fuera de alcance) | REQ-012 |

## Registros de verificación

Evidencia guardada en el repositorio, no solo afirmaciones:

| Evidencia | Archivo |
|---|---|
| Validación cruzada del modelo servido (v5) | `reports/baseline_v5_cv.json` |
| Evaluación única en DS2 (modelo v1) | `reports/ds2_results.json` |
| Validación externa en INCART | `reports/incart_results.json` |
| Validación externa en SVDB | `reports/svdb_results.json` |
| Regla de selección declarada antes de ver DS2 | `reports/model_selection.json` |
