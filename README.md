# Clasificador de arritmias ECG (MIT-BIH)

Clasifica cada latido de un ECG en las 4 clases AAMI (**N, S, V, F**) con evaluación
**inter-paciente** (split de de Chazal DS1/DS2).

> **Aviso:** proyecto educativo y de investigación. No es un dispositivo médico ni debe usarse
> para decisiones clínicas.

🚧 En construcción: fases 1 a 4 completas (EDA, segmentación, baseline y CNN).

## Inicio rápido

```bash
python3.11 -m venv .venv && source .venv/bin/activate   # 3.11 o 3.12
pip install -e ".[api,app,dev]"
python -m ecg.segment          # descarga MIT-BIH (~100 MB) y genera data/processed/ds{1,2}.npz
python -m ecg.train baseline   # validación cruzada por paciente en DS1 + modelo final
python -m ecg.train cnn        # ídem para la CNN (usa GPU de Apple/CUDA si hay; ~5 min en M2)
pytest -q
```

**macOS:** XGBoost necesita OpenMP: `brew install libomp`.

## Estado

| Fase | Contenido | Estado |
|---|---|---|
| 1 | Descarga y EDA — [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) | ✅ |
| 2 | Filtro, segmentación y features RR — [`src/ecg/segment.py`](src/ecg/segment.py) | ✅ |
| 3 | Baseline (Random Forest / XGBoost) — [`notebooks/02_baseline.ipynb`](notebooks/02_baseline.ipynb) | ✅ |
| 4 | CNN 1D (PyTorch) — [`notebooks/03_cnn.ipynb`](notebooks/03_cnn.ipynb) | ✅ |
| 5 | Evaluación final en DS2 | ⏳ |
| 6 | API FastAPI + Docker + demo Streamlit | ⏳ |
| 7 | Documentación estilo IEC 62304 / ISO 14971 | ⏳ |
| 8 | README final y publicación | ⏳ |

Latidos extraídos (difieren de los publicados por de Chazal et al. en ≤ 42 por clase: el primer
y último latido de cada registro no tienen RR previo/siguiente):

| Split | N | S | V | F | Total |
|---|---|---|---|---|---|
| DS1 (entrenamiento) | 45 824 | 943 | 3 788 | 414 | 50 969 |
| DS2 (prueba) | 44 218 | 1 836 | 3 219 | 388 | 49 661 |

### Baseline: validación cruzada por paciente (solo DS1)

XGBoost · 41 features (RR + morfología + latido submuestreado) · pesos de clase balanceados.
Elegido entre 24 experimentos por F1 macro out-of-fold con `GroupKFold` de 5 folds por registro
([tabla completa](reports/baseline_experiments.csv)).

| Clase | Se | +P | F1 |
|---|---|---|---|
| N | 0.916 | 0.980 | 0.947 |
| S | 0.319 | 0.445 | 0.372 |
| V | 0.914 | 0.509 | 0.654 |
| F | 0.000 | 0.000 | 0.000 |
| **F1 macro** | | | **0.493** |

Estas no son las métricas finales: DS2 se evalúa una sola vez en la Fase 5. La exactitud
(0.897) es menor que la de predecir siempre N (0.899). Por eso se reportan Se y +P por clase.

Modos de falla, verificados con las anotaciones de ritmo de MIT-BIH:

- **Bloqueo de rama izquierda con QRS invertido** (registro 207): 1 399 latidos L→V. El LBBB del
  109 tiene otra polaridad.
- **Fibrilación/flutter auricular:** N→V en el 203 y N→S en el 201, porque el RR irregular hace
  que los latidos normales parezcan prematuros.
- **Rachas de SVTA** (209): el RR local también es corto y el latido prematuro deja de parecerlo
  (Se de S 0.22 en SVTA vs 0.71 en ritmo normal).
- **Clase F:** no se detecta. Casi toda está en un paciente (208), así que el modelo no tiene de
  quién aprenderla.

### CNN 1D vs baseline (validación por paciente, solo DS1)

Misma validación y mismos folds que el baseline. El early stopping usa pacientes internos,
separados del fold que se evalúa. Se probaron 6 variantes con 2 semillas cada una
([tabla](reports/cnn_experiments.csv)): con y sin RR, sampler vs pesos, aumento de datos e
inversión de polaridad.

| Modelo | F1 macro |
|---|---|
| Baseline XGBoost | **0.493** |
| Baseline XGBoost con los mismos pacientes que la CNN | 0.474 |
| CNN (cocientes RR + pesos + aumento), 3 semillas | 0.463 ± 0.027 |
| Ensamble de 3 semillas (referencia) | 0.480 |

- **La CNN no le gana al baseline.** Gran parte de la brecha aparente viene de que la CNN reserva
  un cuarto de los pacientes para el early stopping. En igualdad de pacientes, la diferencia
  (0.011) queda por debajo del desvío entre semillas.
- **Se equivocan en cosas distintas.** La CNN reduce los falsos V durante fibrilación auricular
  (1 320 → ~400 en el 203) y los V no detectados (149 → ~27 en el 215). En cambio, la CNN elegida no
  reconoce el bloqueo de rama izquierda de ningún paciente (acierta < 1 % en el 109, contra 97 % del
  baseline), da muchas más falsas F y tiene fallas en pacientes puntuales que **dependen de la
  semilla**.
- El early stopping por F1 macro es ruidoso porque el rendimiento en pacientes nuevos oscila entre
  épocas. Lo analicé con varios estabilizadores
  ([resultados](reports/cnn_early_stopping_check.txt)).

**Modelo primario para DS2:** baseline XGBoost, según la regla fijada de antemano
([`reports/model_selection.json`](reports/model_selection.json)). La CNN se reporta como
comparación.

## Licencia

El código está bajo licencia [MIT](LICENSE). Los datos de MIT-BIH **no** se incluyen en el
repositorio y tienen su propia licencia (ODC-By, ver abajo).

## Datos y citas

MIT-BIH Arrhythmia Database, [PhysioNet](https://physionet.org/content/mitdb/) (ODC-By).

- Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database.* IEEE Eng Med Biol. 2001.
- Goldberger AL et al. *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation. 2000.
- de Chazal P, O'Dwyer M, Reilly RB. *Automatic classification of heartbeats using ECG
  morphology and heartbeat interval features.* IEEE Trans Biomed Eng. 2004.
