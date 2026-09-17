# Clasificador de arritmias ECG (MIT-BIH)

Clasifica cada latido de un ECG en las 4 clases AAMI (**N, S, V, F**) con evaluación
**inter-paciente** (split de de Chazal DS1/DS2).

> **Aviso:** proyecto educativo y de investigación. No es un dispositivo médico ni debe usarse
> para decisiones clínicas.

🚧 En construcción: fases 1 a 5 completas. Resultado final en DS2 más abajo.

## Inicio rápido

```bash
python3.11 -m venv .venv && source .venv/bin/activate   # 3.11 o 3.12
pip install -e ".[api,app,dev]"
python -m ecg.segment          # descarga MIT-BIH (~100 MB) y genera data/processed/ds{1,2}.npz
python -m ecg.train baseline   # validación cruzada por paciente en DS1 + modelo final
python -m ecg.train cnn        # ídem para la CNN (usa GPU de Apple/CUDA si hay; ~5 min en M2)
python -m ecg.evaluate_ds2     # evaluación final en DS2 (reportes y figuras)
python -m ecg.train baseline-v2  # versión 2: RR normalizado por paciente
python -m ecg.external --db incart   # descarga INCART (~820 MB)
python -m ecg.external --db svdb     # descarga SVDB (~54 MB)
python -m ecg.evaluate_external                          # INCART: v1 vs v2 vs CNN
python -m ecg.evaluate_external --db svdb --classes NSV \
    --models v1_rr_absoluto,v2_rr_normalizado,v3_3clases,v4_morfologia_relativa
python -m ecg.train baseline-v3  # versión 3: 3 clases (F fuera de alcance)
python -m ecg.train baseline-v4  # versión 4: morfología relativa al paciente
python -m ecg.train baseline-v5  # versión 5: + contexto de la secuencia de latidos
pytest -q
```

**macOS:** XGBoost necesita OpenMP: `brew install libomp`.

## Resultado final (DS2, 22 pacientes nunca vistos)

Modelo primario: **XGBoost** con features de RR + morfología, elegido por validación cruzada por
paciente dentro de DS1 **antes** de mirar DS2, que se evaluó una sola vez.

| Clase | Se | +P | F1 |
|---|---|---|---|
| N (normal) | 0.905 | 0.962 | 0.933 |
| S (supraventricular) | 0.155 | 0.221 | 0.182 |
| V (ventricular) | **0.964** | **0.852** | **0.905** |
| F (fusión) | 0.137 | 0.017 | 0.030 |
| **F1 macro** | | | **0.512** |

Exactitud 0.876, contra 0.890 de predecir siempre N: por eso la métrica principal es Se y +P por
clase, no la exactitud. La CNN queda en F1 macro 0.49, con muchas más falsas alarmas
(exactitud 0.704); detalle en [`notebooks/04_ds2_evaluation.ipynb`](notebooks/04_ds2_evaluation.ipynb).

**Lo importante de estos números**

- **La validación no fue optimista.** La F1 macro pasó de 0.493 en validación cruzada a 0.512 en
  DS2. Ese es el punto de separar pacientes desde el principio, y la razón por la que este
  proyecto no reporta el 99 % habitual de los tutoriales.
- **V, la clase clínicamente más importante, funciona:** solo 58 de 3 219 latidos ventriculares se
  leyeron como normales.
- **S falla, y se sabe exactamente por qué.** El registro 232 aporta el 75 % de los S de DS2, es
  bradicárdico, y el modelo usa intervalos RR **absolutos**: sus latidos prematuros (RR 0.73 s)
  parecen normales para un modelo entrenado con pacientes cuyos latidos normales están en 0.76 s.
  El EDA lo había anticipado y quedó registrado como riesgo antes de evaluar. La CNN, que usa solo
  **cocientes** de RR, detecta el 55.9 % de esos latidos.
- **F es inservible** (+P 0.017): casi toda la clase F de entrenamiento está en un solo paciente.

### Versión 2 del modelo: RR normalizado por paciente

El fallo con la clase S en DS2 tiene una causa concreta y corregible, así que hay una
**versión 2** ([`notebooks/05_model_v2.ipynb`](notebooks/05_model_v2.ipynb)): en vez de usar los
intervalos RR en segundos, usa **RR dividido por la mediana del RR del propio registro**. Esa
mediana se calcula con la señal de entrada, nunca con las etiquetas, así que la API puede hacer
lo mismo con el ECG que recibe.

Se eligió entre tres candidatos con una regla declarada antes de correr los experimentos: mayor
F1 macro en validación por paciente y, si quedaban dentro de 0.01 (que fue el caso), desempate por
Se de S.

| | v1 (CV DS1) | v2 (CV DS1) |
|---|---|---|
| F1 macro | 0.493 | 0.487 |
| **Se de S** | 0.319 | **0.370** |
| +P de V | 0.509 | 0.480 |

Prueba de estrés, multiplicando los intervalos RR para simular un paciente más lento:

| Escala de RR | ×1 | ×1.5 | ×2 |
|---|---|---|---|
| Se de S, v1 | 0.319 | 0.247 | **0.133** |
| Se de S, v2 | 0.370 | 0.370 | **0.370** |

**Por qué no se mide la v2 en DS2:** ese conjunto ya se usó una vez y, sobre todo, la v2 se
diseñó a partir de lo que se vio ahí. Medirla en DS2 sería elegir el modelo mirando el conjunto
de prueba. Por eso la comparación se hizo con una base externa (abajo), y **el resultado
publicado de DS2 sigue siendo el de v1**.

### Validación externa: INCART

Para comparar v1 y v2 de forma limpia hace falta una base que ninguna de las dos versiones haya
influido. Se usó **INCART** (St Petersburg, PhysioNet): 75 pacientes, 175 718 latidos, 257 Hz y
**sin derivación MLII** (se usa la II estándar). Los modelos siguen entrenados solo con DS1
([`notebooks/06_external_validation.ipynb`](notebooks/06_external_validation.ipynb)).

| Modelo | F1 macro | Se de S | +P de V | Errores |
|---|---|---|---|---|
| v1 (RR absoluto) | 0.617 | 0.794 | 0.905 | 7 323 |
| **v2 (RR normalizado)** | **0.634** | **0.843** | **0.917** | **6 387 (−12.8 %)** |
| CNN 1D | 0.464 | 0.639 | 0.652 | — |

**Lo que muestra, con sus matices:**

- **v2 es mejor que v1 en pacientes nuevos**, y por eso es el modelo por defecto. Pero la ventaja
  **no es uniforme**: gana en 34 de 75 pacientes y la prueba de Wilcoxon pareada por paciente no
  llega a significancia (p = 0.75). La mejora agregada viene de ganancias grandes en pocos
  pacientes.
- **La hipótesis original estaba incompleta.** Se esperaba que v2 ayudara con pacientes lentos
  (el registro 232 de DS2), pero su ventaja correlaciona con frecuencias **altas**
  (Spearman ρ = −0.28, p = 0.013). El mecanismo es el mismo —depender de la frecuencia basal—
  pero actúa en las dos direcciones: a 135 lpm todos los latidos le parecen prematuros a v1.
- **Los dos modelos rinden mejor acá que en DS2** (0.63 contra 0.51), pese al cambio de derivación.
  No es que INCART sea más fácil: en DS2 el 75 % de los latidos S venía de un solo paciente
  bradicárdico. **Lo que domina la métrica es qué pacientes hay, no el modelo.**
- **La CNN queda última en los tres conjuntos**, siempre por falsas alarmas.

### Versión 3: sacar la clase F y probar calibración de umbrales

A partir del análisis de errores de la v2 se probaron las dos correcciones más baratas
([`notebooks/07_calibration.ipynb`](notebooks/07_calibration.ipynb)), decididas y validadas solo
con validación cruzada por paciente en DS1:

| Variante | F1 macro (3 clases) | V no detectados (RISK-01) | Errores totales |
|---|---|---|---|
| v2, con la salida restringida a N/S/V | 0.644 | 311 | 5 137 |
| **v3: 3 clases, sin F** | **0.649** | **298** | 5 050 |
| v3 + umbrales calibrados | 0.654 | 377 | 4 657 |

- **Sacar F: sí.** Cambia poco la métrica, pero elimina una clase cuya +P nunca superó 0.09 y que
  producía miles de falsas alarmas (3 049 en DS2). Pasa a estar declarada fuera del alcance.
- **Calibrar umbrales: no.** Reduce los errores totales un 7.8 %, pero **deja un 27 % más de
  latidos ventriculares sin detectar**, que es el error de mayor severidad en el análisis de
  riesgo. Además, los pesos óptimos son inestables entre folds (el de S va de 1.6 a 4.0). Queda
  implementada y desactivada (`--calibrate`), con el intercambio documentado.

**Una lección de método que salió de acá:** la primera corrida indicaba que sacar F empeoraba todo.
Era un artefacto: `GroupKFold` equilibra por cantidad de latidos, así que al filtrar los latidos F
**9 de 22 pacientes cambiaban de fold**. El reparto quedó congelado en `config.DS1_FOLD_MAP`, con
una prueba que lo verifica.

### Versión 4: morfología relativa al paciente

La mejora más grande del proyecto ([`notebooks/08_patient_relative.ipynb`](notebooks/08_patient_relative.ipynb)).
Es la misma idea que funcionó con el ritmo, aplicada a la forma: en vez de describir el latido en
absoluto, se lo describe **relativo al latido dominante de esa persona** (la mediana de sus
latidos, calculada sin usar etiquetas y refrescada cada 100 latidos).

| | v3 | **v4** |
|---|---|---|
| F1 macro (3 clases) | 0.649 | **0.725** |
| Se de V | 0.914 | **0.953** |
| **+P de V** | 0.471 | **0.832** |
| Latidos V no detectados (RISK-01) | 298 | **166** |
| Errores totales | 5 050 | **2 355** |
| F1 de S | 0.379 | 0.311 |

**Resuelve el peor modo de falla del proyecto.** Los latidos con bloqueo de rama izquierda del
registro 207 pasan de **2.7 % a 99.9 %** correctos, y sus falsas alarmas de 1 419 a 2: esos latidos
son anchos y raros para la población, pero son la morfología **normal de ese paciente**.

**Con dos costos honestos:** la clase S empeora un poco (su morfología es normal por definición,
así que estas features no la ayudan), y el método supone que la morfología habitual del paciente es
mayoría — en INCART hay 5 registros con más del 40 % de latidos anormales.

**Cuánta señal necesita:** una plantilla calculada solo al inicio **no sirve** (con los primeros 30
latidos el resultado es el de v3), porque la morfología deriva durante la grabación. Con bloques
locales de 20 latidos ya alcanza. La variante **causal**, apta para tiempo real, obtiene 0.715.

**Confirmada en una base nueva:** ver la sección siguiente.

### Confirmación externa de la v4 en SVDB

Como la v4 se eligió en DS1, hacía falta confirmarla donde no se decidió nada. DS2 se gastó en la
Fase 5 e INCART en la comparación v1 vs v2, así que la confirmación va sobre **SVDB**
(MIT-BIH Supraventricular Arrhythmia Database): 78 pacientes, 184 324 latidos, **128 Hz** y canales
**ECG1/ECG2 sin identificar** — el cambio de dominio más fuerte de los tres. Tiene 6.6 % de latidos
S, contra 1.85 % en DS1: la prueba real para la clase floja
([`notebooks/09_svdb_validation.ipynb`](notebooks/09_svdb_validation.ipynb)).

| Modelo | F1 macro | F1 de S | F1 de V | +P de V | Errores |
|---|---|---|---|---|---|
| v1 (RR absoluto) | 0.596 | 0.340 | 0.518 | 0.365 | 25 652 |
| v2 (RR normalizado) | 0.621 | 0.380 | 0.539 | 0.388 | 22 039 |
| v3 (3 clases) | 0.612 | 0.367 | 0.526 | 0.376 | 22 702 |
| **v4 (morfología relativa)** | **0.698** | **0.408** | **0.724** | **0.607** | **14 878 (−34 %)** |

- **La mejora se confirma y además es consistente entre pacientes:** v4 gana en **47 de 78**, con
  Wilcoxon pareado **p = 0.008** (la v2 en INCART no había llegado a significancia). Las ganancias
  suman 8 626 errores contra 802 de pérdidas.
- Los falsos V caen un 69 % (10 602 → 3 324) y **S también mejora**, medido sobre 12 194 latidos.

**El límite del enfoque, ahora medido.** Las dos ideas que mejor funcionaron comparten un supuesto:
que **lo dominante en el paciente es lo normal** (el RR local para el ritmo, la plantilla para la
forma). En el registro 865 —134 lpm, con **el 58 % de sus latidos supraventriculares**— las dos se
dan vuelta y la Se de S es **0.00**. Es una limitación de diseño, no un error: el sistema supone
ritmo de base sinusal, y eso va al uso previsto y al análisis de riesgo.

**Estado de las bases externas:** DS2, INCART y SVDB están las tres gastadas. Cualquier iteración
futura se decide en DS1 y necesita otra base para confirmarse.

### Versión 5: contexto de la secuencia, para los latidos S en racha

Con la v4 confirmada, el 57 % de los errores que quedaban eran latidos S no detectados, y el
análisis los separó en dos casos muy distintos
([`notebooks/10_sequence_context.ipynb`](notebooks/10_sequence_context.ipynb)):

| En SVDB | Latidos | Se (v4) | RR previo / RR local |
|---|---|---|---|
| S aislados | 8 795 | 0.365 | 0.78 (se ven prematuros) |
| S en racha | 3 399 | **0.148** | **0.99** (no se ven prematuros) |

Dentro de una racha supraventricular el "RR local" ya está formado por latidos rápidos, así que el
latido prematuro deja de parecerlo: **la referencia está contaminada por lo que quiere detectar.**
La v5 agrega una **referencia de dos escalas** (mediana móvil de 20 latidos y otra de 200), un
**detector de racha** y el **parecido con los latidos vecinos**.

| | v4 | **v5** |
|---|---|---|
| F1 macro (3 clases) | 0.725 | **0.743** |
| Se de S | 0.371 | **0.439** |
| Se de S aislados | 0.487 | **0.568** |
| Se de S en racha | 0.279 | **0.337** |
| V no detectados (RISK-01) | 166 | **139** |
| Errores | 2 355 | **2 265** |

Mejora S **y** baja el error más grave. Una variante de solo 21 features detectaba aún más S
(Se 0.480), pero **quedó descartada por un guardarraíl declarado de antemano**: triplicaba los
latidos ventriculares no detectados (166 → 430).

**Es el mismo patrón por tercera vez:** v2 normalizó el ritmo por paciente, v4 la morfología y v5
la referencia temporal. Las tres mejoran por elegir mejor **contra qué se compara**, no por cambiar
de modelo.

⚠️ **v5 está validada solo en DS1.** DS2, INCART y SVDB ya se gastaron, y European ST-T no sirve
para confirmar S (28 latidos S contra 35 671 N en 5 registros). Falta una base adecuada.

## Estado

| Fase | Contenido | Estado |
|---|---|---|
| 1 | Descarga y EDA — [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) | ✅ |
| 2 | Filtro, segmentación y features RR — [`src/ecg/segment.py`](src/ecg/segment.py) | ✅ |
| 3 | Baseline (Random Forest / XGBoost) — [`notebooks/02_baseline.ipynb`](notebooks/02_baseline.ipynb) | ✅ |
| 4 | CNN 1D (PyTorch) — [`notebooks/03_cnn.ipynb`](notebooks/03_cnn.ipynb) | ✅ |
| 5 | Evaluación final en DS2 — [`notebooks/04_ds2_evaluation.ipynb`](notebooks/04_ds2_evaluation.ipynb) | ✅ |
| 6 | API FastAPI + Docker + demo Streamlit | ✅ |
| 7 | Documentación estilo IEC 62304 / ISO 14971 | ⏳ |
| 8 | README final y publicación | ⏳ |
| + | Versión 2 del modelo y validación externa con INCART | ✅ |
| + | Versión 3: 3 clases y estudio de calibración | ✅ |
| + | Versión 4: morfología relativa al paciente | ✅ |
| + | Confirmación externa de la v4 en SVDB | ✅ |
| + | Versión 5: contexto de secuencia (S en racha) | ✅ |

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

Estas no son las métricas finales (ver DS2 más arriba): así se eligió el modelo. La exactitud
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

## API y demo

```bash
python -m ecg.segment && python -m ecg.train baseline-v5   # datos y modelo (una vez)
uvicorn api.main:app --reload                              # API en http://localhost:8000/docs
streamlit run app/streamlit_app.py                         # demo interactiva
```

Con Docker (el modelo se entrena antes, porque `models/` no está en el repositorio):

```bash
docker build -t ecg-api . && docker run -p 8000:8000 ecg-api
```

### `POST /predict`

```json
{"fs": 360, "signal": [0.12, 0.15, "..."], "r_peaks": [370, 662, "..."]}
```

```json
{
  "model_version": "5.0.0",
  "model_name": "baseline_v5",
  "classes": ["N", "S", "V"],
  "out_of_scope": ["F"],
  "n_beats": 35,
  "beats": [{"r_peak": 662, "time_s": 1.8389, "class": "N",
             "probabilities": {"N": 1.0, "S": 0.0, "V": 0.0}}],
  "warnings": ["2 latidos quedaron sin clasificar por estar en los bordes de la señal"],
  "disclaimer": "Proyecto educativo y de investigación. No es un dispositivo médico; requiere revisión humana de un profesional."
}
```

Si no se envían `r_peaks`, se detectan con `xqrs_detect`. Si `fs` no es 360 Hz, la señal se
remuestrea y se avisa; las posiciones devueltas siempre están en el espacio de la señal enviada.

### Las validaciones son controles de riesgo, no cortesía

| Control | Qué hace |
|---|---|
| Frecuencia de muestreo fuera de 100–2000 Hz | Rechaza (REQ-002) |
| Señal menor a 10 s, vacía o con valores no numéricos | Rechaza |
| Picos R desordenados, repetidos o fuera de la señal | Rechaza |
| Menos de 3 latidos | Rechaza: cada latido necesita vecinos para las features de ritmo |
| Menos de 20 latidos | Avisa: la plantilla del paciente es poco confiable |
| Calidad de señal baja | Avisa (REQ-005) |
| **Ritmo irregular** (>15 % de los RR se apartan >30 % de la mediana) | Avisa |
| **Frecuencia fuera de 54–109 lpm** (el rango de entrenamiento) | Avisa |

Los dos últimos existen por una razón concreta: en el registro 232 el modelo clasifica **todos**
los latidos como normales cuando el 73 % son supraventriculares. Un aviso basado en las
predicciones no lo detectaría —el modelo "no ve" nada raro—, así que ambos se calculan **desde la
señal**, con umbrales calibrados contra los casos de falla conocidos (registros 232 y 865) y
registros de ritmo regular.

### Demo

Elegís un registro de DS2, un modelo (v1 a v5) y una ventana de tiempo, y muestra la **anotación
del cardiólogo y la predicción lado a lado**, con métricas y matriz de confusión del registro
completo. Sugerencias incluidas: el registro 232 para ver el fallo que no se resolvió, el 111 o el
214 para el bloqueo de rama que arregló la v4, y el 105 para ruido.

## Licencia

El código está bajo licencia [MIT](LICENSE). Los datos de MIT-BIH **no** se incluyen en el
repositorio y tienen su propia licencia (ODC-By, ver abajo).

## Datos y citas

MIT-BIH Arrhythmia Database, [PhysioNet](https://physionet.org/content/mitdb/) (ODC-By).

- Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database.* IEEE Eng Med Biol. 2001.
- Goldberger AL et al. *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation. 2000.
- de Chazal P, O'Dwyer M, Reilly RB. *Automatic classification of heartbeats using ECG
  morphology and heartbeat interval features.* IEEE Trans Biomed Eng. 2004.
