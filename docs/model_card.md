# Model card — `baseline_v5`

> Proyecto educativo y de investigación. **No es un dispositivo médico.** Ver
> [`intended_use.md`](intended_use.md).

## Ficha

| | |
|---|---|
| Nombre | `baseline_v5` (versión 5.0.0) |
| Tipo | XGBoost sobre 58 características diseñadas a mano |
| Entrada | Ventana de 252 muestras por latido (0.25 s antes y 0.45 s después del pico R) a 360 Hz, más contexto de ritmo |
| Salida | Clase (N, S, V) y probabilidad de cada una |
| Clases fuera de alcance | **F** (fusión) |
| Entrenamiento | DS1 de de Chazal et al. (2004): 22 registros, 50 555 latidos de las 3 clases |
| Semilla | 42 |
| Archivo | `models/baseline_v5.joblib` (no versionado: se entrena con `python -m ecg.train baseline-v5`) |

Las características combinan tres familias: **ritmo** (intervalos RR normalizados por la mediana
del paciente), **morfología relativa** (cada latido comparado con la plantilla del propio paciente,
que se recalcula cada 100 latidos) y **contexto de secuencia** (mediana móvil de RR a dos escalas,
detector de rachas, correlación con el latido previo y el siguiente).

## Rendimiento

### Modelo servido (v5), validación cruzada inter-paciente en DS1

Cinco particiones por paciente, sin que ningún registro aparezca en dos particiones
(`reports/baseline_v5_cv.json`).

| Clase | Sensibilidad (Se) | Predictividad positiva (+P) | F1 |
|---|---|---|---|
| N | 0.965 | 0.987 | 0.976 |
| S | 0.439 | 0.315 | 0.367 |
| V | 0.959 | 0.821 | 0.885 |
| **F1 macro** | | | **0.743** |

Exactitud 0.955, que **no es la métrica principal**: predecir siempre N ya da 0.906 en DS1.
Latidos V clasificados como N (RISK-01): 139.

### Evaluaciones en otros conjuntos

Estas evaluaciones corresponden a **versiones anteriores** del modelo, cada una gastada en su
momento y nunca reabierta para ajustar nada.

| Conjunto | Modelo evaluado | F1 macro | Nota |
|---|---|---|---|
| DS2 (evaluación única) | v1 (`baseline_xgb`, 4 clases) | 0.513 | V Se 0.964; S Se 0.155; F +P 0.017 |
| INCART (75 pacientes, derivación II) | v2 | 0.634 | S Se 0.843 |
| SVDB (78 pacientes, derivaciones sin identificar) | v4 | 0.698 | V +P 0.607; S Se 0.305 |

### La limitación más importante de este model card

**El modelo que se sirve (v5) nunca se evaluó en DS2 ni en una base externa** (RISK-08). Su único
respaldo es la validación cruzada en DS1. Las tres bases independientes ya se gastaron:

- DS2 se usó una sola vez, con la v1, y no se reabre por regla del proyecto.
- INCART se gastó comparando v1 contra v2.
- SVDB se gastó confirmando la v4.

Se descartó European ST-T como cuarta base porque no sirve para la clase S: una sonda sobre 5
registros encontró 28 latidos S contra 35 671 N. Confirmar la v5 exige una base de datos nueva.

Como referencia del tamaño del salto entre validación interna y externa: la v4 marcó 0.725 en DS1
y 0.698 en SVDB. Es razonable esperar una caída similar, pero **esperar no es medir**.

## Limitaciones conocidas

1. **La clase S sigue siendo débil** (Se 0.44, +P 0.31). Su morfología es normal por definición: lo
   que la distingue es el timing, y el timing es ruidoso.
2. **El supuesto estructural.** El modelo asume que el ritmo y la forma dominantes de cada paciente
   son los normales. Cuando la arritmia es el ritmo de base, falla: sensibilidad de S de 0.00 en el
   registro 865 de SVDB (58 % de sus latidos son S) y 0.004 en el registro 232 de DS2. Está
   documentado en RISK-06 y la API lo advierte, pero no lo corrige.
3. **Clase F fuera de alcance.** Un latido de fusión se informará como N o V (RISK-07). Se quitó
   porque su predictividad positiva en DS2 fue 0.017, con 3 049 falsas alarmas.
4. **Retraso de un latido.** Usa el intervalo RR siguiente, así que no sirve para tiempo real.
5. **Una sola derivación, MLII.** En bases con otras derivaciones el rendimiento cae: en SVDB, con
   derivaciones sin identificar, la +P de V baja a 0.61.
6. **Datos de los años 70 y 80**, 47 sujetos, población adulta y limitada. No representa la
   diversidad de pacientes, equipos ni condiciones de registro actuales.
7. **Sin pacientes con marcapasos:** los registros 102, 104, 107 y 217 se excluyen según AAMI EC57.
8. **Frecuencias cardíacas extremas.** El rango de frecuencias medianas de DS1 va de 54 a 109 lpm;
   fuera de ahí la API advierte y el rendimiento en S cae.

## Cómo se eligió

Regla declarada **antes** de mirar los datos de prueba: mayor F1 macro out-of-fold en DS1, con las
particiones congeladas en `config.DS1_FOLD_MAP` para que todos los experimentos sean comparables.

Dos veces se rechazó una variante que ganaba en la métrica porque empeoraba el error más grave
(latidos V no detectados, RISK-01): la calibración de umbrales de la v3 y la variante de 21
características de la v5. El criterio se fijó antes de correr cada experimento.

La CNN 1D quedó descartada: 0.463 en DS1 contra 0.493 del baseline, y en INCART 0.464 contra
0.634, con 14 614 falsos F.

## Usos no recomendados

Cualquier uso clínico, monitoreo en tiempo real, pacientes con marcapasos, pediatría, detección de
ritmos como la fibrilación auricular, y cualquier decisión tomada sin revisión de un profesional.

## Mantenimiento

Reentrenar y reevaluar si cambia el conjunto de entrenamiento, el conjunto de características o
alguna dependencia de [`soup.md`](soup.md). La sensibilidad de V se controla automáticamente contra
el umbral de REQ-003 en cada corrida de las pruebas.
