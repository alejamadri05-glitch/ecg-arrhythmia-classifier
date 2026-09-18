# Análisis de riesgo

Inspirado en ISO 14971, con fines de aprendizaje. Los riesgos se evalúan **bajo el uso previsto**
descrito en [`intended_use.md`](intended_use.md): herramienta de investigación, en diferido, con
revisión humana obligatoria. Esa revisión es el control que sostiene a todos los demás.

La severidad se refiere al efecto si el resultado se tomara como válido sin revisar.

## Riesgos identificados

| ID | Peligro / falla | Efecto potencial | Severidad | Controles | Riesgo residual |
|---|---|---|---|---|---|
| **RISK-01** | Latido V clasificado como N (falso negativo) | Arritmia ventricular no detectada | Alta | Pesos de clase balanceados; umbral de sensibilidad de V (REQ-003, ≥ 0.90; medido 0.959 en DS1); se reporta V→N en cada evaluación; revisión humana | **Medio.** Es el error que más se vigiló y el que guio las decisiones del proyecto |
| **RISK-02** | Exceso de falsos positivos | Fatiga de alarmas; se deja de confiar en el sistema | Media | Se reporta +P por clase, no solo sensibilidad; se descartó la clase F por +P 0.017; se rechazó una calibración de umbrales que bajaba errores pero perdía V | **Medio.** La +P de S sigue baja (0.31 en DS1) |
| **RISK-03** | Frecuencia de muestreo incorrecta o señal inválida | Predicciones sin sentido, presentadas con la misma apariencia de validez | Alta | Validación de entrada con rechazo 422 (REQ-002, REQ-008); remuestreo explícito y advertido (REQ-006) | **Bajo** |
| **RISK-04** | Señal ruidosa o derivación distinta de MLII | Clasificación errónea | Media | Filtro pasa-banda 0.5–40 Hz; aviso de calidad baja (REQ-005); `/model` declara la derivación de entrenamiento; uso previsto limitado | **Medio.** Medido en SVDB, con derivaciones no identificadas, la +P de V cae a 0.61 |
| **RISK-05** | Paciente distinto a los de entrenamiento | Rendimiento degradado sin aviso | Media | Evaluación inter-paciente en todas las fases; validación externa en dos bases independientes; aviso de frecuencia cardíaca fuera del rango de entrenamiento (REQ-009); limitaciones en el model card | **Medio** |
| **RISK-06** | **El ritmo dominante del paciente es la arritmia** | Falla silenciosa: el modelo llama normales a latidos que no lo son | Alta | Avisos calculados **desde la señal, no desde las predicciones** (REQ-009): ritmo irregular y frecuencia atípica; aviso adicional si más del 50 % de los latidos se clasifica como anormal | **Medio-alto.** Ver abajo: es la limitación estructural del diseño |
| **RISK-07** | Latido de fusión (F) informado como N o V | Se pierde una clase clínicamente relevante | Media | F declarado fuera de alcance de forma explícita (`out_of_scope`) en `/model` y en cada respuesta; documentado en el model card | **Medio.** Aceptado a conciencia: predecir F daba 98 % de falsas alarmas |
| **RISK-08** | El modelo servido (v5) no tiene validación externa ni evaluación en DS2 | Se sobreestima su rendimiento real | Media | El model card lo dice de forma explícita; la respuesta incluye la versión del modelo (REQ-004); DS2 y las bases externas están declaradas como gastadas | **Medio.** Se corrige solo con una base de datos nueva |
| **RISK-09** | Datos de entrenamiento corruptos o alterados en la descarga | Modelo entrenado sobre datos inválidos | Media | Verificación SHA-256 de cada archivo contra el manifiesto de PhysioNet, antes de escribir (REQ-013) | **Bajo** |
| **RISK-10** | Uso fuera del alcance previsto (clínico, tiempo real, marcapasos, pediatría) | Daño por decisión clínica basada en una herramienta no validada | Alta | Aviso en cada respuesta y en `/model` (REQ-012); uso previsto explícito; README y model card | **Medio.** Un aviso no impide el uso indebido |

## RISK-06 en detalle: el supuesto que sostiene el modelo

Las dos ideas que más mejoraron el modelo asumen lo mismo: **que lo dominante en cada paciente es
lo normal**.

- La **morfología relativa** (v4) compara cada latido con una plantilla, que es el latido mediano
  del propio paciente.
- El **contexto de secuencia** (v5) compara cada intervalo RR con la mediana móvil del paciente.

Cuando la arritmia *es* el ritmo de base, las dos se invierten: la plantilla pasa a ser un latido
anormal y el RR de referencia, uno patológico. Medido:

| Caso | Situación | Resultado |
|---|---|---|
| Registro 865 (SVDB) | 58 % de sus latidos son S, a 134 lpm | Sensibilidad de S = **0.00** |
| Registro 232 (DS2) | 75 % de los S de DS2; bradicardia | Sensibilidad de S = **0.004** |

**No está corregido.** Los avisos sirven para que la falla no pase inadvertida, no para evitarla.
Una mitigación posible, no probada, es construir la plantilla solo con latidos de ritmo regular.

## Decisiones de riesgo tomadas durante el proyecto

Dos casos en los que el análisis de riesgo cambió una decisión técnica, en contra de la métrica:

1. **Calibración de umbrales rechazada (v3).** Reducía los errores totales un 7.8 %, pero
   aumentaba los latidos V no detectados de 298 a 377, un 27 % más. RISK-01 (alta) pesa más que
   RISK-02 (media), así que se descartó y quedó detrás de una opción (`--calibrate`).
2. **Variante de 21 features rechazada (v5).** Empataba en F1 macro y mejoraba la sensibilidad de
   S, pero los V no detectados subían de 166 a 430. Se rechazó por la misma regla, declarada
   *antes* de correr el experimento.

## Riesgos de software de terceros

En [`soup.md`](soup.md).
