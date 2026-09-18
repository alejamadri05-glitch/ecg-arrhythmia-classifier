# Uso previsto

> **Aviso.** Proyecto educativo y de investigación. **No es un dispositivo médico**, no tiene
> marcado CE ni autorización de la FDA, y no debe usarse para decisiones clínicas. Esta carpeta
> está *inspirada en* IEC 62304 e ISO 14971 con fines de aprendizaje: no es una declaración de
> cumplimiento.

## Qué hace

Clasifica cada latido de un registro de ECG en tres clases AAMI: **N** (normal y similares),
**S** (ectópico supraventricular) y **V** (ectópico ventricular). Devuelve, por latido, la clase
y las probabilidades de cada una, junto con advertencias sobre la señal.

Es una **herramienta de análisis retrospectivo**, pensada para revisar registros ya grabados, del
estilo de un Holter. Toda salida requiere revisión de un profesional: el sistema no emite
diagnósticos ni decide por sí solo.

## Usuario previsto

Una persona con formación técnica o clínica que analiza registros de ECG con fines de
investigación o docencia, y que interpreta los resultados con criterio propio.

## Condiciones de uso

| Aspecto | Especificación |
|---|---|
| Señal | Una derivación, preferentemente **MLII** |
| Frecuencia de muestreo | 100–2000 Hz; se remuestrea internamente a 360 Hz |
| Duración mínima | 10 s y al menos 3 latidos |
| Población | Adultos, ritmo de base mayoritariamente regular |
| Modo | Diferido (por lotes), sobre registros ya grabados |

## Fuera de alcance

Estas exclusiones no son formalidades: cada una corresponde a una limitación medida, documentada
en [`model_card.md`](model_card.md) y en [`risk_analysis.md`](risk_analysis.md).

- **Uso clínico o diagnóstico**, y cualquier decisión de tratamiento.
- **Monitoreo en tiempo real.** El sistema usa el intervalo RR *siguiente* a cada latido, así que
  clasificar un latido exige esperar al próximo: hay un retraso de un latido por construcción.
- **Pacientes con marcapasos.** Los registros con marcapasos (102, 104, 107, 217) se excluyen
  siguiendo AAMI EC57, así que el modelo nunca los vio.
- **Pediatría.** MIT-BIH es una población adulta.
- **Latidos de fusión (clase F).** Se declararon fuera de alcance en la v3 y el modelo no los
  predice: un latido F se informará como N o como V. La respuesta de la API lo declara
  explícitamente en el campo `out_of_scope`. El motivo está en [`model_card.md`](model_card.md).
- **Pacientes cuyo ritmo dominante es la arritmia.** El modelo asume que lo mayoritario en cada
  paciente es lo normal, tanto para el ritmo como para la forma del latido. Cuando ese supuesto se
  cae, el rendimiento se desploma (medido: sensibilidad de S igual a 0.00 en un caso). El sistema
  lo advierte, pero no lo corrige.
- **Derivaciones distintas de MLII.** Funciona, con peor rendimiento; la API declara en `/model`
  la derivación con la que se entrenó.
- **Detección de ritmos**, como fibrilación auricular. El sistema clasifica latidos, no ritmos.

## Beneficio clínico

Ninguno declarado. Es un proyecto de portafolio con fines de aprendizaje.
