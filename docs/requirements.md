# Requisitos del software

Requisitos del sistema tal como está construido. Cada uno se conecta con el código que lo
implementa y con la prueba que lo verifica en [`traceability.md`](traceability.md), y con los
riesgos que controla en [`risk_analysis.md`](risk_analysis.md).

Un requisito solo se marca como verificado si existe una prueba automática que falla cuando el
requisito deja de cumplirse. Las pruebas corren en cada push (workflow `ci`).

## Funcionales

| ID | Requisito | Verificación |
|---|---|---|
| **REQ-001** | El sistema clasificará cada latido con ventana completa en **N**, **S** o **V**, y devolverá las probabilidades de cada clase. | `pytest` |
| **REQ-002** | El sistema rechazará con HTTP 422 las señales con frecuencia de muestreo fuera de 100–2000 Hz. | `pytest` |
| **REQ-003** | La sensibilidad de la clase **V** del modelo servido será **≥ 0.90** en la validación cruzada inter-paciente de DS1. | `pytest` sobre el reporte de validación |
| **REQ-004** | Cada respuesta incluirá la versión y el nombre del modelo que la produjo. | `pytest` |
| **REQ-005** | El sistema advertirá cuando la calidad de la señal sea baja. | `pytest` |
| **REQ-006** | El sistema remuestreará a 360 Hz cualquier señal con otra frecuencia, y lo advertirá. | `pytest` |
| **REQ-007** | Si no se envían los picos R, el sistema los detectará automáticamente y lo advertirá. | `pytest` |
| **REQ-008** | El sistema rechazará con HTTP 422 entradas inválidas: señal menor a 10 s, valores no numéricos, picos R desordenados o fuera de la señal, y menos de 3 latidos. | `pytest` |
| **REQ-009** | El sistema advertirá cuando el ritmo sea muy irregular o la frecuencia cardíaca mediana quede fuera del rango visto en entrenamiento. | `pytest` |
| **REQ-012** | Toda respuesta incluirá el aviso de que no es un dispositivo médico y requiere revisión humana. | `pytest` |

### Alcance de clases (desviación respecto de la especificación inicial)

La especificación original pedía cuatro clases (N, S, V, **F**). La clase F se **declaró fuera de
alcance** en la versión 3 del modelo, con esta justificación: en la evaluación única sobre DS2, su
predictividad positiva fue de **0.017**, con 3 049 falsos F. Una clase que se equivoca 98 de cada
100 veces que se activa no aporta información y sí genera fatiga de alarmas (RISK-02).

La decisión es explícita en la salida: `out_of_scope: ["F"]` en `/model` y en cada predicción. Los
latidos de fusión se informarán como N o V (RISK-07).

## De datos y entrenamiento

| ID | Requisito | Verificación |
|---|---|---|
| **REQ-010** | La evaluación será **inter-paciente**: ningún registro podrá aparecer a la vez en entrenamiento y en prueba, ni en dos particiones de la validación cruzada. Los registros con marcapasos se excluirán. | `pytest` |
| **REQ-011** | Con la misma semilla y los mismos datos, el entrenamiento y la predicción darán el mismo resultado. | `pytest` |
| **REQ-013** | Los datos descargados de PhysioNet se verificarán por SHA-256 antes de usarse, y una descarga incompleta o alterada no dejará archivos a medias. | `pytest` |

## No funcionales

| ID | Requisito | Verificación |
|---|---|---|
| **REQ-014** | El sistema se distribuirá como imagen de Docker que expone la API, y esa imagen se construirá y responderá correctamente en cada cambio relevante. | Workflow `docker` |
| **REQ-015** | El código pasará `ruff` (formato y linter) y la totalidad de las pruebas en cada push. | Workflow `ci` |

## Límites de entrada implementados

Valores en [`api/main.py`](../api/main.py); son controles de riesgo, no preferencias de estilo.

| Parámetro | Valor | Motivo |
|---|---|---|
| Frecuencia de muestreo | 100–2000 Hz | Fuera de ese rango el remuestreo deforma el latido (RISK-03) |
| Duración mínima | 10 s | Menos señal no da contexto de ritmo suficiente |
| Latidos mínimos | 3 | Cada latido necesita un vecino previo y uno siguiente |
| Muestras máximas | 20 000 000 | ~15 h a 360 Hz; evita agotar la memoria |
| Aviso por pocos latidos | < 20 | La plantilla del paciente es poco confiable |
| Aviso de ritmo irregular | > 15 % de los RR se apartan > 30 % de la mediana | Umbral calibrado contra casos de falla conocidos |
| Aviso de frecuencia atípica | RR mediano fuera de 0.55–1.12 s (54–109 lpm) | Rango de frecuencias medianas visto en DS1 |
| Aviso de calidad | índice de `neurokit2` < 0.6 | REQ-005 |
