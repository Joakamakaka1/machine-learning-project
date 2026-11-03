# Segunda revisión: datos y validación

Revisión independiente del dataset y del proceso de validación, hecha en la rama `review/data-validation-audit`. El [protocolo de la revisión](revision_protocolo.md) se fijó antes de experimentar. `models/champion/` y `outputs/final_evaluation/` no cambian.

## Veredicto

**El proceso de ingeniería es sólido y el modelo no se puede mejorar de forma demostrable. Lo que falla es lo que se afirma sobre él: la cifra de 0.82 no mide lo que dice medir.**

| Pregunta | Respuesta |
|---|---|
| ¿Se reproduce lo publicado? | Sí, exactamente |
| ¿Hay fugas en el preprocesado o en la partición? | No de importancia. El ajuste está dentro del pipeline y la partición temporal elimina los duplicados |
| ¿Se conocen las 19 variables al confirmar? | **No está demostrado, y la fuente lo contradice en parte** |
| ¿Mide 0.82 el rendimiento al confirmar? | **No del todo.** Con las reservas confirmadas tras el despliegue da 0.79 en 2017 y 0.75 en 2016 |
| ¿Sirve para las cancelaciones que más cuestan? | **Casi no.** Frente a las estancias: 0.55 en las canceladas a 7 días o menos de llegar y 0.52 en los `No-Show` |
| ¿Hay un modelo mejor? | No. **0 de 50 comparaciones** cumplen la regla de aceptación |
| ¿Hay techo? | Sí: unos 0.83 en total y 0.75 con `No Deposit`, con estas 19 variables |

Tu sospecha se confirma a medias. El trabajo fue riguroso en la mecánica (partición, pipeline, ejecución única) y flojo en un punto concreto: **no se contrastó el momento de captura de las variables con el artículo que describe el dataset.**

## Lo que encontré en la documentación

**Objetivo y problema:** clasificación binaria de `is_canceled`, declarada como predicción al confirmar la reserva. La métrica principal es el ROC-AUC. Es la práctica de un máster, también pensada como portfolio. Todo es local.

**Qué se hizo y en qué orden.** Una auditoría previa detectó la inflación de 0.952 con partición aleatoria y fijó las 19 variables con las ventanas de 2016. Después se escribió el protocolo, se seleccionó el modelo y se ejecutó el test una vez el 2026-11-02. El historial de git es coherente con ese orden.

**Lo que ya estaba declarado y es correcto:** que 2017 no era virgen, el origen de `ADR_UPPER_LIMIT`, `min_frequency` y las rejillas, el empate con XGBoost, y que los costes son supuestos.

**Limitación de partida:** el test ya está gastado y no hay más datos. Tu paso de "evaluación final única" no se puede cumplir con un modelo nuevo de forma ciega. Como ningún cambio se aceptó, tampoco hizo falta.

## Línea base reproducida

Con `experiments/review/baseline.py`, sin tocar nada congelado.

| Comprobación | Resultado |
|---|---|
| Hash del campeón frente al manifest | Coincide |
| ROC-AUC, PR-AUC y Brier del test, desde las predicciones guardadas | Coinciden: 0.8175, 0.7608 y 0.1627 |
| ROC-AUC de las 4 ventanas de 2016, reentrenando | Diferencia máxima de 0.0000: 0.8198, 0.8346, 0.7840 y 0.8307 |
| Modelo servido, reentrenado en memoria | Probabilidades idénticas en las 78 054 filas de desarrollo |

## Problemas detectados

### Impacto alto

**1. Disponibilidad al confirmar: no demostrada.**

- **Evidencia:** Antonio et al. (2019, *Data in Brief*) fijan el punto temporal de cada observación en el día anterior a la llegada: los valores se reconstruyen "with a timestamp relative to the day prior to arrival date". Las variables de la reserva salen de esa tabla y de su registro de cambios. `deposit_type` se calcula con los pagos anteriores a la llegada o a la cancelación, y `adr`, con las transacciones de alojamiento.
- **Qué contradice:** la data card dice que todas se conocen al confirmar, que `deposit_type` se conoce al reservar y que ese es el único momento defendible. Ninguna de las tres cosas se puede sostener con la fuente.
- **Impacto:** las únicas variables definidas en la creación o la confirmación dan 0.6866, frente a 0.8207 de las 19 (`M12`, XGBoost de referencia). **El rendimiento real al confirmar está en algún punto de ese rango, y este archivo no permite saber dónde.**
- **Hipótesis, no demostrada:** `Non Refund` cancela el 99.2 %, y puede que parte de esa etiqueta se fije al procesar la cancelación. No encontré ninguna fuente accesible que lo explique. Sin `deposit_type`, el ROC-AUC baja 0.0136 y con `No Deposit` baja 0.0058.
- **Estado:** no se puede resolver sin el registro de cambios. Hay que declararlo.
- **A favor del proyecto:** esto refuerza la exclusión de `total_of_special_requests` y `booking_changes`. Si los valores se leen tan cerca de la llegada, es razonable que quien cancela deje de acumular antes que quien llega, y eso es lo que el diagnóstico de acumulación vio de forma observacional. El artículo no lo afirma: es mi lectura.

**2. La cifra global mezcla reservas que el modelo no vería al confirmar.**

- **Evidencia:** la partición corta por llegada, de modo que entre el 31 % y el 69 % de las filas de cada ventana de 2016 se confirmaron **antes** del corte. El entrenamiento incluye reservas posteriores a ellas. Un modelo desplegado en una fecha solo predice las confirmadas después.
- **Impacto en 2016:** XGBoost de referencia, 0.8141 con todas las filas y 0.7591 con las confirmadas tras el corte. Random Forest sobre cohortes de reservas confirmadas tras cada origen: **0.7499** de media (de 0.7136 a 0.7773), y 0.7176 con `No Deposit`.
- **Entrenando solo con lo conocido en la fecha de cada reserva:** en las mismas filas de 2016, el XGBoost de referencia baja de 0.7963 a 0.7645. Esta cifra mezcla dos efectos que no se pueden separar: la fuga hacia atrás y tener un modelo más antiguo y con menos filas.
- **Impacto en 2017, análisis posterior:** **0.7917** en las confirmadas desde 2017-01-01, frente a 0.8175 con todas. Con `No Deposit`, 0.7540.
- **Matiz que modera mi alarma inicial:** la caída real en 2017 es de 0.026, menor que la de 2016. Con `No Deposit` no hay caída, porque la diferencia viene sobre todo de la composición: las `Non Refund` son el 4.9 % de esas filas y el 10.2 % del total. Además, los plazos largos quedan recortados por el final de los datos.
- **Estado:** se puede corregir en la comunicación. **La cifra más defendible es 0.79, y 0.75 con `No Deposit`.**

**3. La métrica no refleja el coste que justifica el proyecto.**

- **Evidencia:** el README justifica el problema por la cancelación tardía, pero el modelo apenas la ve. ROC-AUC de cada tipo frente a las estancias:

| Tipo de cancelación | 2016, validación | 2017, análisis posterior |
|---|---:|---:|
| `No-Show` | 0.527 | 0.517 |
| Hasta 7 días antes de llegar | 0.591 | 0.554 |
| De 8 a 30 días antes | 0.762 | 0.762 |
| 31 días o más antes | 0.899 | 0.874 |
| `No Deposit`, solas | 0.633 | 0.662 |
| `No Deposit`, en sucesos de 10 cancelaciones o más | 0.805 | 0.834 |

- **Lectura:** el modelo detecta mejor las tandas y las reservas con mucha antelación. Parte del gradiente es estructural, porque con poca antelación no se puede cancelar 31 días antes. Los grupos usan el resultado para definirse, así que solo sirven para diagnosticar.
- **Estado:** no se resuelve con estas variables. Haría falta predecir más cerca de la llegada, con el historial de la reserva.

### Impacto medio

**4. Tandas de reservas: menos información de la que parece.**

- **Evidencia:** el 53 % de las filas de desarrollo está en tandas de 10 o más, con correlación intraclase de 0.47, efecto de diseño de 26 y unas 3 000 filas efectivas. Es una cota sin condicionar a las variables. Hay días de hasta 1 112 cancelaciones de un solo agente, y el 73 % de las cancelaciones cae en sucesos de 10 filas o más.
- **Fuga entre ventanas:** existe, pero es pequeña. Con Random Forest, purgar da 0.8111, y quitar tandas al azar del mismo tamaño, 0.8147: la fuga neta es de **0.0036**, casi toda en 2016T2.
- **Intervalo publicado:** se queda algo estrecho. Con los mismos datos de 2017:

| Bloques | IC 95 % | Anchura |
|---|---|---:|
| Hotel y día de llegada, como el publicado | [0.8054, 0.8292] | 0.0238 |
| Tanda de reserva | [0.8041, 0.8311] | 0.0270 |
| Semana de llegada | [0.8016, 0.8318] | 0.0302 |
| Hotel, día de estado y agente | [0.8010, 0.8336] | 0.0326 |
| Agente y mes de reserva | [0.7839, 0.8479] | 0.0640 |

- **Estado:** se puede corregir en la comunicación. Yo daría [0.801, 0.834], y diría que depende de pocos agentes.

**5. El modelo no se reentrena, y eso cuesta.**

- **Evidencia:** los mismos meses de 2016, con Random Forest, según cuándo acaba el entrenamiento. Los meses medios desde el último dato son 0, 1, 2.5 y 5.5.

| Frecuencia | ROC-AUC | `No Deposit` | Frente a trimestral, IC 95 % |
|---|---:|---:|---|
| Mensual | 0.8242 | 0.7412 | +0.0083 [+0.0011, +0.0155] |
| Trimestral | 0.8159 | 0.7286 | |
| Semestral | 0.8062 | 0.7125 | -0.0098 [-0.0190, -0.0006] |
| Congelado desde enero | 0.7911 | 0.6918 | -0.0249 [-0.0377, -0.0120] |

- **Salvedad:** el congelado solo tiene 21 820 filas de entrenamiento, de modo que esta comparación también mezcla el tamaño. Con Holm sobre las tres comparaciones, la mensual da p = 0.0559, así que no cumple la regla por poco.
- **Dato nuevo a favor del campeón:** Random Forest envejece mejor que XGBoost. En 2016T4 con 9 meses sin datos da 0.7833, frente a 0.7324.
- **Estado:** es la mejora real más probable, pero es operativa y no se ha podido medir en 2017.

**6. Umbral y calibración: se eligieron con una sola pareja.**

- **Evidencia:** con el umbral 0.25, las tres parejas de 2016 avisan del 46 % al 63 % de las reservas, y al remuestrear el calibrador, del 45 % al 55 %. **En 2017 avisa del 77.3 %**, con un rango mensual del 65 % al 83 %.
- **Qué corrige:** el informe lo atribuye a los costes supuestos. Es incompleto: con esos mismos costes, el rango de 2016 no lo anticipaba.
- **Isotónica frente a sigmoide:** la isotónica gana por 0.001 en una pareja de tres, y la sigmoide gana por menos de 0.001 en las otras dos (Brier de 0.1541 y 0.1687 frente a 0.1550 y 0.1692). El criterio se aplicó solo en la pareja donde gana.
- **Estado:** no cambia el modelo congelado. En una versión futura, el umbral debería fijarse por la capacidad de revisión y vigilarse.

**7. Cobertura desigual de City Hotel.**

- **Evidencia:** en julio de 2015 tiene 33 habitaciones ocupadas de media según los datos, frente a 212 en julio de 2016, y cancela el 67 %. En noviembre de 2015, 91 frente a 175. Resort Hotel no tiene ese patrón, de modo que parece que faltan estancias, no cancelaciones. No sé si es falta de registro o menos actividad del hotel, y el artículo no lo aclara.
- **Tratamiento probado:** quitar esas filas no mejora: -0.0007 [-0.0047, +0.0032].
- **Estado:** se conservan y se declara.

**8. `previous_cancellations` no es estable.**

- **Evidencia:** fuera de `Non Refund` y de las tandas, su tasa de cancelación cae del 97.5 % en 2015T3 al 27.1 % en 2016T4.
- **Tratamiento probado:** quitarla o binarizarla no cambia nada significativo: -0.0040 [-0.0095, +0.0014] y +0.0002 [-0.0009, +0.0013].
- **Estado:** se conserva, y se vigila junto con `agent`.

### Impacto bajo

| Problema | Evidencia | Estado |
|---|---|---|
| "Test ya en train: 0.0 %" es trivial en el escalón temporal | Su clave incluye la llegada. Con solo las 19 variables es del 0.7 % al 2.5 % en 2016. El docstring de `expanding_windows` tampoco es exacto | La conclusión se mantiene: la partición temporal sí elimina los duplicados |
| "Filas únicas" usa el objetivo en la clave | Sin usarlo, 2017 da 0.7667 en lugar de 0.7608, y 2016, 0.0065 más de media | Definición nueva en esta revisión |
| Fechas y noches de fin de semana no cuadran | `stays_in_weekend_nights` solo cuadra (100 % de las filas) si el fin de semana son las noches de domingo y lunes de las fechas del CSV. Parecen desplazadas | Sin efecto, porque no se usa el calendario. Cuidado al crear variables de día de la semana |
| `No-Show` cuenta como cancelación | Son el 3.0 % de los positivos. El artículo no lo dice de forma expresa: se ve en los datos | Se declara |
| Nulos y valores extremos | `children`: 4 nulos. Grupos de hasta 55 adultos. Quitarlos: +0.0006 [-0.0013, +0.0024] | Sin tratamiento |

### Lo que busqué y no encontré

- **Fuga en el preprocesado:** imputación, escalado y codificación se ajustan dentro del pipeline en cada ventana.
- **Incoherencias en las etiquetas:** `is_canceled` coincide siempre con el estado, ninguna cancelación es posterior a la llegada ni anterior a la reserva, y el 99.96 % de las estancias tiene fecha de estado igual a la salida.
- **Huella de la etiqueta en `adr`:** sus decimales apenas distinguen, con un ROC-AUC de 0.48 a 0.54 con `No Deposit`.
- **Otra variable delatora:** ninguna de entrada supera por sí sola a `lead_time` (0.72 en 2016T4).
- **Optimismo de selección:** la búsqueda anidada baja solo 0.0007 en Random Forest y 0.0018 en XGBoost.
- **Azar:** la desviación entre 5 semillas es de 0.0006 en Random Forest, y de 0 en XGBoost, porque su configuración no submuestrea.
- **Deriva anómala de 2017:** un clasificador de periodos da 0.81 frente a 2017, y 0.83 en el control dentro de desarrollo. 2017 no es más distinto que 2016T3 y T4 respecto a lo anterior.

## Estrategia de validación

- **Se mantiene el origen móvil por llegada:** comparado en desarrollo con XGBoost, la partición aleatoria da 0.8877, la agrupada por tandas 0.8575 y la temporal 0.8141. De esos 0.074, unos 0.030 son tandas y unos 0.043 son tiempo.
- **Se añaden 12 ventanas mensuales para decidir,** y las 4 trimestrales solo para comparar.
- **Lo que le falta al esquema original:** medir aparte las cohortes confirmadas tras el corte y la caída con el horizonte.

## Registro de experimentos

Cada uno cambia una sola cosa. Las diferencias son pareadas frente a su referencia, en `M12`, con intervalo t del 95 %. Las tablas completas, por ventana y con el motivo de cada veredicto, están en [`outputs/review/`](../outputs/review/), y el resumen en [`decision_log.csv`](../outputs/review/decision_log.csv).

**Resultado: 0 de 50 cumplen la regla.**

### Escalera de modelos

Media y desviación entre ventanas `M12`. Comprueba de lo simple a lo complejo que cada paso se justifica, y que el campeón no tiene rival.

| Modelo | ROC-AUC | `No Deposit` | Distintas | Log-loss |
|---|---:|---:|---:|---:|
| Prevalencia | 0.5000 ± 0.0000 | 0.5000 | 0.5000 | 0.6507 |
| Regla: `deposit_type` | 0.6629 ± 0.0534 | 0.5000 | 0.5188 | 0.5236 |
| Regla: depósito y antelación | 0.7254 ± 0.0565 | 0.5924 | 0.6263 | 0.5112 |
| Logística, 5 variables | 0.7628 ± 0.0461 | 0.6480 | 0.6874 | 0.5004 |
| Logística, 19 variables | 0.7982 ± 0.0325 | 0.7013 | 0.7290 | 0.4939 |
| Logística con splines | 0.8018 ± 0.0313 | 0.7069 | 0.7377 | 0.4865 |
| Árbol | 0.8051 ± 0.0289 | 0.7104 | 0.7319 | 0.4895 |
| **Random Forest, configuración del campeón** | **0.8242 ± 0.0325** | **0.7412** | **0.7576** | **0.4509** |
| XGBoost | 0.8236 ± 0.0346 | 0.7391 | 0.7555 | 0.4536 |
| `HistGradientBoosting` nativo | 0.8192 ± 0.0381 | 0.7324 | 0.7521 | 0.4668 |

XGBoost frente a Random Forest: -0.0006 [-0.0067, +0.0055]. **El empate se confirma con 12 ventanas.** Una regla de dos variables ya da 0.7254: el dummy era un baseline demasiado débil para juzgar el 0.82. La red de Keras no se ha repetido, por coste y porque el catálogo ya la medía por debajo (0.8053 en `Q4`).

### Tratamientos de datos

Cambian solo el entrenamiento. XGBoost de referencia.

| Cambio y motivo | ROC-AUC [IC 95 %] | `No Deposit` [IC 95 %] | `Q4` |
|---|---|---|---:|
| Sin City Hotel antes de 2015-09: parecen faltar estancias | -0.0007 [-0.0047, +0.0032] | -0.0012 [-0.0075, +0.0050] | -0.0024 |
| Sin los meses de City Hotel con poca cobertura | -0.0082 [-0.0221, +0.0057] | -0.0130 [-0.0350, +0.0090] | -0.0074 |
| Solo los últimos 12 meses: deriva | +0.0005 [-0.0007, +0.0017] | +0.0008 [-0.0009, +0.0025] | +0.0004 |
| Solo los últimos 9 meses | +0.0020 [-0.0013, +0.0053] | +0.0031 [-0.0014, +0.0076] | +0.0005 |
| Pesos por antigüedad, vida media de 180 días | +0.0029 [-0.0005, +0.0062] | +0.0042 [-0.0009, +0.0092] | +0.0012 |
| Pesos por antigüedad, vida media de 365 días | +0.0017 [+0.0001, +0.0034] | +0.0026 [+0.0002, +0.0050] | +0.0016 |
| Peso inverso al tamaño de la tanda | -0.0071 [-0.0171, +0.0028] | -0.0097 [-0.0248, +0.0055] | -0.0066 |
| Peso inverso a su raíz | -0.0002 [-0.0053, +0.0050] | -0.0002 [-0.0084, +0.0080] | +0.0009 |
| Sin tandas de 10 o más | -0.0121 [-0.0191, -0.0052] | -0.0167 [-0.0273, -0.0062] | -0.0134 |
| Sin `Non Refund` | -0.1157 [-0.1622, -0.0691] | +0.0005 [-0.0030, +0.0039] | -0.1190 |
| Sin cancelaciones previas de 2015 | -0.0008 [-0.0055, +0.0039] | -0.0011 [-0.0084, +0.0062] | -0.0044 |
| Sin valores extremos | +0.0006 [-0.0013, +0.0024] | +0.0006 [-0.0022, +0.0035] | -0.0006 |

**Lo más prometedor, los pesos por antigüedad, desaparece con Holm** (p = 0.41 con 365 días) **y con Random Forest:** +0.0003 [-0.0012, +0.0019]. Quitar `Non Refund` del entrenamiento no cambia `No Deposit`: esa cifra no depende de esas filas.

### Variables por grupo de riesgo

Los grupos siguen la tabla 1 del artículo.

| Variables | ROC-AUC | Diferencia [IC 95 %] | `No Deposit` |
|---|---:|---|---:|
| Las 19 | 0.8207 | | 0.7352 |
| Sin `deposit_type` | 0.8072 | -0.0136 [-0.0194, -0.0077] | 0.7294 |
| Sin depósito, agente, segmento, canal ni tipo de cliente | 0.7247 | -0.0960 [-0.1228, -0.0693] | 0.6600 |
| Solo las definidas al crear o confirmar | 0.6866 | -0.1341 [-0.1659, -0.1023] | 0.6175 |
| Las mismas, sin el historial del cliente | 0.6754 | -0.1454 [-0.1679, -0.1228] | 0.6029 |

**Quitando una a una,** solo pierden con claridad `lead_time` (-0.0173), `agent` (-0.0154), `deposit_type` (-0.0136) y `customer_type` (-0.0081). Las otras 15, con Holm, no se distinguen de cero por sí solas. Pero un Random Forest con solo las 8 mejores no iguala a las 19: -0.0047 [-0.0106, +0.0012]; y con 5 pierde: -0.0128 [-0.0234, -0.0022]. Hay que conservar las 19.

### Variables nuevas

Solo con el pasado: una reserva cuenta como historia cuando su llegada y su último estado son anteriores a la fecha de reserva de la fila. Un test lo comprueba.

| Cambio y motivo | ROC-AUC [IC 95 %] | `No Deposit` [IC 95 %] | `Q4` |
|---|---|---|---:|
| Tasa histórica del agente: `agent` deriva mucho | +0.0020 [-0.0023, +0.0063] | +0.0030 [-0.0030, +0.0090] | -0.0066 |
| Tasa del agente en los últimos 180 días | +0.0028 [-0.0020, +0.0076] | +0.0042 [-0.0028, +0.0111] | -0.0075 |
| Tasa histórica en lugar de `agent` | -0.0045 [-0.0121, +0.0031] | -0.0072 [-0.0186, +0.0042] | -0.0155 |
| Tasa del hotel y el segmento | +0.0029 [-0.0027, +0.0084] | +0.0041 [-0.0044, +0.0127] | +0.0005 |
| Las dos tasas | +0.0009 [-0.0052, +0.0069] | +0.0010 [-0.0081, +0.0101] | -0.0059 |
| Noches, huéspedes, tarifa por huésped, tiene agente | -0.0005 [-0.0027, +0.0016] | -0.0007 [-0.0042, +0.0028] | -0.0013 |
| Mes de llegada, una categoría por mes | -0.0062 [-0.0107, -0.0017] | -0.0084 [-0.0149, -0.0019] | -0.0068 |
| Estación, seno y coseno | -0.0008 [-0.0068, +0.0052] | -0.0010 [-0.0105, +0.0084] | -0.0060 |
| Cancelación previa como binaria | +0.0002 [-0.0009, +0.0013] | +0.0004 [-0.0012, +0.0020] | -0.0008 |
| `min_frequency` de 200 | -0.0035 [-0.0106, +0.0037] | -0.0051 [-0.0166, +0.0064] | -0.0054 |
| `min_frequency` de 10 | -0.0006 [-0.0020, +0.0008] | -0.0009 [-0.0031, +0.0013] | -0.0007 |

Esto **descarta una mejora que proponía el informe original:** sustituir los identificadores de agente por su tasa histórica no ayuda en estos datos. También confirma, con solo el mes, la exclusión del calendario.

### Robustez y techo

- **Curva de aprendizaje:** de 40 % a 100 % de las filas, el ROC-AUC sube de 0.8192 a 0.8314 en 2016T2, de 0.7656 a 0.7690 en T3 y de 0.8226 a 0.8275 en T4. Más filas de este tipo ayudarían poco.
- **Techo temporal:** dando al XGBoost ajustado filas del mismo trimestre, en grupos por tanda, se llega a 0.8325 y 0.7507 con `No Deposit`. Es una cifra inalcanzable en producción, y está a solo +0.0163 de lo temporal. En 2016T4 la distancia es casi nula, y en 2016T1 es de +0.0349.
- **Ensamble:** Random Forest y XGBoost juntos dan 0.8183 frente a 0.8172. Los modelos han convergido.
- **Perturbaciones:** ruido del 10 % en `adr`, -0.0011; ±7 días en `lead_time`, -0.0025; todos los agentes desconocidos, -0.0203.
- **Hiperparámetros:** las 20 configuraciones de Random Forest van de 0.8081 a 0.8172 (desviación de 0.0029), mucho menos que entre ventanas (0.0231).

## Resultado final frente a la línea base

**El modelo no cambia, y su cifra publicada tampoco:** el test no se ha repetido. Lo que cambia es cómo leerla.

| Medida del campeón en 2017 | Filas | ROC-AUC | Lectura |
|---|---:|---:|---|
| **Publicada** | 40 509 | **0.8175**, IC [0.8058, 0.8297] | Correcta como cálculo, optimista como afirmación |
| Publicada, por subconjunto | | Únicas 0.7608, `No Deposit` 0.7529, `Non Refund` 0.9998 | Sin cambios |
| Con IC por suceso de cancelación | 40 509 | 0.8175, IC [0.8010, 0.8336] | Análisis posterior |
| Confirmadas desde 2017-01-01 | 26 409 | **0.7917** | Análisis posterior. Plazo máximo de 241 días |
| Lo mismo, con `No Deposit` | | **0.7540** | Análisis posterior |
| Filas distintas, sin usar el objetivo | | 0.7667 | Análisis posterior |
| Solo variables definidas al confirmar | | Sin medir en 2017 | En 2016, 0.6866 |

Las cifras de análisis posterior **no son una evaluación nueva:** no se puntuó ninguna fila ni salió ninguna decisión de ellas.

## Limitaciones

- No hay historial de la reserva: sin él, el problema 1 no tiene solución.
- El test no es ciego, y el análisis posterior lo ha leído otra vez. Las cifras de arriba no sustituyen a una validación con datos nuevos.
- Las ventanas mensuales se solapan en su entrenamiento, y sus intervalos pueden quedar estrechos. Por eso se exige también el signo en `Q4`.
- Los experimentos usan el XGBoost de referencia, y solo los más prometedores se repitieron con Random Forest. No repetí encodings ni balanceo más allá de `min_frequency`: se apoyan en los experimentos originales, que no he vuelto a ejecutar.
- Hice 50 comparaciones. El umbral de +0.002 y la regla son criterio propio, fijado antes de medir. No busqué una fuente que los respalde.
- **Las diferencias pequeñas son ruido, y hay una prueba directa.** Cambiar un valor constante en las filas sin historia, sin relación con ninguna etiqueta, movió la diferencia de `plus_agent_history` de +0.0006 a +0.0020.
- **La revisión de código encontró errores míos, corregidos antes de cerrar.** El tamaño de los sucesos contaba también las estancias de ese día: las filas "solas" y "en sucesos" daban 0.552 y 0.804 en 2016, frente a 0.633 y 0.805. El techo comparaba dos configuraciones de XGBoost. Y la tasa histórica usaba en esas filas sin historia un valor calculado con el futuro. Ninguno cambia el veredicto.
- Dos hoteles y dos años: nada se puede generalizar.

## Próximos pasos

Por orden de valor:

1. **Conseguir el registro de cambios** de la reserva, o cualquier dato con el valor al confirmar. Es lo único que resolvería los problemas 1 y 3.
2. **Reentrenar con frecuencia,** midiendo con cohortes por fecha de reserva. Es la mejora real más probable.
3. **Reformular el objetivo,** si importa el coste de verdad: predecir la cancelación tardía cerca de la llegada, con lo que se sepa entonces.
4. **Fijar el umbral según la capacidad de revisión,** no solo con la fórmula, y vigilar la proporción de avisos.

**No merece la pena seguir ajustando el modelo:** las diferencias ya están por debajo de la variabilidad.

### Afirmaciones corregidas

Corregidas en la misma rama, a petición del autor, sin cambiar el modelo ni las cifras del test. El protocolo original y el ADR 0002 se conservan como se escribieron, con una nota de corrección al final.

| Dónde | Qué decía | Problema |
|---|---|---|
| [`data_card.md`](data_card.md), [`README.md`](../README.md), [`informe_final.md`](informe_final.md), [`defensa.md`](defensa.md), [ADR 0002](adr/0002-momento-de-prediccion-y-variables-excluidas.md) | Confirmar es el único momento defendible | La fuente referencia los valores al día anterior a la llegada |
| [`data_card.md`](data_card.md), [`model_card.md`](model_card.md), [`informe_final.md`](informe_final.md), la interfaz | Las 19 se conocen al confirmar | No demostrado |
| [`data_card.md`](data_card.md), [`informe_final.md`](informe_final.md), [ADR 0002](adr/0002-momento-de-prediccion-y-variables-excluidas.md), [`defensa.md`](defensa.md), [`01_eda`](../notebooks/01_eda.ipynb) | `deposit_type` se conoce al reservar y no es una fuga | Se calcula con pagos hasta la llegada o la cancelación |
| [`README.md`](../README.md), [`informe_final.md`](informe_final.md), [`defensa.md`](defensa.md) | La cifra creíble es 0.82, y coincide con lo esperado | Mezcla reservas anteriores al corte, y coincidir en una sola muestra no lo valida |
| [`informe_final.md`](informe_final.md) | Avisar del 77.3 % es consecuencia de los costes | Incompleto: en 2016, con esos costes, era del 46 % al 63 % |
| [`informe_final.md`](informe_final.md) | Propone usar la tasa histórica del agente | Medido aquí: no mejora |
| [`informe_final.md`](informe_final.md), [`data_loader.py`](../src/hotel_cancellation/data/data_loader.py) | 0.0 % ya en train, y las filas idénticas no cruzan ventanas | Solo es cierto porque la clave incluye la llegada |

## Fuentes

| Fuente | Qué respalda | Cómo se comprobó |
|---|---|---|
| Antonio, de Almeida y Nunes (2019), *Hotel booking demand datasets*, Data in Brief 22. [PMC6297060](https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/) | Momento de captura y definición de variables | Directamente, en el texto completo |
| Antonio, de Almeida y Nunes (2019), Data Science Journal 18:32. [DOI](https://doi.org/10.5334/dsj-2019-032) | Fuga de `country` | Versión de pruebas, por un subagente; no la final |
| [scikit-learn: errores comunes](https://scikit-learn.org/stable/common_pitfalls.html) | Ajuste solo con entrenamiento, dentro de un pipeline | Por un subagente |
| [scikit-learn: validación cruzada](https://scikit-learn.org/stable/modules/cross_validation.html) | Partición temporal y por grupos | Por un subagente |
| [scikit-learn: calibración](https://scikit-learn.org/stable/modules/calibration.html) | Riesgo de sobreajuste de la isotónica y datos independientes | Por un subagente |
| [scikit-learn: validación anidada](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html) y [Cawley y Talbot (2010)](https://jmlr.org/papers/v11/cawley10a.html) | Optimismo al seleccionar y evaluar con los mismos datos | Por un subagente, en la página o el resumen |
| [Kapoor y Narayanan (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856/) | Taxonomía de fugas | Por un subagente, en el resumen |
| [Grinsztajn et al. (2022)](https://arxiv.org/abs/2207.08815) | Árboles frente a redes en datos tabulares | Por un subagente, en el resumen |
| [Bouthillier et al. (2021)](https://arxiv.org/abs/2103.03098) | Medir la variabilidad entre semillas y muestras | Por un subagente, en el resumen |

**Sin respaldo accesible, y lo digo:** no pude abrir Roberts et al. (2017) sobre validación con estructura, ni Tashman (2000) sobre origen móvil, ni el dato de tamaño de Niculescu-Mizil y Caruana (2005). Tampoco encontré una explicación publicada de la cancelación casi total de `Non Refund`. El bootstrap por tandas es una decisión mía, por analogía con el bloque del protocolo original.

## Cómo se reproduce

Cada módulo escribe en `outputs/review/`. Solo dos leen 2017: `baseline`, que recalcula las cifras publicadas desde las predicciones guardadas, y `posthoc_2017`. `feature_tiers` admite el nombre de una familia para ejecutar solo esa, y `validation_schemes`, el de una tabla.

```bash
uv run python -m experiments.review.baseline
```

```bash
uv run python -m experiments.review.data_audit
```

```bash
uv run python -m experiments.review.validation_schemes
```

```bash
uv run python -m experiments.review.ladder
```

```bash
uv run python -m experiments.review.data_treatments
```

```bash
uv run python -m experiments.review.feature_tiers
```

```bash
uv run python -m experiments.review.parsimony
```

```bash
uv run python -m experiments.review.decide
```

```bash
uv run python -m experiments.review.cadence
```

```bash
uv run python -m experiments.review.calibration_review
```

```bash
uv run python -m experiments.review.nested_search
```

```bash
uv run python -m experiments.review.errors
```

```bash
uv run python -m experiments.review.robustness
```

```bash
uv run python -m experiments.review.posthoc_2017
```
