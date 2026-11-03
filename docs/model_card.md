# Model card: cancelación de reservas de hotel

## Modelo

| | |
|---|---|
| Tipo | Random Forest con calibración isotónica |
| Hiperparámetros | 400 árboles, profundidad máxima 14, `min_samples_leaf` 10, `max_features` 0.2 |
| Entrenamiento | Base con llegadas anteriores a 2016-10-01; calibrador en 2016T4 |
| Umbral servido | 0.25, con costes 1:3 **supuestos** |
| Artefacto | [`models/champion/`](../models/champion/), con hash y versiones en `manifest.json` |
| SHA-256 | `41d3a28592ae93367b8a88ca80ed60bb1e5ac2356ec2344d4984c0770f13d7f1` |
| Autor | Joaquín Castro Salas |

Está **congelado**: desde que se ejecutó el test no se toca el modelo, las variables ni el umbral.

## Uso previsto

- Estimar, **al confirmar una reserva**, la probabilidad de que se cancele, en hoteles como los dos del dataset. Es la intención: no consta que las variables de entrenamiento sean las de ese momento.
- Ordenar reservas para que una persona decida qué revisar, y explorar qué pasaría con otros costes en la interfaz.
- Aprender y enseñar: es un proyecto de máster y portfolio.

## Uso no previsto

- **Decidir automáticamente** sobre un cliente, como cancelarle la reserva o cambiarle las condiciones. El umbral marca el 77.3 % de las reservas.
- **Otros hoteles o mercados.** Solo ha visto dos hoteles portugueses y dos años.
- **Predecir después de confirmar.** El modelo no incorpora lo que ocurra durante la vida de la reserva.
- **Explicar causas.** La importancia de las variables no indica que cambiarlas cambie el resultado.
- **Producción sin vigilancia.** No se reentrena, y su rendimiento ya cae dentro de 2017.

## Datos

Antonio, de Almeida y Nunes (2019), *Hotel booking demand datasets*, CC BY 4.0. 19 variables de entrada, usadas como si se conocieran al confirmar. El detalle y el registro de disponibilidad están en la [data card](data_card.md).

| Parte | Llegadas | Filas |
|---|---|---:|
| Desarrollo | 2015-07-01 a 2016-12-31 | 78 054 |
| Test | 2017-01-01 a 2017-08-31 | 40 509 |

## Métricas

Test de 2017, ejecutado una sola vez, sin reentrenar. ROC-AUC de 0.8175, con un intervalo del 95 % de [0.8058, 0.8297] por bloques de hotel y fecha. El baseline da 0.5000.

**La cifra global no va sola.** Umbral de decisión de 0.25.

| Filas analizadas | Filas | Prevalencia | ROC-AUC | PR-AUC | Brier | Precisión | Recall | F1 | Coste 1:3 |
|---|---|---|---|---|---|---|---|---|---|
| Todas | 40 509 | 0.3883 | 0.8175 | 0.7608 | 0.1627 | 0.4729 | 0.9409 | 0.6294 | 0.4761 |
| Únicas | 29 667 | 0.3300 | 0.7608 | 0.5820 | 0.1807 | 0.4082 | 0.9253 | 0.5665 | 0.5166 |
| `No Deposit` | 36 356 | 0.3187 | 0.7529 | 0.5456 | 0.1810 | 0.3926 | 0.9199 | 0.5503 | 0.5301 |
| `Non Refund` | 4 126 | 0.9981 | 0.9998 | 1.0000 | 0.0010 | 0.9981 | 1.0000 | 0.9990 | 0.0019 |
| `City Hotel` | 27 396 | 0.4262 | 0.8201 | 0.7917 | 0.1644 | 0.4851 | 0.9670 | 0.6461 | 0.4797 |
| `Resort Hotel` | 13 113 | 0.3091 | 0.8020 | 0.6440 | 0.1592 | 0.4375 | 0.8658 | 0.5813 | 0.4685 |

El ROC-AUC de `Non Refund` no dice nada útil: el 99.8 % de esas filas se cancela. Las cifras completas, por modelo y por segmento, están en el [informe](informe_final.md) y en [`segments.csv`](../outputs/final_evaluation/segments.csv).

**Análisis posterior.** Leyendo las predicciones guardadas, sin puntuar ninguna fila ni cambiar nada, la [segunda revisión](revision_datos_y_validacion.md) da esto:

| Medida en 2017 | ROC-AUC |
|---|---:|
| Reservas confirmadas desde 2017-01-01 | 0.7917 |
| Las mismas, con `No Deposit` | 0.7540 |
| Canceladas hasta 7 días antes de llegar, frente a las estancias | 0.5541 |
| `No-Show`, frente a las estancias | 0.5170 |

Con bloques de hotel, día de estado y agente, el intervalo de la cifra global es [0.8010, 0.8336].

## Limitaciones

- **Umbral con muchas falsas alarmas.** Avisa del 77.3 % de las reservas, con 16 498 falsas alarmas. Es la consecuencia de los costes supuestos.
- **No es el más barato.** Con costes 1:3, cuesta 0.4761 por reserva, el peor de los cinco modelos. Se seleccionó por ROC-AUC.
- **Empata con XGBoost.** La diferencia pareada va de [-0.0033, +0.0008]. Se eligió por ser más simple.
- **PR-AUC.** La mejor es la de la MLP, con 0.7654 frente a 0.7608.
- **Deriva.** El ROC-AUC baja de 0.8742 en enero a 0.7568 en julio de 2017, y el intervalo no lo cubre. `agent` tiene un PSI de 0.7591.
- **Probabilidades en escalones.** La isotónica concentra filas en pocos valores: el 10.05 % tiene probabilidad exactamente 1.
- **Variables no comprobadas al confirmar.** El artículo del dataset referencia los valores al día anterior a la llegada, y `deposit_type` sale de los pagos hasta la llegada o la cancelación. Con solo las variables definidas al crear o confirmar, el ROC-AUC en 2016 es de 0.6866. El rendimiento real está entre esa cifra y la publicada.
- **No ve la cancelación tardía.** Detecta sobre todo tandas y reservas con mucha antelación, no las cancelaciones que dejan la habitación vacía a última hora.
- **El umbral avisó más de lo previsto.** En 2016 avisaba del 46 % al 63 % de las reservas, y en 2017 del 77.3 %.
- **El test no es estrictamente ciego.** Una auditoría previa había visto 2017, y de ella salen `ADR_UPPER_LIMIT`, `min_frequency=50` y las rejillas de búsqueda. Sus predicciones guardadas se han vuelto a leer después, sin que salga ninguna decisión.
- **`total_of_special_requests` queda fuera por cautela.** Con ella, el ROC-AUC sería de 0.8369. El [informe](informe_final.md) publica ese escenario.

## Sesgos a vigilar

No hay variables de personas como edad o género, y `country` queda fuera. Aun así, conviene vigilar:

- **Rendimiento desigual por grupo.** En `Online TA` el ROC-AUC es 0.7203, frente a 0.9105 en `Groups`. En `Direct` es 0.6919, con pocas cancelaciones. El modelo acierta menos justo donde hay más incertidumbre.
- **Segmentos pequeños.** `Aviation` y `Complementary`, con 106 y 206 filas, tienen cifras poco fiables.
- **Clientes repetidores y `previous_cancellations`.** Penalizar a quien ya canceló puede tratar peor a un cliente por su pasado. No se ha medido el efecto de ninguna acción, porque el dataset no las recoge.
- **`agent` y `deposit_type`.** Pueden hacer que un cliente reciba peor trato por el canal por el que reservó, no por su conducta.
- **Bucle de realimentación.** Si el hotel actúa sobre las reservas marcadas, los datos futuros ya no reflejarán lo que habría pasado sin actuar.

**Por eso el modelo debe apoyar la decisión de una persona, y las acciones sobre clientes deben revisarse aparte.**

## Cómo se usa

La API, la interfaz, Docker y los comandos están en el [README](../README.md). `hotel-ml evaluate` se niega a repetir el test sin `--force`, y hacerlo habría que declararlo.
