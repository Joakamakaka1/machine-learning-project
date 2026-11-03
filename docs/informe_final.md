# Informe final

> Esta versión recoge el resultado del test, es decir, la fase 6. La documentación completa llegará con la fase 10.

## Resultado

**El campeón, un Random Forest con calibración isotónica, obtiene un ROC-AUC de 0.8175 en el test de 2017, con un intervalo del 95 % de [0.8058, 0.8297].** El baseline da 0.5000. El test se ejecutó una sola vez, con 40 509 reservas y una prevalencia de 0.3883.

- **Coincide con lo esperado:** el protocolo anticipaba unos 0.81, y la validación en 2016 dio 0.8172 ± 0.0231.
- **La cifra global no va sola:** con filas únicas baja a 0.7608, y con `No Deposit` a 0.7529. Esa es la dificultad real cuando se dejan fuera los duplicados y los `Non Refund`, que casi siempre se cancelan.
- **El 0.952 de partida estaba inflado:** la auditoría lo reproduce y explica de dónde sale.

## Qué se ha ejecutado

- **Registro:** [`outputs/final_evaluation/record.json`](../outputs/final_evaluation/record.json), el 2026-11-02, con el código en `185aa76` y sin `--force`.
- **Modelo:** SHA-256 `41d3a28592ae93367b8a88ca80ed60bb1e5ac2356ec2344d4984c0770f13d7f1`, entrenado con llegadas anteriores a 2016-10-01 y calibrado en 2016T4. Se evalúa sin reentrenar.
- **El test no es estrictamente ciego:** una auditoría previa había visto 2017, y de ella salen `ADR_UPPER_LIMIT`, `min_frequency=50` y las rejillas de búsqueda. Lo detalla el [protocolo](protocolo.md).
- **Modelo congelado:** desde la ejecución no se toca el modelo, las variables ni el umbral.
- **`precision=0` del `Dummy` en validación:** las tablas de validación lo conservan. En las del test, esa celda vale `n/d`.

Toda probabilidad es la calibrada, salvo en SHAP. Los costes son supuestos.

## Modelos y baseline

Umbral de decisión de 0.25, costes 1:3.

| Modelo | ROC-AUC | IC 95 %, hotel y fecha | IC 95 %, semana | PR-AUC | Brier | Pendiente | Accuracy | Precisión | Recall | F1 | Coste 1:3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Random Forest (campeón)** | 0.8175 | [0.8058, 0.8297] | [0.8008, 0.8339] | 0.7608 | 0.1627 | 0.959 | 0.5698 | 0.4729 | 0.9409 | 0.6294 | 0.4761 |
| XGBoost | 0.8164 | [0.8046, 0.8284] | [0.7990, 0.8333] | 0.7564 | 0.1639 | 0.908 | 0.6062 | 0.4962 | 0.9214 | 0.6450 | 0.4549 |
| MLP | 0.8039 | [0.7907, 0.8178] | [0.7846, 0.8228] | 0.7654 | 0.1667 | 1.017 | 0.6809 | 0.5599 | 0.8329 | 0.6696 | 0.4489 |
| Árbol de decisión | 0.8040 | [0.7921, 0.8169] | [0.7851, 0.8227] | 0.7147 | 0.1660 | 1.141 | 0.6240 | 0.5089 | 0.9031 | 0.6510 | 0.4512 |
| Regresión logística | 0.7956 | [0.7809, 0.8096] | [0.7734, 0.8161] | 0.7353 | 0.1700 | 0.912 | 0.6424 | 0.5240 | 0.8618 | 0.6517 | 0.4650 |
| Dummy (baseline) | 0.5000 | [0.5000, 0.5000] | [0.5000, 0.5000] | 0.3883 | 0.2384 | n/d | 0.3883 | 0.3883 | 1.0000 | 0.5594 | 0.6117 |

La pendiente recorta las probabilidades a `[1e-3, 1 - 1e-3]`. Con el campeón vale 0.944 con `1e-6` y 1.000 con `1e-2`. El 0.46% de sus filas tiene probabilidad exactamente 0, y el 10.05%, exactamente 1.

### Con un umbral fijo de 0.5

Sirve para comparar con otros trabajos. No es el umbral servido.

| Modelo | Accuracy | Precisión | Recall | F1 | VP | FP | FN | VN |
|---|---|---|---|---|---|---|---|---|
| **Random Forest (campeón)** | 0.7484 | 0.6833 | 0.6563 | 0.6695 | 10 324 | 4 785 | 5 406 | 19 994 |
| XGBoost | 0.7455 | 0.6721 | 0.6728 | 0.6724 | 10 583 | 5 163 | 5 147 | 19 616 |
| MLP | 0.7422 | 0.6982 | 0.5919 | 0.6406 | 9 310 | 4 025 | 6 420 | 20 754 |
| Árbol de decisión | 0.7310 | 0.6274 | 0.7571 | 0.6861 | 11 909 | 7 074 | 3 821 | 17 705 |
| Regresión logística | 0.7363 | 0.7473 | 0.4850 | 0.5882 | 7 629 | 2 580 | 8 101 | 22 199 |
| Dummy (baseline) | 0.6117 | n/d | 0.0000 | 0.0000 | 0 | 0 | 15 730 | 24 779 |

### Costes

| Modelo | 1:2 en su t* | 1:3 en su t* | 1:5 en su t* | 1:2 con umbral 0.25 | 1:5 con umbral 0.25 | VP | FP | FN | VN |
|---|---|---|---|---|---|---|---|---|---|
| **Random Forest (campeón)** | 0.3524 | 0.4761 | 0.5217 | 0.4531 | 0.5219 | 14 801 | 16 498 | 929 | 8 281 |
| XGBoost | 0.3517 | 0.4549 | 0.5159 | 0.4244 | 0.5159 | 14 494 | 14 718 | 1 236 | 10 061 |
| MLP | 0.3660 | 0.4489 | 0.5488 | 0.3840 | 0.5786 | 13 102 | 10 300 | 2 628 | 14 479 |
| Árbol de decisión | 0.3571 | 0.4512 | 0.5163 | 0.4136 | 0.5265 | 14 206 | 13 707 | 1 524 | 11 072 |
| Regresión logística | 0.3851 | 0.4650 | 0.5812 | 0.4113 | 0.5723 | 13 556 | 12 314 | 2 174 | 12 465 |
| Dummy (baseline) | 0.6117 | 0.6117 | 0.6117 | 0.6117 | 0.6117 | 15 730 | 24 779 | 0 | 0 |

### ROC-AUC por subconjunto

| Modelo | Todas | Únicas | `No Deposit` | `Non Refund` | Diferencia con el campeón, IC 95 % |
|---|---|---|---|---|---|
| **Random Forest (campeón)** | 0.8175 | 0.7608 | 0.7529 | 0.9998 | — |
| XGBoost | 0.8164 | 0.7596 | 0.7516 | 0.9998 | [-0.0033, +0.0008] |
| MLP | 0.8039 | 0.7452 | 0.7349 | 0.9925 | [-0.0179, -0.0101] |
| Árbol de decisión | 0.8040 | 0.7399 | 0.7356 | 0.4676 | [-0.0170, -0.0099] |
| Regresión logística | 0.7956 | 0.7405 | 0.7240 | 0.9055 | [-0.0276, -0.0167] |
| Dummy (baseline) | 0.5000 | 0.5000 | 0.5000 | 0.5000 | [-0.3297, -0.3058] |

![Curvas ROC](../outputs/figures/roc.png)

![Curvas de precisión y recall](../outputs/figures/precision_recall.png)

![ROC-AUC con sus intervalos](../outputs/figures/intervalos_roc_auc.png)

![Matrices de confusión de los modelos](../outputs/figures/matrices_confusion_modelos.png)

![Calibración](../outputs/figures/calibracion.png)

## Segmentos del campeón

Umbral de decisión de 0.25. El resto de segmentos está en [`segments.csv`](../outputs/final_evaluation/segments.csv).

| Filas analizadas | Filas | Prevalencia | ROC-AUC | PR-AUC | Brier | Pendiente | Precisión | Recall | F1 | Coste 1:3 |
|---|---|---|---|---|---|---|---|---|---|---|
| Todas | 40 509 | 0.3883 | 0.8175 | 0.7608 | 0.1627 | 0.959 | 0.4729 | 0.9409 | 0.6294 | 0.4761 |
| Únicas | 29 667 | 0.3300 | 0.7608 | 0.5820 | 0.1807 | 0.937 | 0.4082 | 0.9253 | 0.5665 | 0.5166 |
| `No Deposit` | 36 356 | 0.3187 | 0.7529 | 0.5456 | 0.1810 | 0.949 | 0.3926 | 0.9199 | 0.5503 | 0.5301 |
| `Non Refund` | 4 126 | 0.9981 | 0.9998 | 1.0000 | 0.0010 | 3.996 | 0.9981 | 1.0000 | 0.9990 | 0.0019 |
| `City Hotel` | 27 396 | 0.4262 | 0.8201 | 0.7917 | 0.1644 | 1.051 | 0.4851 | 0.9670 | 0.6461 | 0.4797 |
| `Resort Hotel` | 13 113 | 0.3091 | 0.8020 | 0.6440 | 0.1592 | 0.903 | 0.4375 | 0.8658 | 0.5813 | 0.4685 |

La pendiente de `Non Refund` no se debe leer: el 99.8% de esas filas se cancela.

## Mes a mes

| Mes | Filas | Prevalencia | ROC-AUC | Brier |
|---|---|---|---|---|
| 2017-01 | 3 650 | 0.3422 | 0.8742 | 0.1299 |
| 2017-02 | 4 142 | 0.3271 | 0.8165 | 0.1520 |
| 2017-03 | 4 933 | 0.3387 | 0.8340 | 0.1461 |
| 2017-04 | 5 648 | 0.4352 | 0.8358 | 0.1588 |
| 2017-05 | 6 291 | 0.4389 | 0.8312 | 0.1597 |
| 2017-06 | 5 632 | 0.4329 | 0.8075 | 0.1693 |
| 2017-07 | 5 298 | 0.3745 | 0.7568 | 0.1890 |
| 2017-08 | 4 915 | 0.3691 | 0.7697 | 0.1854 |

![ROC-AUC por mes](../outputs/figures/roc_auc_por_mes.png)

![Prevalencia por mes](../outputs/figures/prevalencia_por_mes.png)

## Deriva

PSI de cada variable frente a las llegadas anteriores a 2016-10-01, con las que aprendió el modelo base.

| Variable | PSI |
|---|---|
| `agent` | 0.7591 |
| `previous_cancellations` | 0.2379 |
| `adr` | 0.1706 |
| `customer_type` | 0.1321 |
| `days_in_waiting_list` | 0.1233 |
| `market_segment` | 0.1182 |
| `meal` | 0.0677 |
| `reserved_room_type` | 0.0343 |
| `lead_time` | 0.0285 |
| `stays_in_week_nights` | 0.0159 |
| `deposit_type` | 0.0138 |
| `children` | 0.0074 |
| `adults` | 0.0072 |
| `stays_in_weekend_nights` | 0.0052 |
| `previous_bookings_not_canceled` | 0.0040 |
| `distribution_channel` | 0.0031 |
| `hotel` | 0.0019 |
| `is_repeated_guest` | 0.0013 |
| `babies` | 0.0009 |

![Deriva por variable](../outputs/figures/deriva_psi.png)

## Escenario con `total_of_special_requests`

**Es solo informativo y no puede cambiar el campeón.** Es un Random Forest con los mismos hiperparámetros, ventanas y calibración, y con esa variable añadida. El diagnóstico de la [data card](data_card.md) indica que se acumula durante la vida de la reserva, aunque es observacional, y por eso se publican los dos escenarios.

ROC-AUC de 0.8369, con un intervalo de [0.8260, 0.8484]. Frente al campeón, la diferencia pareada va de [+0.0155, +0.0233] con bloques de hotel y fecha, y de [+0.0148, +0.0239] por semana.

| Filas analizadas | ROC-AUC oficial | ROC-AUC escenario | PR-AUC oficial | PR-AUC escenario | Brier oficial | Brier escenario | Pendiente escenario | Recall escenario | F1 escenario | Coste 1:3 escenario |
|---|---|---|---|---|---|---|---|---|---|---|
| Todas | 0.8175 | 0.8369 | 0.7608 | 0.7835 | 0.1627 | 0.1577 | 0.776 | 0.8975 | 0.6775 | 0.4114 |
| Únicas | 0.7608 | 0.7782 | 0.5820 | 0.6122 | 0.1807 | 0.1785 | 0.724 | 0.8705 | 0.6037 | 0.4626 |
| `No Deposit` | 0.7529 | 0.7792 | 0.5456 | 0.5856 | 0.1810 | 0.1754 | 0.757 | 0.8609 | 0.5977 | 0.4580 |
| `Non Refund` | 0.9998 | 0.9997 | 1.0000 | 1.0000 | 0.0010 | 0.0011 | 3.764 | 1.0000 | 0.9990 | 0.0019 |

## Valor de negocio

**Los costes son supuestos, no datos del hotel.** Cada fila indica qué haría el campeón si un falso positivo costara lo que dice el escenario. El coste sin modelo es el de la mejor política trivial: avisar de todas las reservas o de ninguna.

| Costes FP:FN (supuestos) | Umbral | Recall | Precisión | Cancelaciones detectadas | Falsas alarmas | Cancelaciones perdidas | % de no canceladas avisadas | % de reservas avisadas | Coste por reserva | Coste sin modelo | Ahorro por reserva |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1:2 | 0.333 | 0.7991 | 0.6124 | 12 570 | 7 956 | 3 160 | 32.1 % | 50.7 % | 0.3524 | 0.6117 | 0.2593 |
| 1:3 | 0.250 | 0.9409 | 0.4729 | 14 801 | 16 498 | 929 | 66.6 % | 77.3 % | 0.4761 | 0.6117 | 0.1356 |
| 1:5 | 0.167 | 0.9411 | 0.4729 | 14 803 | 16 500 | 927 | 66.6 % | 77.3 % | 0.5217 | 0.6117 | 0.0900 |

![Matrices de confusión del campeón por escenario](../outputs/figures/matrices_confusion_campeon.png)

![Coste según el umbral](../outputs/figures/coste_umbral.png)

## Auditoría de la inflación

Se reproduce con `hotel-ml audit`, y los tres primeros escalones se entrenan de nuevo. La cuarta fila copia la cifra publicada.

| Escalón | Modelo | Filas de train | Filas de test | Test ya en train | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|---|---|---|
| Partición aleatoria, variables de partida | XGBoost de partida | 94 850 | 23 713 | 32.0 % | 0.9524 | 0.9339 | 0.0838 |
| Partición temporal, variables de partida | XGBoost de partida | 78 054 | 40 509 | 0.0 % | 0.8683 | 0.8270 | 0.1630 |
| Partición temporal, sin country ni booking_changes | XGBoost de partida | 78 054 | 40 509 | 0.0 % | 0.8184 | 0.7802 | 0.1775 |
| Modelo oficial: 19 variables, sin calendario ni variables acumuladas | Random Forest | 78 054 | 40 509 | n/d | 0.8175 | 0.7608 | 0.1627 |

![De la cifra inflada a la oficial](../outputs/figures/inflacion.png)

- **De 0.952 a 0.868:** es el efecto de evaluar en el futuro. En la partición aleatoria, el 32.0 % de las filas de test ya estaba en train.
- **De 0.868 a 0.818:** es lo que aportaban `country` y `booking_changes`.
- **El tercer escalón y el oficial casi coinciden, y es casualidad.** El tercero conserva `total_of_special_requests`, el calendario y árboles de profundidad 16, y esos efectos se compensan.

## Interpretación

**Sirve para interpretar, no decide nada y no indica causas.** Los valores se agregan por variable original.

- `feature_importances_`: sale del modelo base, y favorece a las variables con muchas categorías.
- Permutación: caída del ROC-AUC del campeón al barajar cada variable en 2017, con 10 repeticiones.
- SHAP: 2000 filas de 2017, en la probabilidad sin calibrar del modelo base.

| Variable | `feature_importances_` | Permutación (caída de ROC-AUC) | ± desviación | |SHAP| medio |
|---|---|---|---|---|
| `deposit_type` | 0.4352 | 0.1502 | 0.0018 | 0.1281 |
| `lead_time` | 0.0908 | 0.0583 | 0.0015 | 0.0475 |
| `agent` | 0.0821 | 0.0437 | 0.0008 | 0.0431 |
| `customer_type` | 0.0454 | 0.0332 | 0.0007 | 0.0326 |
| `market_segment` | 0.0568 | 0.0210 | 0.0008 | 0.0335 |
| `previous_bookings_not_canceled` | 0.0061 | 0.0053 | 0.0001 | 0.0032 |
| `adr` | 0.0318 | 0.0050 | 0.0005 | 0.0109 |
| `hotel` | 0.0098 | 0.0048 | 0.0006 | 0.0068 |
| `distribution_channel` | 0.0143 | 0.0041 | 0.0003 | 0.0086 |
| `previous_cancellations` | 0.1787 | 0.0030 | 0.0004 | 0.0340 |
| `stays_in_week_nights` | 0.0115 | 0.0026 | 0.0002 | 0.0046 |
| `adults` | 0.0058 | 0.0021 | 0.0003 | 0.0019 |
| `meal` | 0.0087 | 0.0020 | 0.0001 | 0.0046 |
| `stays_in_weekend_nights` | 0.0066 | 0.0009 | 0.0002 | 0.0022 |
| `is_repeated_guest` | 0.0026 | 0.0008 | 0.0001 | 0.0005 |
| `reserved_room_type` | 0.0046 | 0.0007 | 0.0002 | 0.0020 |
| `children` | 0.0013 | 0.0006 | 0.0001 | 0.0014 |
| `babies` | 0.0001 | 0.0001 | 0.0000 | 0.0000 |
| `days_in_waiting_list` | 0.0076 | -0.0000 | 0.0000 | 0.0014 |

![Importancia de las variables](../outputs/figures/importancia_variables.png)

`deposit_type` domina en los tres métodos. Se conoce al reservar, pero no debe leerse como causa: `Non Refund` concentra grupos de reservas casi idénticas que se cancelan juntas.

## Lo que el test señala y no se corrige

Se declara tal cual. El modelo sigue congelado.

1. **El campeón no es el más barato con su propio umbral.** Con costes 1:3 cuesta 0.4761 por reserva, el peor de los cinco modelos: MLP 0.4489, árbol 0.4512, XGBoost 0.4549 y regresión logística 0.4650. La selección se hizo por ROC-AUC, como manda el protocolo.
2. **El umbral 0.25 cae justo antes de un gran escalón de la isotónica.** Con umbrales entre 0.27 y 0.30, el coste 1:3 baja a unos 0.43. Por el mismo motivo, los escenarios 1:3 y 1:5 toman casi las mismas decisiones: solo 4 filas tienen una probabilidad entre 0.167 y 0.25.
3. **El umbral servido da muchas falsas alarmas.** Avisa del 77.3% de las reservas, con 16 498 falsas alarmas. Es la consecuencia de suponer que perder una cancelación cuesta tres veces más, no una decisión empírica.
4. **XGBoost no se distingue del campeón.** La diferencia pareada va de [-0.0033, +0.0008], y la regla de selección ya los había dejado empatados en validación.
5. **El rendimiento cae al final del periodo.** El ROC-AUC baja de 0.8742 en enero a 0.7568 en julio. El intervalo del bootstrap no cubre esta deriva.
6. **`agent` deriva mucho** (PSI 0.7591) y `previous_cancellations` algo menos (0.2379). Es una limitación de usar identificadores de agente en un modelo que no se reentrena.

## Limitaciones

- Dos hoteles y dos años de datos: no se puede generalizar a otros hoteles.
- `adr` y otras variables son la foto final de la reserva, y su disponibilidad al confirmarla no se puede demostrar del todo.
- Los costes son supuestos.
- El test no es estrictamente ciego, como se declara arriba.
