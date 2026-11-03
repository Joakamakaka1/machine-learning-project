# Data card: reservas de hotel

Qué datos usa el proyecto, de dónde salen, cómo se limpian y por qué cada variable entra o no al modelo.

## Origen

Reservas reales y anonimizadas de dos hoteles portugueses, uno urbano y otro vacacional, con llegadas entre el 1 de julio de 2015 y el 31 de agosto de 2017: 119 390 filas y 32 columnas. El hash del archivo y la descripción de las columnas están en [`data/raw/README.md`](../data/raw/README.md).

## Cita y licencia

Antonio, N., de Almeida, A. y Nunes, L. (2019). *Hotel booking demand datasets*. Data in Brief, 22, 41-49. <https://doi.org/10.1016/j.dib.2018.11.126>

**Licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**, que permite reutilizar los datos y exige citar la fuente. El archivo se versiona comprimido y sin más cambios, salvo el nombre que le da el enunciado; `load_raw` comprueba su hash.

La fuga de `country` la documentan los mismos autores en *An Automated Machine Learning Based Decision Support System to Predict Hotel Booking Cancellations*, Data Science Journal, 18, 32. <https://doi.org/10.5334/dsj-2019-032>

## Usos y límites

- **Para qué sirve:** aprender a predecir cancelaciones con datos reales.
- **Para qué no:** generalizar a otros hoteles, o saber qué pasó durante la vida de cada reserva.
- **Personas:** no hay datos identificativos. `country` y `agent` son los campos más cercanos a una persona o empresa, y el primero queda fuera del modelo.

El CSV guarda **el valor final de cada reserva, no su historial**. Por eso importa cuándo se conoce cada variable: lo que el archivo muestra no es necesariamente lo que se sabía al confirmar la reserva.

## Limpieza

`clean` elimina las reservas imposibles. Una fila puede cumplir más de un motivo, por eso los motivos suman más que el total eliminado.

| Motivo | Filas |
|---|---:|
| Sin huéspedes (`adults + children + babies = 0`, con `children` ausente como 0) | 180 |
| Sin noches (`stays_in_week_nights + stays_in_weekend_nights = 0`) | 715 |
| `adr < 0` | 1 |
| `adr >= 5400` | 1 |
| **Filas eliminadas** | **827** |
| **Filas que quedan, de 119 390** | **118 563** |

## Particiones

| Parte | Llegadas | Filas limpias |
|---|---|---:|
| Desarrollo | 2015-07-01 a 2016-12-31 | 78 054 |
| Test | 2017-01-01 a 2017-08-31 | 40 509 |

## Momento de la predicción

La intención es predecir **al confirmar la reserva**, y una variable solo entra al modelo si hay motivos para creer que se conoce en ese momento. **Es un supuesto, no un hecho comprobado:** Antonio et al. (2019, *Data in Brief*) referencian los valores al día anterior a la llegada, y los leen de la reserva y de su registro de cambios. Este archivo no guarda el valor al confirmar.

| Grupo, según la tabla 1 del artículo | Variables | ROC-AUC en 2016 |
|---|---|---|
| Definidas al crear o confirmar | `hotel`, `lead_time`, `is_repeated_guest`, `previous_cancellations`, `previous_bookings_not_canceled`, `days_in_waiting_list` | 0.6866, solo con este grupo |
| Se pueden modificar | El resto, salvo la siguiente | 0.8072, sumando este grupo |
| Se calcula después | `deposit_type`, con los pagos anteriores a la llegada o a la cancelación | 0.8207, con las 19 |

El rendimiento real al confirmar está entre esas cifras, medidas con un XGBoost de referencia y 12 ventanas mensuales. Este archivo no permite saber dónde. Lo detalla la [segunda revisión](revision_datos_y_validacion.md).

## Registro de disponibilidad

ρ es la correlación de Spearman entre la variable y la parte de vida de la reserva transcurrida al cancelar, en Online TA. Entre paréntesis, la misma correlación dentro de grupos del mismo hotel, mes de llegada y tramo de antelación. V es la V de Cramér frente a los tramos de vida, también en Online TA. El apartado siguiente explica cómo leerlas.

### Variables de entrada (19)

Se usan como si se conocieran al confirmar la reserva, pero no consta que sean los valores de ese momento.

| Variable | Evidencia en el diagnóstico |
|---|---|
| `lead_time` | ρ +0.040 (+0.009); su relación con la vida es estructural, porque es su denominador |
| `stays_in_weekend_nights` | ρ −0.026 (−0.028) |
| `stays_in_week_nights` | ρ −0.031 (−0.034) |
| `adults` | ρ −0.020 (−0.022) |
| `children` | ρ −0.027 (−0.022) |
| `babies` | ρ +0.005 (+0.006) |
| `is_repeated_guest` | ρ −0.084 (−0.084); cae con la vida, de modo que no se acumula |
| `previous_cancellations` | ρ +0.029 (+0.019) |
| `previous_bookings_not_canceled` | ρ +0.001 (+0.006) |
| `days_in_waiting_list` | ρ −0.009 (−0.011) |
| `adr` | ρ −0.045 (−0.038); negativa en todas las cohortes |
| `hotel` | V 0.077; no puede cambiar, sirve de referencia del ruido de composición |
| `meal` | V 0.035 |
| `market_segment` | V no calculable en Online TA, por ser constante; V 0.139 en el total |
| `distribution_channel` | V 0.027 |
| `reserved_room_type` | V 0.046 |
| `deposit_type` | V 0.049; se calcula con pagos hasta la llegada o la cancelación, así que **no consta que se conozca al reservar**. Se conserva sin leerla como causa |
| `customer_type` | V 0.043 |
| `agent` | V 0.070 |

### Variables prohibidas (12)

| Variable | Cuándo se conoce | Motivo y evidencia |
|---|---|---|
| `reservation_status` | Al terminar la reserva | Repite el objetivo |
| `reservation_status_date` | Al terminar la reserva | Fecha del resultado |
| `country` | Se corrige en el check-in | Fuga documentada por Antonio et al. (2019, *Data Science Journal*). V 0.087 |
| `required_car_parking_spaces` | En el check-in o después | Ninguna de las 27 389 canceladas analizadas tiene plaza |
| `assigned_room_type` | En el check-in o después | V 0.051 |
| `booking_changes` | Durante la vida de la reserva | Cuenta cambios por definición. Es el control positivo: ρ +0.105 (+0.108) |
| `total_of_special_requests` | Durante la vida de la reserva | Supera al control: ρ +0.244 (+0.261). Se publican los dos escenarios |
| `arrival_date_year` | Al reservar | Calendario: no generaliza y añade varianza entre ventanas |
| `arrival_date_month` | Al reservar | Calendario: no generaliza y añade varianza entre ventanas |
| `arrival_date_week_number` | Al reservar | Calendario: no generaliza y añade varianza entre ventanas |
| `arrival_date_day_of_month` | Al reservar | Calendario: no generaliza y añade varianza entre ventanas |
| `company` | Al reservar | Sin efecto en validación; el modelo más simple la deja fuera |

`is_canceled` es el objetivo. `arrival_date` se construye con el año, el mes y el día solo para partir los datos, y nunca llega al modelo.

## Diagnóstico de acumulación

Una reserva cancelada deja de cambiar al cancelarse. Si una variable se acumula, debe crecer con la parte de vida transcurrida hasta ese momento: los días entre la reserva y la cancelación divididos por `lead_time`. `booking_changes`, que cuenta cambios por definición, hace de control positivo.

Se usan solo llegadas anteriores a 2017: 27 389 reservas con estado `Canceled` (quedan fuera 210 con `lead_time = 0`). Ninguna cae fuera de la vida de la reserva. De ellas, 3 648 se cancelaron en un día o menos, y 11 002 son de Online TA, el segmento mayor. Online TA se usa como control porque las tandas de `Non Refund`, en su mayoría duplicados exactos que se cancelan juntos, distorsionan la medida global.

Se reproduce con:

```bash
uv run python -m experiments.feature_availability
```

Los resultados completos están en `outputs/experiments/`.

### `total_of_special_requests` frente al control

| Cohorte | Reservas | ρ peticiones | ρ `booking_changes` | Con petición, 0-5 % de vida | Con petición, 75-100 % de vida |
|---|---:|---:|---:|---:|---:|
| Online TA | 11 002 | +0.244 | +0.105 | 24.0 % | 58.5 % |
| Online TA, City Hotel | 7 555 | +0.290 | +0.110 | 17.3 % | 56.4 % |
| Online TA, Resort Hotel | 3 447 | +0.164 | +0.100 | 38.7 % | 65.0 % |
| Filas únicas | 13 004 | +0.148 | +0.117 | 22.7 % | 42.8 % |
| `No Deposit`, filas únicas | 12 354 | +0.162 | +0.115 | 22.9 % | 44.9 % |
| City Hotel | 20 582 | +0.100 | +0.093 | 12.2 % | 21.2 % |
| Todas | 27 389 | +0.066 | +0.090 | 17.7 % | 23.2 % |
| Resort Hotel | 6 807 | +0.004 | +0.095 | 31.4 % | 29.2 % |

Las canceladas en un día también tienen menos peticiones: en Online TA, la diferencia estandarizada frente a las demás es +0.566, y la del control, +0.230.

**El efecto no es uniforme.** Supera al control en seis de las ocho cohortes, con claridad en Online TA, y no lo alcanza ni en el total ni en Resort Hotel. Por eso la exclusión es una decisión de cautela y se publican los dos escenarios, con y sin la variable.

### Otras variables de entrada

- **Ninguna variable de entrada alcanza al control por acumulación.** En Online TA, la mayor ρ positiva es la de `lead_time` (+0.040), y dentro de grupos comparables, la de `previous_cancellations` (+0.019). El control marca +0.105 y +0.108.
- **`lead_time`** supera al control en el total (+0.097 frente a +0.090) y en Resort Hotel (+0.126 frente a +0.095). Es estructural, porque `lead_time` es el denominador de la vida transcurrida: dentro de grupos comparables se queda en +0.009.
- **`adr`** no se acumula: ρ negativa en las ocho cohortes (de −0.028 a −0.136) y media que baja en Online TA de 116.5 a 110.3. El diagnóstico no descarta que se sobrescriba, y eso queda como limitación.
- **`previous_cancellations`** se acerca al control en el total (+0.075) y en City Hotel (+0.085), pero sin alcanzarlo, y cae a +0.029 en Online TA y a +0.054 en filas únicas. Apunta a las tandas de duplicados y no a acumulación.
- **`is_repeated_guest`** va en sentido contrario: del 8.1 % en el primer tramo de vida al 0.8 % en el último. Es compatible con que los clientes repetidores cancelen enseguida.
- **`days_in_waiting_list`** muestra +0.316 de diferencia estandarizada entre las canceladas en un día y las demás, pero +0.021 en Online TA. Es compatible con un efecto mecánico y con las tandas de grupos. Se conoce al confirmar por definición.
- **Categóricas de entrada:** su V en Online TA va de 0.027 a 0.077, por debajo del control (0.100). `hotel`, que no puede cambiar, marca 0.077, así que esa cifra da una idea del ruido de composición.

### Limitaciones

- El diagnóstico es observacional: quien cancela enseguida puede ser un cliente distinto. Comparar dentro de grupos reduce ese riesgo, pero no lo elimina.
- Solo mide las canceladas. Las reservas que llegan al hotel no tienen un momento de congelación comparable.
- Detecta acumulación, no sobrescritura: una variable que se reemplaza sin crecer, como una tarifa, no deja esta huella.
- Dos hoteles y dos años de datos: nada de esto se puede generalizar a otros hoteles.

## Otros problemas de los datos

Salen de la [segunda revisión](revision_datos_y_validacion.md), con llegadas anteriores a 2017.

- **Tandas de reservas:** el 53 % de las filas está en tandas de 10 o más, con la misma fecha de reserva, agente, segmento, depósito, tipo de cliente y habitación. Su correlación intraclase es de 0.47, de modo que hay mucha menos información independiente que filas. Hay días de hasta 1 112 cancelaciones de un solo agente.
- **Cobertura desigual de City Hotel:** en julio de 2015 hay 33 habitaciones ocupadas de media según los datos, frente a 212 en julio de 2016, y cancela el 67 %. Parece que faltan estancias, no cancelaciones, y no se sabe por qué. Quitar esas filas no mejora, y se conservan.
- **`previous_cancellations` no es estable:** fuera de `Non Refund` y de las tandas, su tasa de cancelación cae del 97.5 % en 2015T3 al 27.1 % en 2016T4.
- **Fechas:** `stays_in_weekend_nights` solo cuadra si el fin de semana son las noches de domingo y lunes de las fechas del CSV, que parecen desplazadas. No afecta al modelo, porque no usa el calendario, pero sí a cualquier variable de día de la semana.
- **`No-Show`:** cuenta como cancelación, y son el 3.0 % de los positivos. El artículo no lo dice de forma expresa, y se ve en los datos.
