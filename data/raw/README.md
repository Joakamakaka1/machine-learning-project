# Dataset de reservas de hotel

Reservas de dos hoteles portugueses con llegadas entre el 1 de julio de 2015 y el 31 de agosto de 2017. El objetivo es predecir si una reserva se cancela (`is_canceled = 1`) o no (`is_canceled = 0`). Es el conjunto de datos que proporciona el enunciado del máster.

## Archivo e integridad

| | |
|---|---|
| Archivo | `dataset_practica_final.csv.gz` |
| Filas y columnas | 119 390 filas y 32 columnas |
| SHA-256 del CSV sin comprimir | `d10cdeb07de9309007f528d6499a2c0467ae7a6b287b80ffd759e11e61801639` |
| SHA-256 del `.gz` | `16a2fdad1c50bfde326b54903a519559d18f0e2d0b799389951638089707330a` |
| Tamaño | 1.15 MB comprimido, 16.97 MB sin comprimir |

`load_raw` comprueba el hash del CSV sin comprimir, de modo que volver a comprimir el archivo no invalida la comprobación. El `.gz` se genera con `mtime=0` y nivel 9, y su hash solo sirve para identificar este archivo concreto.

## Cita y licencia

Antonio, N., de Almeida, A. y Nunes, L. (2019). *Hotel booking demand datasets*. Data in Brief, 22, 41-49. <https://doi.org/10.1016/j.dib.2018.11.126>

Licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), que exige citar la fuente.

## Variables

Qué variables entran al modelo y por qué se indica en [`schema.py`](../../src/hotel_cancellation/schema.py) y en la [data card](../../docs/data_card.md).

| Nombre Variable                  | Descripción                                              |
| -------------------------------- | -------------------------------------------------------- |
| `hotel`                          | Tipo de hotel: City Hotel o Resort Hotel                 |
| `is_canceled`                    | Variable objetivo: 1 si fue cancelado, 0 si no           |
| `lead_time`                      | Días entre la reserva y la fecha de llegada              |
| `arrival_date_year`              | Año de llegada                                           |
| `arrival_date_month`             | Mes de llegada                                           |
| `arrival_date_week_number`       | Número de la semana del año                              |
| `arrival_date_day_of_month`      | Día del mes de llegada                                   |
| `stays_in_weekend_nights`        | Noches de fin de semana reservadas                       |
| `stays_in_week_nights`           | Noches entre semana reservadas                           |
| `adults`                         | Número de adultos                                        |
| `children`                       | Número de niños                                          |
| `babies`                         | Número de bebés                                          |
| `meal`                           | Tipo de comida reservada                                 |
| `country`                        | País de origen del cliente                               |
| `market_segment`                 | Canal de marketing (online, offline, grupos...)          |
| `distribution_channel`           | Canal de distribución (directo, TA/TO...)                |
| `is_repeated_guest`              | 1 si el cliente ha estado anteriormente                  |
| `previous_cancellations`         | Nº de cancelaciones anteriores                           |
| `previous_bookings_not_canceled` | Nº de reservas previas no canceladas                     |
| `reserved_room_type`             | Tipo de habitación reservada                             |
| `assigned_room_type`             | Tipo de habitación asignada                              |
| `booking_changes`                | Nº de cambios en la reserva                              |
| `deposit_type`                   | Tipo de depósito: No Deposit, Refundable, etc.           |
| `agent`                          | ID del agente (puede ser nulo)                           |
| `company`                        | ID de la empresa (puede ser nulo)                        |
| `days_in_waiting_list`           | Días en lista de espera                                  |
| `customer_type`                  | Tipo de cliente: Transient, Group, etc.                  |
| `adr`                            | Average Daily Rate (precio promedio por noche)           |
| `required_car_parking_spaces`    | Plazas de parking solicitadas                            |
| `total_of_special_requests`      | Nº de peticiones especiales                              |
| `reservation_status`             | Estado final de la reserva: Check-Out, Canceled, No-Show |
| `reservation_status_date`        | Fecha en que se actualizó el estado                      |

## Valores ausentes

En el CSV, `NULL` marca el valor ausente en `agent`, `company` y `country`, y `NA` en `children`. `Undefined` en `meal`, `market_segment` y `distribution_channel` es un valor propio de la fuente, no un nulo.
