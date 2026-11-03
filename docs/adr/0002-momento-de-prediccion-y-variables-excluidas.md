# Se predice al confirmar la reserva, y solo entran variables conocidas entonces

El CSV guarda el valor final de cada reserva, no su historial, de modo que confirmar es el único momento defendible: cualquier otro necesitaría datos que no existen. Quedan 19 variables de entrada, y 12 prohibidas en `schema.py`, con un test que falla si alguna llega al modelo.

- **Conocidas tarde o iguales al objetivo:** `reservation_status`, `reservation_status_date`, `country`, `required_car_parking_spaces` y `assigned_room_type`. La fuga de `country` la documentan Antonio, de Almeida y Nunes (2019).
- **Se acumulan:** `booking_changes` y `total_of_special_requests`.
- **Calendario:** año, mes, semana y día de llegada, que no generalizan a un año que el modelo no ha visto.
- **Sin efecto:** `company`.

**La decisión discutible es `total_of_special_requests`.** Supera al control de acumulación en seis de ocho cohortes, no en todas, y el diagnóstico es observacional. Excluirla baja la cifra de validación, y por eso se fijó antes del test un escenario informativo con ella, que da 0.8369 frente a 0.8175 y no puede cambiar el campeón.

`deposit_type` se conserva aunque `Non Refund` casi siempre se cancele, porque se conoce al reservar. Por eso el rendimiento se publica también sin esas filas.

## Corrección (2026-11-03)

La decisión se mantiene, pero dos de sus razones eran erróneas, y las detecta la [segunda revisión](../revision_datos_y_validacion.md).

- **Confirmar no es el único momento defendible, sino una intención.** Antonio et al. (2019, *Data in Brief*) referencian los valores al día anterior a la llegada, de modo que no consta que las 19 variables sean las conocidas al confirmar.
- **No consta que `deposit_type` se conozca al reservar.** El artículo la calcula con los pagos anteriores a la llegada o a la cancelación. Se conserva porque quitarla no resuelve el problema: sin ella, el ROC-AUC en 2016 solo baja de 0.8207 a 0.8072, porque otras variables recogen parte de su señal.

Esto refuerza las exclusiones que se tomaron por cautela, como `total_of_special_requests`.
