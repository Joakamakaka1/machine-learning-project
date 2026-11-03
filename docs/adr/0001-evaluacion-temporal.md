# Evaluación temporal con un test único

La evaluación oficial corta por fecha de llegada: se desarrolla con lo anterior a 2017 y se mide una sola vez en enero-agosto de 2017, sin reentrenar. Un modelo de reservas siempre se usa sobre el futuro, y una partición aleatoria mezcla fechas y reparte filas repetidas a los dos lados: el mismo XGBoost pasa de 0.9524 a 0.8683 solo por cambiar de partición, y el 32.0 % del test aleatorio ya estaba en train.

Se valida con cuatro ventanas expansivas, una por trimestre de 2016, y `hotel-ml evaluate` deja un registro y se niega a repetir sin `--force`. Cambiar esto obligaría a rehacer la búsqueda, la selección y todas las cifras publicadas.

**Consecuencia no evidente:** la cifra baja mucho y el test ya no se puede usar para mejorar nada. Lo que señala queda declarado y sin corregir.
