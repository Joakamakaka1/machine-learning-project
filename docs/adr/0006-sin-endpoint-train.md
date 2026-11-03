# La API no tiene endpoint `/train`

El enunciado cita `/train` como ejemplo, y aquí no existe. Entrenar tarda mucho y consume mucha memoria, exigiría TensorFlow en la imagen del servicio, y dejaría que una petición HTTP sustituyera el modelo que el protocolo congeló. Se entrena con `hotel-ml`, y la API solo sirve lo ya evaluado.

Por la misma razón, `/evaluate` devuelve la evaluación guardada y nunca calcula sobre el test.
