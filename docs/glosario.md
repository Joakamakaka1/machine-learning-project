# Glosario

Los términos que hay que entender para seguir este proyecto, en lenguaje claro y con lo que significan aquí. Las cifras salen de `outputs/` y del informe final.

## El problema

**Reserva**:
Una fila del dataset: una estancia reservada en uno de los dos hoteles, con los valores que tenía al terminar.

**Cancelación (`is_canceled`)**:
Lo que se predice. Vale 1 si la reserva acabó como `Canceled` o `No-Show`, y 0 si el cliente llegó. Es el 36.4 % de las filas de desarrollo.

**Momento de predicción**:
Cuándo se supone que el modelo decide. Aquí, **al confirmar la reserva**. Marca qué variables valen: solo las que deberían conocerse entonces. Es un supuesto, porque el dataset referencia sus valores al día anterior a la llegada.

**Antelación (`lead_time`)**:
Días entre la reserva y la llegada. Es una de las variables más útiles: se cancela el 10.1 % con menos de una semana y el 77.8 % con más de un año.

**Tarifa (`adr`)**:
Precio medio por noche. Ya es diaria, de modo que dividirla por las noches sería un error. Se conserva, aunque no se puede demostrar que no se sobrescriba.

**`deposit_type` y `Non Refund`**:
El tipo de depósito. `Non Refund` significa sin devolución, y aun así se cancela casi siempre: el 99.2 % en desarrollo. Son tandas de reservas casi iguales, no un efecto de pagar. Por eso se mide aparte.

**Online TA**:
Reservas hechas por agencias de viajes en línea, el mayor segmento de mercado. El diagnóstico de acumulación se hace aquí, porque las tandas de `Non Refund` lo distorsionan menos que al total.

## Evaluar sin engañarse

**Fuga de datos**:
Que el modelo aprenda de algo que no sabría en el momento de predecir. Da cifras altas que luego no se cumplen. Aquí, `country` se corrige en el check-in, y `reservation_status` repite la respuesta.
_Evitar_: "data leakage" sin explicar de qué dato.

**Inflación**:
Lo que sube una métrica por evaluar mal. Este proyecto la mide por escalones: 0.9524 con partición aleatoria, 0.8683 con temporal, 0.8184 sin `country` ni `booking_changes`.

**Partición aleatoria**:
Repartir las filas al azar entre train y test. Es lo habitual en los ejemplos, y aquí es un error: mezcla fechas y deja filas repetidas a los dos lados.

**Partición temporal**:
Entrenar con el pasado y medir en el futuro, como se usará el modelo. Aquí, se corta por fecha de llegada.

**Filas únicas**:
Las que no repiten otra en las 19 variables de entrada y el objetivo. Las repetidas son el 35.9 % en desarrollo. Son la versión honesta de la cifra: 0.7608 frente a 0.8175.

**Ventana expansiva**:
Una forma de validar en el tiempo: se valida en un trimestre de 2016 y se entrena con todo lo anterior a él, y el entrenamiento crece en cada ventana. Hay cuatro.

**Test único**:
Enero-agosto de 2017, medido una sola vez y al final. Si se repitiera para mejorar, dejaría de estimar el futuro. El registro está en `outputs/final_evaluation/record.json`.

**2017 no es virgen**:
Una auditoría previa ya había visto esos datos, y de ella salen tres elecciones. El test no es estrictamente ciego, y el proyecto lo declara junto a la cifra.

**Modelo congelado**:
Desde que se ejecutó el test no se toca el modelo, las variables ni el umbral, aunque el test señale algo mejorable.

**Variable que se acumula**:
La que crece durante la vida de la reserva. Las que se cancelan pronto se quedan con valores bajos, y el modelo aprendería eso aunque al confirmar nadie lo supiera.

**Diagnóstico de acumulación**:
Mide si una variable crece con la parte de vida de la reserva transcurrida al cancelar, porque una reserva cancelada ya no cambia. Es observacional: indica, pero no demuestra.

**Control positivo**:
Algo que sabes que debe dar positivo, para calibrar la prueba. Aquí, `booking_changes`, que cuenta cambios por definición.

**Escenario informativo**:
Un modelo igual al campeón con `total_of_special_requests` añadida, medido en el test. Solo informa de cuánto se renuncia por cautela: da 0.8369, y no puede cambiar el campeón.

## Métricas

**Baseline (`Dummy`)**:
Un modelo que no aprende nada y que hay que superar. Da 0.5000 de ROC-AUC, y por eso una cifra solo significa algo comparada con él.

**ROC-AUC**:
La probabilidad de que una reserva cancelada reciba más riesgo que una no cancelada. 0.5 es azar, y 1 es perfecto. Es la métrica principal porque no depende del umbral. Solo mide el orden: no dice si la probabilidad es correcta ni cuánto cuesta decidir.

**PR-AUC**:
El área bajo la curva de precisión y recall. Se fija más en la clase positiva, y su referente es la prevalencia: 0.3883 en el test. La mejor es la de la MLP, con 0.7654.

**Accuracy**:
Parte de las decisiones que aciertan. Engaña con clases desiguales: no avisar nunca acierta el 61.2 % en el test.

**Precisión**:
De las reservas de las que se avisa, cuántas se cancelan. Baja si hay muchas falsas alarmas: 0.4729 con el umbral servido.

**Recall**:
De las reservas que se cancelan, de cuántas se avisa: 0.9409 con el umbral servido. Sube y baja con el umbral, al contrario que la precisión.

**F1**:
Una media de precisión y recall, siempre de la clase positiva. No distingue qué error es más caro.

**Matriz de confusión**:
La tabla de aciertos y errores con un umbral: verdaderos y falsos positivos, y verdaderos y falsos negativos.

**Brier**:
El error cuadrático medio de la probabilidad. Mide a la vez si ordena bien y si la probabilidad es creíble: cuanto más bajo, mejor.

**Pendiente de calibración**:
Si es 1, las probabilidades están bien de escala. Por debajo de 1, el modelo es demasiado extremo; por encima, demasiado tímido. Con isotónica depende del recorte, y el proyecto da los tres valores.

**Prevalencia**:
La proporción de cancelaciones en un grupo. Cambia mes a mes, y por eso importa medir con fechas.

**Bootstrap por bloques**:
Una forma de calcular un intervalo repitiendo la medida con muestras de los datos. Se toman bloques de hotel y fecha, y no filas sueltas, porque las reservas del mismo día no son independientes.

**Intervalo de confianza (IC)**:
El rango plausible de la cifra por el azar de la muestra. Aquí, [0.8058, 0.8297]. No cubre que el futuro cambie: eso es la deriva.

**Deriva**:
Que los datos del futuro se parezcan menos a los de entrenamiento. El ROC-AUC baja de 0.8742 en enero a 0.7568 en julio de 2017.

**PSI**:
Un número que mide cuánto se ha movido la distribución de una variable. Como guía, 0.10 es moderado y 0.25, grande. `agent` marca 0.7591.

## Modelos

**Regresión logística**:
Un modelo lineal, el más simple de los comparados. Da 0.7956, y le cuestan las relaciones que no son rectas.

**Árbol de decisión**:
Una serie de preguntas sobre las variables. Fácil de leer, pero varía mucho si cambian los datos.

**Random Forest**:
Muchos árboles, cada uno con parte de los datos y de las variables, cuyas respuestas se promedian. Es el **campeón**: 400 árboles de profundidad máxima 14.

**XGBoost**:
Árboles que se entrenan uno detrás de otro para corregir a los anteriores. Empata con Random Forest.

**MLP**:
Una red neuronal de capas densas, hecha con Keras. No los iguala en estos datos, aunque tiene la mejor PR-AUC.

**One-hot con `min_frequency`**:
Convierte cada categoría en una columna de ceros y unos. Con `min_frequency=50`, las categorías con menos de 50 filas se agrupan, y las desconocidas también caben ahí.

**Embeddings categóricos**:
Representan cada categoría con unos pocos números que la red aprende, en lugar de una columna por valor. Se probaron, y aportan +0.0010 frente a one-hot, así que quedan como experimento.

**Balanceo de clases**:
Cambiar el peso o la cantidad de cada clase al entrenar, con SMOTE, undersampling o `class_weight`. Se midió y no mejora, y además distorsiona la probabilidad.

**Pipeline**:
La cadena de scikit-learn que aplica imputación, codificación y modelo como una sola pieza. Evita que el preprocesado aprenda de los datos de validación.

## Elegir y decidir

**`RandomizedSearchCV`**:
Prueba configuraciones al azar de hiperparámetros, en lugar de todas. Aquí, 20 como máximo, y 6 en la MLP, con las ventanas de 2016.

**Hiperparámetro**:
Lo que se fija antes de entrenar, como la profundidad de los árboles.

**Regla de 1 error estándar**:
Considera empatados a los modelos cuya pérdida frente al mejor cabe en un error estándar, y elige el más simple. Evita elegir por ruido.

**Calibración**:
Ajustar las probabilidades para que un 30 % signifique que se cancela el 30 %. Se aprende en un trimestre que el modelo base no ha visto.

**Isotónica y sigmoide**:
Dos formas de calibrar. La sigmoide ajusta una curva suave; la isotónica, una función en escalones, más flexible, y por eso muchas filas comparten el mismo valor.

**Umbral (`t*`)**:
La probabilidad a partir de la cual se avisa. Aquí sale de `C_FP / (C_FP + C_FN)`: 0.25 con costes 1:3.

**Costes (`C_FP`, `C_FN`)**:
Lo que se supone que cuesta una falsa alarma y una cancelación perdida. **Son supuestos**, no datos del hotel, y se publican tres escenarios: 1:2, 1:3 y 1:5.

**Campeón**:
El modelo elegido y servido: el Random Forest con isotónica de `models/champion/`.
_Evitar_: "mejor modelo", porque en el test no es el mejor en todo.

**Manifest**:
El `manifest.json` que acompaña al modelo: hash, versiones, umbral y métricas. Permite comprobar que lo servido es lo evaluado.

**Hash (SHA-256)**:
Una huella del archivo: si cambia un solo byte, cambia la huella. La API la comprueba al arrancar.

## Interpretar

**`feature_importances_`**:
Cuánto reduce la impureza cada variable en los árboles. Rápida, pero favorece a las variables con muchas categorías.

**Permutación**:
Cuánto cae el ROC-AUC si se baraja una variable. Mide de cuánto depende el modelo, no la causa.

**SHAP**:
Reparte cada predicción entre las variables. Aquí se calcula sobre la probabilidad sin calibrar del modelo base.

## Sistema

**MLflow**:
Guarda cada entrenamiento con sus parámetros, métricas y modelo. Aquí usa un SQLite local, `mlflow.db`.

**API (FastAPI)**:
El servicio que recibe una reserva y devuelve la probabilidad y la decisión. Sin `/train`.

**Interfaz (Streamlit)**:
Las páginas web en español que consumen la API.

**Docker Compose**:
Levanta la API, la interfaz y MLflow con un comando, en contenedores y solo en `127.0.0.1`.

**Paridad**:
Que la API dé las mismas probabilidades que se guardaron en el test. Lo comprueba `scripts/verify_served_predictions.py`, y así se demuestra que un modelo creado en Windows se comporta igual en Linux.
