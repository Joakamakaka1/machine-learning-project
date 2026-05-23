# Guía ML: Predicción de Cancelaciones

## 1. El Problema de Negocio: ¿Por qué predecimos cancelaciones?

Imagínate que gestionas un hotel. Si un cliente reserva una habitación y al final no viene (cancela a última hora), esa habitación se queda vacía y tú pierdes dinero. Para evitarlo, los hoteles hacen **overbooking** (venden más habitaciones de las que tienen, esperando que algunos cancelen). 

Si el hotel tiene un modelo de Machine Learning que le dice con precisión: *"Oye, esta reserva de esta familia tiene un 85% de probabilidad de cancelarse"*, el hotel puede:
*   Reconfirmar su reserva llamándoles por teléfono.
*   Ofrecer esa habitación a otra persona con antelación.
*   Ajustar el precio de la habitación en tiempo real.

En términos de Machine Learning, esto es un problema de **Clasificación Binaria**: la salida solo puede tener dos valores: `1` (el cliente cancela la reserva) o `0` (el cliente viene y hace el check-in).

---

## 2. El Pipeline de Datos: Preparando la Receta

Los algoritmos de Machine Learning son como procesadores de comida: si les metes basura, te devuelven basura (*Garbage In, Garbage Out*). Por eso, antes de entrenar modelos, construimos un **Pipeline** (una tubería o flujo automatizado) que limpia y prepara los datos de forma idéntica en el entrenamiento y en producción.

### ⚠️ Fuga de Datos (Target Leakage): El Error más común
Este es el concepto más importante de esta fase. Ocurre cuando entrenas a tu modelo con información "del futuro" que no tendrás en la vida real cuando alguien intente hacer una predicción.
*   *El sospechoso en nuestro dataset:* Las columnas `reservation_status` (que dice "Canceled", "Check-Out" o "No-Show") y `reservation_status_date`.
*   *¿Por qué es un error?* Si dejas esas columnas, el modelo aprenderá que si `reservation_status` es "Canceled", entonces `is_canceled` es `1`. ¡Obvio! Tendrás un 100% de precisión al entrenar, pero cuando un cliente nuevo reserve hoy en la web del hotel, **aún no sabrás su estado de reserva futuro**, por lo que tu modelo no sabrá qué hacer. **Eliminar estas columnas es obligatorio.**

### 🛠️ Los Pasos de Limpieza (Preprocesamiento)
Nuestros datos vienen con texto, números y valores vacíos (nulos). Los preparamos así:

1.  **Imputación de Nulos:** En la vida real hay datos que faltan (por ejemplo, clientes que no especifican su país). Para que el modelo no falle:
    *   Si la variable es **numérica** (como el precio `adr`), rellenamos el hueco con la **mediana** de todos los precios.
    *   Si es **categórica** (como el régimen de comidas `meal`), rellenamos el hueco con la etiqueta `"Unknown"` (Desconocido).
2.  **Escalado Estándar (StandardScaler):** Los algoritmos comparan números. Si tienes una variable como `lead_time` (días de anticipación, rango de 0 a 300) y otra como `adults` (rango de 1 a 4), el modelo podría pensar que la primera es mucho más importante solo porque sus números son más grandes. El escalador ajusta todos los números para que tengan media 0 y desviación estándar 1, poniéndolos a todos en la misma "escala de importancia".
3.  **Codificación One-Hot (OneHotEncoder):** Los modelos solo entienden de matemáticas y números, no de palabras como `"Resort Hotel"` o `"City Hotel"`. Con esta técnica, creamos columnas nuevas con 1 y 0 para cada opción. Si la columna `hotel` es `"Resort Hotel"`, se convierte en `hotel_Resort Hotel = 1` y `hotel_City Hotel = 0`.

---

## 3. El "Zoológico" de Modelos: ¿Quién es quién?

Entrenamos 5 modelos distintos para comparar su rendimiento. Aquí tienes la explicación de cómo funciona cada uno como si se lo explicaras a un amigo en una cafetería:

*   **Regresión Logística:** Es el modelo más básico y rápido. Intenta trazar una línea recta para separar los que cancelan de los que no. Como la salida debe ser una probabilidad (entre 0% y 100%), pasa el resultado por una curva matemática llamada **sigmoide** que aplasta los números entre 0 y 1. Es genial porque es muy rápido y te permite ver fácilmente qué variables influyen más.
*   **Árbol de Decisión:** Funciona como el juego de las 20 preguntas o un diagrama de flujo. Empieza arriba y pregunta: *¿El cliente viene de Portugal? Si es Sí, ve a la izquierda. ¿Reservó con más de 100 días de antelación? Si es Sí, ve a la derecha...* Al final de las ramas, llega a una decisión. Es muy fácil de entender visualmente, pero si lo dejamos crecer mucho se vuelve demasiado específico y memoriza los datos (sobreajuste o *overfitting*).
*   **Random Forest (Bosque Aleatorio):** Como un solo árbol de decisión puede equivocarse, plantamos un "bosque". Creamos cientos de árboles de decisión independientes, dándole a cada uno una parte aleatoria de los datos y de las variables. Cuando queremos predecir, les preguntamos a todos los árboles y tomamos la decisión de la mayoría (votación democrática). Es uno de los modelos más estables y fiables en ML.
*   **XGBoost (Gradient Boosting):** Es una evolución del Random Forest, pero en lugar de entrenar árboles independientes, los entrena secuencialmente (uno detrás de otro). El primer árbol hace sus predicciones, ve dónde se ha equivocado y el segundo árbol se entrena específicamente para corregir los errores del primero. Y así sucesivamente. Es extremadamente rápido y preciso, y suele ganar la mayoría de competiciones de Machine Learning con datos tabulares.
*   **Red Neuronal Multicapa (Keras/TensorFlow):** Se inspira de forma muy simplificada en el cerebro humano. Consiste en capas de "neuronas" artificiales interconectadas. Los datos entran por un extremo, se multiplican por ciertos "pesos" que se van ajustando con el entrenamiento (gracias a un algoritmo llamado retropropagación) y devuelven la probabilidad final por el otro. Son capaces de aprender patrones hiper-complejos y no lineales, pero requieren más datos para entrenar y son difíciles de interpretar por dentro (se consideran "cajas negras").

---

## 4. ¿Cómo medimos el éxito? Las Métricas de Evaluación

Para saber cuál es nuestro mejor modelo, no nos fijamos solo en cuántas veces acierta en general. Evaluamos con rigor:

### ❌ La Trampa de la Exactitud (Accuracy)
La exactitud nos dice simplemente qué porcentaje de predicciones fueron correctas. Parece la métrica ideal, pero tiene truco. Si en nuestro hotel el 90% de la gente viene y solo el 10% cancela, yo puedo crear un modelo tonto que prediga siempre: *"Nadie va a cancelar"*. 
Este modelo tonto tendrá un **90% de exactitud**, ¡pero es completamente inútil porque no detectará ni una sola cancelación! Por eso, la exactitud no nos sirve cuando las clases están desbalanceadas.

### 🏆 Nuestra Métrica Principal: F1-Score
Para evaluar de forma justa, cruzamos dos conceptos:
*   **Precisión (Precision):** De todas las reservas que el modelo dijo que se iban a cancelar, ¿cuántas se cancelaron de verdad? (Queremos una precisión alta para no molestar a clientes reales llamándoles para reconfirmar si no iban a cancelar).
*   **Sensibilidad (Recall):** De todas las cancelaciones reales que ocurrieron, ¿cuántas fue capaz de cazar el modelo? (Queremos un recall alto para enterarnos de todas las posibles cancelaciones).

El **F1-Score** es el promedio armónico entre la Precisión y la Sensibilidad. Al buscar un equilibrio perfecto entre ambas, es la métrica más honesta y segura para elegir el modelo ganador.

### 📈 Matriz de Confusión y Curva ROC-AUC
*   **Matriz de Confusión:** Es una tabla de 2x2 que te muestra de forma súper visual dónde se ha equivocado el modelo (cuántos falsos positivos y falsos negativos ha cometido).
*   **Curva ROC y su área bajo la curva (AUC):** Grafica la tasa de verdaderos positivos frente a la de falsos positivos a diferentes umbrales de decisión. Un modelo perfecto tiene un AUC de `1.0` (la curva va directa a la esquina superior izquierda), mientras que tirar una moneda al aire da un AUC de `0.5`. Nos ayuda a ver qué tan bueno es el modelo separando ambas clases.

---

## 5. Buenas Prácticas y el Principio KISS (*Keep It Simple, Stupid*)

En este proyecto hemos aplicado el principio **KISS** (Manténlo Sencillo). 
*   En lugar de crear una base de datos con sistemas de persistencia complejos, repositorios y modelos de bases de datos que añadirían sobreingeniería innecesaria, hemos diseñado la API de forma directa:
    *   Los datos se procesan en memoria a través de la tubería (`Pipeline`) de Scikit-learn.
    *   El modelo se carga directamente desde el archivo `.pkl` serializado en disco.
    *   La API expone los endpoints necesarios de manera limpia y modular, facilitando su comprensión para cualquiera que esté empezando en Machine Learning.