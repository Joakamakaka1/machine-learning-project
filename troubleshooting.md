# 📋 Documento de Troubleshooting y Defensa del Modelo de ML

Este documento recopila las incidencias técnicas identificadas durante la fase de desarrollo y producción de los modelos de Machine Learning para la predicción de cancelaciones de reservas hoteleras, así como la justificación de la arquitectura y la selección del algoritmo final.

---

## 1. Problemas Técnicos Encontrados y Soluciones Aplicadas

### A. Fuga de Información (Target Leakage) en `reservation_status`
* **El Problema**: Al realizar el análisis exploratorio de datos (EDA), identificamos una correlación del 100% entre la variable `reservation_status` (que toma valores como "Check-Out", "Canceled" o "No-Show") y la variable objetivo `is_canceled`. 
  * Si la reserva tiene estado "Canceled" o "No-Show", `is_canceled` es `1`.
  * Si la reserva tiene estado "Check-Out", `is_canceled` es `0`.
* **El Riesgo**: Si dejamos esta columna, el modelo aprende una regla simple ("si es Check-Out predice 0, si es Canceled predice 1"). Tendrá un 100% de precisión en los datos de entrenamiento y prueba. Sin embargo, **en producción, cuando un cliente hace una reserva nueva, no sabemos si hará Check-Out o Cancelación en el futuro** (ese estado futuro es el que queremos predecir). Si mantuviéramos esta columna, la API fallaría catastróficamente al no disponer de ese dato.
* **La Solución**: Eliminamos de forma obligatoria las columnas `reservation_status` y `reservation_status_date` en la fase inicial de carga de datos (`src/config.py` y `src/data_loader.py`).

### B. Fallos de Tipos de Datos e Inferencia en Producción (Streamlit y API)
* **El Problema**: En producción, las entradas de usuario pueden venir en formatos mixtos o contener cadenas como `"NULL"`, `"None"` o texto vacío `""` en campos numéricos (por ejemplo, el ID de la agencia `agent` o cargos diarios `adr`). Scikit-Learn e imputadores numéricos fallaban arrojando excepciones del tipo: `ValueError: Cannot use median strategy with non-numeric data: could not convert string to float: 'NULL'`.
* **El Riesgo**: Caída del servidor de la API REST o cuelgue de la interfaz web interactiva de Streamlit ante entradas con formatos inesperados.
* **La Solución**: Implementamos una capa de preprocesamiento y limpieza robusta y automática dentro del wrapper de inferencia en `src/predictor.py`. El método `_clean_data` identifica todas las columnas numéricas que el modelo espera, convierte los nulos textuales a `NaN` reales de Python y coacciona de forma segura a tipo numérico usando `pd.to_numeric(..., errors='coerce')`. De esta manera, el imputador del Pipeline puede rellenar los valores nulos con la mediana de forma transparente y sin caídas.

### C. Fallos de Codificación Unicode en Consolas Windows (cp1252)
* **El Problema**: El pipeline principal fallaba al final de su ejecución en Windows debido a que la consola por defecto utiliza codificación `cp1252`, la cual es incapaz de renderizar ciertos caracteres Unicode y emojis (como el trofeo `🏆`).
* **La Solución**: Reemplazamos los emojis en los print de consola y usamos cadenas de texto seguras y compatibles con ASCII en `main.py` para garantizar la compatibilidad multiplataforma.

---

## 2. Justificación de la Métrica Principal de Evaluación: F1-Score

### ¿Por qué no usamos Accuracy (Exactitud)?
La Exactitud representa la proporción de predicciones correctas sobre el total. En datasets con cierto desbalanceo de clases (en este caso, ~37% de cancelaciones y ~63% de confirmaciones), la exactitud puede ser engañosa:
* Un clasificador "tonto" que siempre prediga que nadie cancelará la reserva obtendría un **63% de exactitud**. El modelo parecería útil, pero no detectaría ninguna cancelación, siendo inútil para el negocio.
* Para el hotel, el coste de los **Falsos Negativos** (predecir que un cliente vendrá y luego cancela, dejando la habitación vacía) y los **Falsos Positivos** (predecir que cancelará y aplicar overbooking o cancelarle la reserva, cuando en realidad venía) es muy alto.

### El poder del F1-Score
El **F1-Score** es el promedio armónico entre **Precisión** (de las que predigo que cancelan, cuántas cancelan realmente) y **Exhaustividad/Recall** (de todas las cancelaciones reales, cuántas detecto). Al equilibrar ambos errores, el F1-Score penaliza a los modelos sesgados y garantiza que estemos evaluando de forma honesta el rendimiento predictivo del negocio.

---

## 3. Defensa del Modelo Ganador: XGBoost

Evaluamos y entrenamos 5 algoritmos sobre el conjunto de test estratificado:

1. **Regresión Logística**: Modelo lineal de referencia. Rápido e interpretable, pero incapaz de capturar interacciones no lineales complejas entre variables. (F1: 81.47%).
2. **Árbol de Decisión**: Propenso a sobreajuste (*overfitting*), creando ramificaciones demasiado específicas para el conjunto de entrenamiento. (F1: 84.24%).
3. **Random Forest**: Robusto conjunto de árboles paralelos. Reduce el sobreajuste, pero a veces pierde capacidad predictiva al promediar. (F1: 79.85%).
4. **Red Neuronal (Keras)**: Muy potente, pero requiere mayor volumen de datos, gran afinamiento de hiperparámetros y complejidad añadida para serializar/cargar en producción. (F1: 85.45%).
5. **XGBoost (Extreme Gradient Boosting)**: Entrenamiento secuencial enfocado en corregir los errores de los árboles anteriores. (F1: 86.72%).

### ¿Por qué elegimos XGBoost?
* **Rendimiento Predictivo Máximo**: Logró el **F1-Score más alto (86.72%)** y un **ROC-AUC sobresaliente (0.9446)**.
* **Manejo de Datos Tabulares**: Las estructuras basadas en boosting de árboles son los campeones empíricos indiscutibles para datos estructurados/tabulares.
* **Simplicidad de Despliegue**: Se puede empaquetar de forma transparente en un único pipeline de Scikit-Learn `.pkl` y exportarlo a la API sin requerir el arranque de dependencias complejas de Deep Learning (como librerías pesadas de TensorFlow) ni archivos de arquitectura separados.
