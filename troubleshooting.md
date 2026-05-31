# 📋 Documento de Troubleshooting y Defensa del Modelo de ML

Este documento recopila las incidencias técnicas identificadas durante la fase de desarrollo y producción de los modelos de Machine Learning para la predicción de cancelaciones de reservas hoteleras, así como la justificación de la arquitectura y la selección del algoritmo final.

---

## 1. Problemas Técnicos Encontrados y Soluciones Aplicadas

### A. Fuga de Información (Target Leakage) en `reservation_status`

- **El Problema**: Al realizar el análisis exploratorio de datos (EDA), identificamos una correlación del 100% entre la variable `reservation_status` (que toma valores como "Check-Out", "Canceled" o "No-Show") y la variable objetivo `is_canceled`.
  - Si la reserva tiene estado "Canceled" o "No-Show", `is_canceled` es `1`.
  - Si la reserva tiene estado "Check-Out", `is_canceled` es `0`.
- **El Riesgo**: Si dejamos esta columna, el modelo aprende una regla simple ("si es Check-Out predice 0, si es Canceled predice 1"). Tendrá un 100% de precisión en los datos de entrenamiento y prueba. Sin embargo, **en producción, cuando un cliente hace una reserva nueva, no sabemos si hará Check-Out o Cancelación en el futuro** (ese estado futuro es el que queremos predecir). Si mantuviéramos esta columna, la API fallaría catastróficamente al no disponer de ese dato.
- **La Solución**: Eliminamos de forma obligatoria las columnas `reservation_status` y `reservation_status_date` en la fase inicial de carga de datos (`src/config.py` y `src/data_loader.py`).

### B. Fallos de Tipos de Datos e Inferencia en Producción (Streamlit y API)

- **El Problema**: En producción, las entradas de usuario pueden venir en formatos mixtos o contener cadenas como `"NULL"`, `"None"` o texto vacío `""` en campos numéricos (por ejemplo, el ID de la agencia `agent` o cargos diarios `adr`). Scikit-Learn e imputadores numéricos fallaban arrojando excepciones del tipo: `ValueError: Cannot use median strategy with non-numeric data: could not convert string to float: 'NULL'`.
- **El Riesgo**: Caída del servidor de la API REST o cuelgue de la interfaz web interactiva de Streamlit ante entradas con formatos inesperados.
- **La Solución**: Implementamos una capa de preprocesamiento y limpieza robusta y automática dentro del wrapper de inferencia en `src/predictor.py`. El método `_clean_data` identifica todas las columnas numéricas que el modelo espera, convierte los nulos textuales a `NaN` reales de Python y coacciona de forma segura a tipo numérico usando `pd.to_numeric(..., errors='coerce')`. De esta manera, el imputador del Pipeline puede rellenar los valores nulos con la mediana de forma transparente y sin caídas.

### C. Fallos de Codificación Unicode en Consolas Windows (cp1252)

- **El Problema**: El pipeline principal fallaba al final de su ejecución en Windows debido a que la consola por defecto utiliza codificación `cp1252`, la cual es incapaz de renderizar ciertos caracteres Unicode y emojis (como el trofeo `🏆`).
- **La Solución**: Reemplazamos los emojis en los print de consola y usamos cadenas de texto seguras y compatibles con ASCII en `main.py` para garantizar la compatibilidad multiplataforma.

### D. Desbalanceo de Clases (~37% cancelaciones vs. ~63% no cancelaciones)

- **El Problema**: El dataset presenta un desbalanceo moderado (ratio ~1:1.7). Sin corrección, los modelos tienden a optimizar para la clase mayoritaria, penalizando insuficientemente los errores en la clase minoritaria (cancelaciones).
- **El Riesgo**: El modelo puede alcanzar buenas métricas globales ignorando la clase que realmente interesa al negocio: predecir qué reservas se van a cancelar.
- **La Solución**: Se aplican dos mecanismos complementarios según el tipo de modelo:
  - **XGBoost**: parámetro `scale_pos_weight=1.7` (ratio clases negativa/positiva), que pondera más fuertemente los errores sobre la clase minoritaria durante el boosting.
  - **Red Neuronal (Keras)**: `class_weight` calculado con `sklearn.utils.class_weight.compute_class_weight("balanced")` y pasado a `model.fit()`, consiguiendo el mismo efecto en el entrenamiento con backpropagation.
  - **Nota**: Estos parámetros afectan principalmente a F1-Score y Recall de la clase minoritaria. El impacto en ROC-AUC es marginal, ya que esta métrica es independiente del umbral de decisión.

---

## 2. Justificación de la Métrica Principal de Evaluación: ROC-AUC

### ¿Por qué no usamos Accuracy (Exactitud)?

La Exactitud representa la proporción de predicciones correctas sobre el total. En datasets con desbalanceo de clases (~37% de cancelaciones y ~63% de confirmaciones), la exactitud puede ser engañosa:

- Un clasificador "tonto" que siempre prediga que nadie cancelará la reserva obtendría un **63% de exactitud**. El modelo parecería útil, pero no detectaría ninguna cancelación, siendo inútil para el negocio.

### ¿Por qué no usamos F1-Score como métrica de selección?

El F1-Score es una métrica útil para informar, pero **depende del umbral de decisión** (por defecto 0.5). Esto lo hace sensible a cómo cada algoritmo calibra sus probabilidades, lo que introduce una fuente de variación que no refleja necesariamente capacidad predictiva real al comparar modelos de familias muy distintas.

### El poder del ROC-AUC

El área bajo la curva ROC (**ROC-AUC**) mide la capacidad del modelo de ordenar correctamente una reserva que cancela por encima de una que no cancela, **para cualquier umbral posible**. Sus ventajas en este contexto son:

- **Independiente del umbral**: No asume un punto de corte específico, por lo que compara todos los modelos en igualdad de condiciones.
- **Robusto al desbalance moderado**: A diferencia de Accuracy o F1, no se ve artificialmente inflado cuando una clase domina.
- **Accionable para el negocio hotelero**: El hotel puede ajustar el umbral de alerta según su capacidad de gestión; ROC-AUC mide la calidad del ranking independientemente de ese ajuste.

**Todas las métricas** (Accuracy, F1-Score, Precision, Recall, ROC-AUC, PR-AUC) se calculan y guardan en la tabla comparativa. ROC-AUC se usa únicamente para la selección automática del mejor modelo.

---

## 3. Defensa del Modelo Ganador

Evaluamos y entrenamos 5 algoritmos sobre el conjunto de test estratificado. Todos — incluida la red neuronal — se encapsulan en un **Pipeline sklearn** (preprocesado + clasificador), lo que garantiza:

- Ausencia de fuga de datos entre entrenamiento y evaluación.
- Serialización uniforme en un único `.pkl` por modelo.
- Competencia en **igualdad de condiciones**: la red neuronal se selecciona o descarta exactamente igual que los modelos clásicos.

1. **Regresión Logística**: Modelo lineal de referencia. Rápido e interpretable, pero incapaz de capturar interacciones no lineales complejas entre variables.
2. **Árbol de Decisión**: Propenso a sobreajuste (_overfitting_), creando ramificaciones demasiado específicas para el conjunto de entrenamiento.
3. **Random Forest**: Robusto conjunto de árboles paralelos. Reduce el sobreajuste, pero a veces pierde capacidad predictiva al promediar.
4. **XGBoost**: Entrenamiento secuencial enfocado en corregir los errores de los árboles anteriores. Campeón empírico en datos tabulares estructurados. Incluye `scale_pos_weight=1.7` para compensar el desbalanceo de clases y soporte para **RandomizedSearchCV** (20 iteraciones, 3-fold CV sobre ROC-AUC) activable mediante `tune_xgboost=True`. Los mejores hiperparámetros encontrados se guardan en `outputs/best_hiperparametros.json`.
5. **Red Neuronal (Keras)**: Arquitectura MLP (64 → 32 → 16 → 1) encapsulada en el wrapper `KerasMLPClassifier`. Compite directamente con el resto gracias a la integración en el Pipeline sklearn. Aplica `class_weight="balanced"` internamente para compensar el desbalanceo de clases.

### Criterio de selección automática

El pipeline selecciona el modelo con el **mayor ROC-AUC** sobre el conjunto de test y lo exporta a `models/best_model.pkl`. Si en una ejecución concreta la red neuronal supera a XGBoost en ROC-AUC, la red neuronal será el modelo desplegado en la API.
