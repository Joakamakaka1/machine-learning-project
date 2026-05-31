# 🏨 Hotel Booking Cancellation — Machine Learning Project

Sistema predictivo de cancelaciones de reservas hoteleras (`is_canceled`), modelado como una tarea de **Clasificación Binaria** sobre el dataset estructurado `dataset_practica_final.csv`.

El sistema incluye un pipeline completo de Machine Learning (carga, preprocesamiento con pipelines de scikit-learn, ingeniería de features, entrenamiento de 5 modelos, evaluación avanzada y exportación automática), MLOps con **MLflow**, una **API REST modular** basada en FastAPI y una **interfaz gráfica interactiva** con Streamlit.

---

## 🛠️ Stack Tecnológico

| Capa                    | Herramientas Utilizadas                         |
| :---------------------- | :---------------------------------------------- |
| **Modelado y Pipeline** | Scikit-learn, XGBoost, TensorFlow / Keras       |
| **MLOps (Tracking)**    | MLflow                                          |
| **Interpretabilidad**   | SHAP (SHapley Additive exPlanations)            |
| **API Backend**         | FastAPI, Uvicorn, Pydantic v2                   |
| **Frontend UI**         | Streamlit                                       |
| **Linter / Formatter**  | Ruff                                            |
| **Pruebas (Testing)**   | Pytest                                          |
| **Gestión de Entorno**  | uv (reemplazo ultra-rápido de pip y virtualenv) |

---

## 📁 Estructura del Proyecto

```
machine-learning-project/
├── api/                 # API REST (FastAPI)
│   ├── config.py        # Variables de entorno y constantes (pydantic-settings)
│   ├── exceptions.py    # Excepciones personalizadas: APIException, ModelNotLoadedError, PredictionError
│   ├── router.py        # Endpoints /health y /predict
│   ├── schemas.py       # Modelos Pydantic: BookingCreate (28 campos) y PredictionResult
│   └── main.py          # Punto de entrada: FastAPI app + manejador global de excepciones
├── src/                 # Pipeline de Machine Learning
│   ├── config.py        # Rutas de directorios, listas de columnas y constantes de entrenamiento
│   ├── data_loader.py   # FeatureEngineer (transformer sklearn), load_data y preprocess_and_split
│   ├── model_trainer.py # KerasMLPClassifier (wrapper sklearn) y ModelTrainer (5 algoritmos)
│   ├── evaluator.py     # Métricas, matrices de confusión, curvas ROC/PR, SHAP summary y comparativa
│   └── predictor.py     # BookingPredictor: carga best_model.pkl y expone predict / predict_proba
├── utils/
│   └── tuning_utils.py  # _TuningProgressPipeline: log de progreso durante RandomizedSearchCV
├── notebooks/           # Notebooks Jupyter: EDA, entrenamiento, evaluación y predicción
├── tests/               # Pruebas automatizadas unitarias y de integración
├── data/
│   ├── raw/             # Dataset original CSV (gitignored)
│   └── processed/       # Splits train.csv / test.csv generados por main.py (gitignored)
├── models/              # Pipelines sklearn serializados en .pkl (uno por modelo + best_model.pkl)
├── outputs/             # Gráficos PNG, comparativa CSV y best_hiperparametros.json
├── app.py               # Interfaz Streamlit
├── main.py              # Orquestador principal del pipeline end-to-end
├── .env.example         # Plantilla de variables de configuración local
├── troubleshooting.md   # Registro de problemas encontrados y decisiones de diseño
└── pyproject.toml       # Dependencias, configuración de Ruff y Pytest
```

---

## 🔬 Pipeline de Machine Learning

El flujo completo se orquesta desde `main.py` y sigue estos pasos:

```
dataset_practica_final.csv
        │
        ▼
   load_data()               ← Normaliza tokens nulos (NULL, NA, NaN, "")
        │
        ▼
preprocess_and_split()       ← Elimina columnas con fuga de datos, divide train/test (80/20, stratify)
        │
        ▼
  Pipeline sklearn
  ┌────────────────────────────────────────────────────────────┐
  │  FeatureEngineer (transformer)                             │
  │    · total_nights, total_guests, adr_per_night             │
  │    · room_type_changed, previous_cancel_rate, is_new_guest │
  │                                                            │
  │  ColumnTransformer                                         │
  │    · Numéricas   → SimpleImputer(median) + StandardScaler  │
  │    · Categóricas → SimpleImputer(constant) + OHE           │
  └────────────────────────────────────────────────────────────┘
        │
        ▼
  ModelTrainer.train_all_models()   ← 5 pipelines independientes (clone del preprocessor)
        │
        ├── logistic_regression.pkl
        ├── decision_tree.pkl
        ├── random_forest.pkl
        ├── xgboost.pkl  ←── RandomizedSearchCV opcional (30 iter × 3-fold = 90 fits)
        └── neural_network.pkl  ←── KerasMLPClassifier (128→64→32, EarlyStopping)
        │
        ▼
  evaluate_model()   ← Accuracy, Precision, Recall, F1, ROC-AUC por modelo
        │
        ▼
  Selección por ROC-AUC → best_model.pkl
        │
        ▼
  MLflow tracking  ←── Parámetros, métricas y artefactos por ejecución
```

### Columnas eliminadas por fuga de datos

| Columna                   | Motivo                                                          |
| :------------------------ | :-------------------------------------------------------------- |
| `reservation_status`      | Codifica directamente si la reserva fue cancelada (target leak) |
| `reservation_status_date` | Fecha asociada a ese estado; también filtra el futuro           |
| `company`                 | ID numérico de alta cardinalidad sin valor predictivo relevante |

---

## 📊 Resultados Obtenidos

Mejor ejecución del pipeline completo con `tune_xgboost=True`:

| Modelo              |  Accuracy  |  F1-Score  |  ROC-AUC   |
| :------------------ | :--------: | :--------: | :--------: |
| Logistic Regression |   0.8365   |   0.8340   |   0.9160   |
| Decision Tree       |   0.8615   |   0.8602   |   0.9362   |
| Random Forest       |   0.8254   |   0.8132   |   0.9292   |
| **XGBoost**         | **0.8923** | **0.8929** | **0.9626** |
| Neural Network      |   0.8771   |   0.8772   |   0.9503   |

> Los valores completos de Precision y Recall por modelo se encuentran en `outputs/comparativa_modelos.csv`.

**Mejor modelo seleccionado:** XGBoost (ROC-AUC = 0.9626), guardado en `models/best_model.pkl`.

### Visualizaciones generadas en `outputs/`

| Archivo                         | Contenido                                            |
| :------------------------------ | :--------------------------------------------------- |
| `confusion_matrix_{modelo}.png` | Matriz de confusión normalizada por fila (5 modelos) |
| `roc_curve_comparison.png`      | Curvas ROC comparativas de los 5 modelos             |
| `precision_recall_curves.png`   | Curvas Precision-Recall comparativas                 |
| `comparativa_modelos_chart.png` | Barplot agrupado de todas las métricas               |
| `feature_importances.png`       | Top-15 variables más importantes (XGBoost)           |
| `shap_summary_xgboost.png`      | SHAP beeswarm plot (dirección e impacto por feature) |
| `comparativa_modelos.csv`       | Tabla resumen de métricas (todas las ejecuciones)    |
| `best_hiperparametros.json`     | Mejores hiperparámetros de XGBoost tras tuning       |

---

## ✅ Bonus Técnicos Implementados

| Bonus                              | Estado | Detalle                                                                                             |
| :--------------------------------- | :----: | :-------------------------------------------------------------------------------------------------- |
| 🌐 API con FastAPI                 |   ✅   | Endpoints `/api/v1/health` y `/api/v1/predict` en `api/`                                            |
| ⚙️ Optimización de hiperparámetros |   ✅   | `RandomizedSearchCV` (30 iter × 3-fold CV = 90 fits) sobre XGBoost                                  |
| 🧪 Registro de experimentos        |   ✅   | MLflow: parámetros, métricas y artefactos por ejecución                                             |
| 🖥️ Interfaz visual                 |   ✅   | Streamlit en `app.py`                                                                               |
| ⚖️ Balanceo de clases              |   ✅   | `class_weight="balanced"` en LR/DT/RF; `scale_pos_weight=1.7` en XGBoost; `class_weight_dict` en NN |
| 🔍 Interpretabilidad               |   ✅   | `feature_importances_` (XGBoost) + SHAP beeswarm via `TreeExplainer` / `LinearExplainer`            |

---

## 💡 Conceptos Clave de Machine Learning

### 1. Fuga de Datos (Target Leakage)

Ocurre cuando se introduce información del futuro en el conjunto de entrenamiento. En este dataset, `reservation_status` ("Canceled", "Check-Out", "No-Show") codifica directamente la variable objetivo. Si se mantiene, el modelo obtiene ~100% de acierto en entrenamiento pero falla completamente en producción (donde el estado final aún no ha ocurrido). **El pipeline elimina automáticamente `reservation_status`, `reservation_status_date` y `company` antes del modelado.**

### 2. Ingeniería de Features (`FeatureEngineer`)

El transformer `FeatureEngineer` genera 6 variables derivadas dentro del pipeline sklearn, garantizando que se calculen correctamente en cada fold de la validación cruzada:

| Feature derivada       | Descripción                                              |
| :--------------------- | :------------------------------------------------------- |
| `total_nights`         | Noches de semana + noches de fin de semana               |
| `total_guests`         | Adults + children + babies (mínimo 1)                    |
| `adr_per_night`        | Precio efectivo por noche (`adr / total_nights`)         |
| `room_type_changed`    | 1 si la habitación asignada difiere de la reservada      |
| `previous_cancel_rate` | Tasa histórica de cancelación del cliente                |
| `is_new_guest`         | 1 si el cliente no tiene historial previo de ningún tipo |

### 3. La Métrica Principal: ROC-AUC

La exactitud (_Accuracy_) en conjuntos desbalanceados (~37% cancelaciones) puede ser engañosa: un predictor que siempre responda «no cancela» obtiene un 63% de acierto sin detectar ninguna cancelación real.

Se usa **ROC-AUC** como criterio de selección porque es **independiente del umbral de decisión** y mide directamente la capacidad del modelo de ordenar las reservas por riesgo de cancelación. Es además comparable entre modelos con distribuciones de probabilidad muy distintas (Regresión Logística vs. Red Neuronal).

### 4. Desbalanceo de Clases

El dataset tiene ~63% reservas que no cancelan y ~37% que sí cancelan. Todos los modelos incorporan estrategias de compensación:

- **LR / DT / RF**: `class_weight="balanced"` — scikit-learn ajusta los pesos inversamente a la frecuencia de clase.
- **XGBoost**: `scale_pos_weight=1.7` (ratio aproximado de negativos/positivos).
- **Red Neuronal**: `class_weight` calculado con `compute_class_weight("balanced")` de sklearn, pasado al `.fit()` de Keras.

---

## 🌐 API REST (FastAPI)

La API expone el modelo ganador para integrarlo con sistemas externos.

### Endpoints

| Método | Ruta              | Descripción                                    |
| :----- | :---------------- | :--------------------------------------------- |
| `GET`  | `/api/v1/health`  | Comprueba que la API y el modelo estén activos |
| `POST` | `/api/v1/predict` | Predice si una reserva será cancelada          |

### Ejemplo de petición a `/api/v1/predict`

```json
{
  "hotel": "Resort Hotel",
  "lead_time": 342,
  "arrival_date_year": 2015,
  "arrival_date_month": "July",
  "arrival_date_week_number": 27,
  "arrival_date_day_of_month": 1,
  "stays_in_weekend_nights": 0,
  "stays_in_week_nights": 0,
  "adults": 2,
  "children": 0.0,
  "babies": 0,
  "meal": "BB",
  "country": "PRT",
  "market_segment": "Direct",
  "distribution_channel": "Direct",
  "is_repeated_guest": 0,
  "previous_cancellations": 0,
  "previous_bookings_not_canceled": 0,
  "reserved_room_type": "C",
  "assigned_room_type": "C",
  "booking_changes": 3,
  "deposit_type": "No Deposit",
  "agent": "NULL",
  "days_in_waiting_list": 0,
  "customer_type": "Transient",
  "adr": 0.0,
  "required_car_parking_spaces": 0,
  "total_of_special_requests": 0
}
```

### Ejemplo de respuesta

```json
{
  "prediction": 0,
  "probability": 0.12,
  "label": "No Cancelado",
  "message": "Probabilidad de cancelacion del 12.00%"
}
```

- **Swagger UI interactivo**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Errores manejados**: `503` si el modelo no está cargado, `400` si la inferencia falla.

---

## 📈 MLflow: MLOps y Registro de Experimentos

El pipeline está integrado con **MLflow** para registrar y auditar cada ejecución de entrenamiento.

- **Parámetros**: `nn_epochs`, `nn_batch_size`, filas de train/test y nombre del mejor modelo.
- **Métricas**: Accuracy, F1-Score y ROC-AUC sobre el conjunto de test para cada modelo.
- **Artefactos**: Tabla comparativa CSV, curva ROC conjunta y pipeline `best_model.pkl`.

```bash
# Iniciar el servidor MLflow (requiere haber ejecutado main.py al menos una vez)
uv run mlflow ui --port 5000
```

Panel interactivo: [http://localhost:5000](http://localhost:5000)

Las variables `MLFLOW_TRACKING_URI` y `MLFLOW_EXPERIMENT_NAME` se pueden configurar en `.env` (usar `.env.example` como plantilla).

---

## 🚀 Guía de Instalación y Configuración

El proyecto usa `uv` para garantizar la rapidez y reproducibilidad del entorno.

```bash
# 1. Copiar y ajustar el archivo de variables de entorno
cp .env.example .env

# 2. Instalar dependencias (crea el .venv automáticamente)
uv sync --extra dev --extra notebook
```

Coloca el dataset original en `data/raw/dataset_practica_final.csv` antes de ejecutar el pipeline.

---

## 💻 Comandos de Ejecución

### 1. Entrenar el Pipeline Completo

Entrena los 5 modelos, evalúa, genera todas las visualizaciones en `outputs/` y exporta el mejor a `models/best_model.pkl`:

```bash
uv run python main.py
```

> Para activar la búsqueda automática de hiperparámetros de XGBoost (30 iter × 3-fold = 90 fits secuenciales), cambia `tune_xgboost=False` a `tune_xgboost=True` en la llamada a `train_all_models` dentro de `main.py`. Los mejores parámetros se guardan en `outputs/best_hiperparametros.json`.

### 2. Iniciar la API REST (FastAPI)

```bash
uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Iniciar la Interfaz Gráfica (Streamlit)

```bash
uv run streamlit run app.py
```

Acceso web: [http://localhost:8501](http://localhost:8501)

### 4. Lanzar las Pruebas Automatizadas

```bash
uv run pytest
```

### 5. Formatear el código con Ruff

```bash
uvx ruff format
uvx ruff check --fix
```
