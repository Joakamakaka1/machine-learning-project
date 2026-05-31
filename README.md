# 🏨 Hotel Booking Cancellation — Machine Learning Project

Este proyecto consiste en un sistema predictivo de cancelaciones de reservas hoteleras (`is_canceled`), modelando el problema como una tarea de **Clasificación Binaria** sobre el dataset estructurado `dataset_practica_final.csv`.

El sistema incluye un pipeline completo de Machine Learning (carga, preprocesamiento con pipelines de scikit-learn, entrenamiento de 5 modelos, evaluación avanzada y exportación automática), MLOps con **MLflow**, una **API REST modular** basada en FastAPI y una **interfaz gráfica interactiva** con Streamlit.

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
├── .github/workflows/   # CI/CD: GitHub Actions pipeline
├── api/                 # API REST (FastAPI)
│   ├── config.py        # Variables de entorno y constantes
│   ├── exceptions.py    # Excepciones personalizadas de la API
│   ├── router.py        # Endpoints de salud y predicción
│   ├── schemas.py       # Modelos Pydantic para peticiones y respuestas
│   └── main.py          # Punto de entrada de la API REST
├── src/                 # Pipeline de Machine Learning
│   ├── config.py        # Configuraciones globales de carpetas y listas de columnas
│   ├── data_loader.py   # FeatureEngineer transformer, carga y preprocesamiento con Pipeline
│   ├── model_trainer.py # 5 algoritmos (LR, DT, RF, XGBoost, NN) y búsqueda de hiperparámetros
│   ├── evaluator.py     # Métricas, matrices de confusión, curvas ROC y SHAP summary
│   ├── predictor.py     # Clase de inferencia para la API
│   └── tuning_utils.py  # _TuningProgressPipeline: log de progreso durante RandomizedSearchCV
├── notebooks/           # Notebooks Jupyter interactivos e hiper-explicativos
├── tests/               # Pruebas automatizadas unitarias y de integración
├── data/
│   ├── raw/             # Dataset original en formato CSV (gitignored)
│   └── processed/       # Datasets limpios particionados train/test (gitignored)
├── models/              # Modelos serializados en formato .pkl
├── outputs/             # Plots generados, comparativa CSV y best_hiperparametros.json (gitignored)
├── .env.example         # Plantilla de variables de configuración local
├── troubleshooting.md   # Registro de problemas de producción y defensa del modelo
└── pyproject.toml       # Definición de dependencias y configuraciones de Ruff y Pytest
```

---

## � Resultados Obtenidos

Última ejecución del pipeline completo con `tune_xgboost=True`:

| Modelo              |  Accuracy  |  F1-Score  |  ROC-AUC   |
| :------------------ | :--------: | :--------: | :--------: |
| Logistic Regression |   0.8365   |   0.8340   |   0.9160   |
| Decision Tree       |   0.8615   |   0.8602   |   0.9362   |
| Random Forest       |   0.8254   |   0.8132   |   0.9292   |
| **XGBoost**         | **0.8923** | **0.8929** | **0.9626** |
| Neural Network      |   0.8771   |   0.8772   |   0.9503   |

**Mejor modelo seleccionado:** XGBoost (ROC-AUC = 0.9626), guardado en `models/best_model.pkl`.

---

## ✅ Bonus Técnicos Implementados

| Bonus                              | Estado | Detalle                                                                                             |
| :--------------------------------- | :----: | :-------------------------------------------------------------------------------------------------- |
| 🌐 API con FastAPI                 |   ✅   | Endpoints `/health` y `/predict` en `api/`                                                          |
| ⚙️ Optimización de hiperparámetros |   ✅   | `RandomizedSearchCV` (30 iter × 3-fold CV = 90 fits) en XGBoost                                     |
| 🧪 Registro de experimentos        |   ✅   | MLflow: parámetros, métricas y artefactos por ejecución                                             |
| 🖥️ Interfaz visual                 |   ✅   | Streamlit en `app.py`                                                                               |
| ⚖️ Balanceo de clases              |   ✅   | `class_weight="balanced"` en LR/DT/RF; `scale_pos_weight=1.7` en XGBoost; `class_weight_dict` en NN |
| 🧠 Embeddings personalizados       |   ✅   | `TargetEncoder` para `country` y `agent` (alta cardinalidad)                                        |
| 🔍 Interpretabilidad               |   ✅   | `feature_importances_` + SHAP beeswarm plot (`outputs/shap_summary_xgboost.png`)                    |

---

## �💡 Conceptos Clave de Machine Learning

### 1. Fuga de Datos (Target Leakage)

Ocurre cuando introducimos información del futuro en el conjunto de entrenamiento. En este dataset, la columna `reservation_status` ("Canceled", "Check-Out", "No-Show") filtra directamente el target. Si se mantiene esta variable, el modelo tendrá un 100% de acierto en el entrenamiento, pero fallará catastróficamente al predecir reservas futuras de clientes reales (donde el estado final aún no ha ocurrido). **El pipeline elimina automáticamente esta variable antes del modelado.**

### 2. La Métrica Principal: ROC-AUC

La exactitud (_Accuracy_) mide el total de aciertos, pero en conjuntos desbalanceados (~37% cancelaciones) un predictor que siempre diga «no cancela» obtiene un 63% de acierto sin detectar ninguna cancelación real.

- Elegimos **ROC-AUC** como criterio de selección del mejor modelo porque es **independiente del umbral de decisión** y mide directamente la capacidad del modelo de ordenar las reservas por riesgo. Un modelo con ROC-AUC = 1 distingue perfectamente cancelaciones de no-cancelaciones independientemente del punto de corte elegido. Esto es especialmente importante en desbalanceo moderado porque no asume ningún umbral concreto y es comparable entre modelos con distribuciones de probabilidad muy distintas (p. ej. Regresión Logística vs. Red Neuronal).

---

## 📈 MLflow: MLOps y Registro de Experimentos

El pipeline está completamente integrado con **MLflow** para registrar y auditar cada ejecución de entrenamiento.

- **Parámetros**: Registra los hiperparámetros de entrenamiento de los 5 algoritmos.
- **Métricas**: Registra Accuracy, F1-Score y ROC-AUC sobre el conjunto de test para cada modelo.
- **Artefactos**: Sube la tabla comparativa final, la curva ROC conjunta y las matrices de confusión generadas.

Para iniciar el servidor de MLflow localmente:

```bash
uv run mlflow ui --port 5000
```

Entra en su panel interactivo mediante: [http://localhost:5000](http://localhost:5000)

---

## 🚀 Guía de Instalación y Configuración

El proyecto utiliza `uv` para garantizar la rapidez y reproducibilidad del entorno de desarrollo.

```bash
# 1. Copiar y ajustar el archivo de variables de entorno
cp .env.example .env

# 2. Instalar dependencias del proyecto (crea el .venv automáticamente)
uv sync --extra dev --extra notebook
```

Coloca el dataset original en `data/raw/dataset_practica_final.csv`.

---

## 💻 Comandos de Ejecución

### 1. Entrenar el Pipeline Completo

Entrena los 5 modelos, los evalúa, genera visualizaciones en `outputs/` y exporta el mejor a `models/best_model.pkl`:

```bash
uv run python main.py
```

Por defecto, XGBoost se entrena con los hiperparámetros fijados en el código. Para activar la búsqueda automática de hiperparámetros mediante **RandomizedSearchCV** (30 iteraciones × 3-fold CV = 90 fits, ejecución secuencial con `n_jobs=1` para evitar oversubscription en Windows), establece `tune_xgboost=True` en la llamada a `train_all_models` dentro de `main.py`. Los mejores parámetros se guardan automáticamente en `outputs/best_hiperparametros.json`.

### 2. Iniciar la API REST (FastAPI)

Expone el modelo ganador para integrarlo con sistemas externos:

```bash
uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

- **Swagger UI interactivo**: Consulta y haz pruebas HTTP directas en [http://localhost:8000/docs](http://localhost:8000/docs)
- **Endpoint de salud**: `/api/v1/health`
- **Endpoint de predicción**: `/api/v1/predict` (acepta un JSON con los datos de la reserva)

### 3. Iniciar la Interfaz Gráfica (Streamlit)

Ideal para probar interactivamente el modelo de forma visual:

```bash
uv run streamlit run app.py
```

- **Acceso web**: [http://localhost:8501](http://localhost:8501)

### 4. Lanzar las Pruebas Automatizadas

Ejecuta la suite de pruebas unitarias y de integración del sistema:

```bash
uv run pytest
```
