# 🏨 Hotel Booking Cancellation — Machine Learning Project

Este proyecto consiste en un sistema predictivo de cancelaciones de reservas hoteleras (`is_canceled`), modelando el problema como una tarea de **Clasificación Binaria** sobre el dataset estructurado `dataset_practica_final.csv`.

El sistema incluye un pipeline completo de Machine Learning (carga, preprocesamiento con pipelines de scikit-learn, entrenamiento de 5 modelos, evaluación avanzada y exportación automática), MLOps con **MLflow**, una **API REST modular** basada en FastAPI y una **interfaz gráfica interactiva** con Streamlit.

---

## 🛠️ Stack Tecnológico

| Capa | Herramientas Utilizadas |
| :--- | :--- |
| **Modelado y Pipeline** | Scikit-learn, XGBoost, TensorFlow / Keras |
| **MLOps (Tracking)** | MLflow |
| **API Backend** | FastAPI, Uvicorn, Pydantic v2 |
| **Frontend UI** | Streamlit |
| **Linter / Formatter** | Ruff |
| **Pruebas (Testing)** | Pytest |
| **Gestión de Entorno** | uv (reemplazo ultra-rápido de pip y virtualenv) |

---

## 📁 Estructura del Proyecto

```
machine-learning-project/
├── .github/workflows/   # CI/CD: GitHub Actions pipeline
├── api/                 # API REST Modular (FastAPI)
│   ├── config/          # Variables de entorno y configuraciones
│   ├── endpoints/v1/    # Endpoints organizados por versión (salud, predicción)
│   ├── exceptions/      # Excepciones personalizadas globales de la API
│   ├── schemas/         # Modelos de validación Pydantic para peticiones/respuestas
│   ├── services/        # Lógica de negocio (adaptación y consumo de modelos)
│   └── main.py          # Punto de entrada de la API REST
├── src/                 # Pipeline de Machine Learning
│   ├── config.py        # Configuraciones globales de carpetas y hiperparámetros
│   ├── data_loader.py   # Limpieza, división train/test y pipelines de preprocesamiento
│   ├── model_trainer.py # Entrenamiento de los 5 algoritmos y red de Keras
│   ├── evaluator.py     # Métricas avanzadas, matrices de confusión y curvas ROC
│   ├── predictor.py     # Clase de inferencia segura con coacción de tipos
│   └── theory.md        # Documentación teórica detallada
├── notebooks/           # Notebooks Jupyter interactivos e hiper-explicativos
├── tests/               # Pruebas automatizadas unitarias y de integración
├── data/
│   ├── raw/             # Dataset original en formato CSV (gitignored)
│   └── processed/       # Datasets limpios particionados train/test (gitignored)
├── models/              # Modelos serializados en formato .pkl y .keras (gitignored)
├── outputs/             # Plots generados y comparativa CSV (gitignored)
├── .env.example         # Plantilla de variables de configuración local
├── troubleshooting.md   # Registro de problemas de producción y defensa del modelo
└── pyproject.toml       # Definición de dependencias y configuraciones de Ruff y Pytest
```

---

## 💡 Conceptos Clave de Machine Learning

### 1. Fuga de Datos (Target Leakage)
Ocurre cuando introducimos información del futuro en el conjunto de entrenamiento. En este dataset, la columna `reservation_status` ("Canceled", "Check-Out", "No-Show") filtra directamente el target. Si se mantiene esta variable, el modelo tendrá un 100% de acierto en el entrenamiento, pero fallará catastróficamente al predecir reservas futuras de clientes reales (donde el estado final aún no ha ocurrido). **El pipeline elimina automáticamente esta variable antes del modelado.**

### 2. El dilema de la Métrica Principal: F1-Score vs Accuracy
La exactitud (*Accuracy*) mide el total de aciertos. No obstante, en conjuntos moderadamente desbalanceados, un predictor tonto que siempre diga "no cancela" puede obtener un ~63% de acierto sin aportar valor al negocio.
* Elegimos **F1-Score** porque combina armónicamente **Precision** (minimizar falsas alarmas de cancelación) y **Recall** (detectar la mayor cantidad de cancelaciones reales para evitar pérdidas). Es la métrica más honesta y robusta para este negocio hotelero.

---

## 📈 MLflow: MLOps y Registro de Experimentos

El pipeline está completamente integrado con **MLflow** para registrar y auditar cada ejecución de entrenamiento.
* **Parámetros**: Registra los hiperparámetros de entrenamiento de los 5 algoritmos.
* **Métricas**: Registra Accuracy, F1-Score y ROC-AUC sobre el conjunto de test para cada modelo.
* **Artefactos**: Sube la tabla comparativa final, la curva ROC conjunta y las matrices de confusión generadas.

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

### 2. Iniciar la API REST (FastAPI)
Expone el modelo ganador para integrarlo con sistemas externos:
```bash
uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
* **Swagger UI interactivo**: Consulta y haz pruebas HTTP directas en [http://localhost:8000/docs](http://localhost:8000/docs)
* **Endpoint de salud**: `/api/v1/health`
* **Endpoint de predicción**: `/api/v1/predict` (acepta un JSON con los datos de la reserva)

### 3. Iniciar la Interfaz Gráfica (Streamlit)
Ideal para probar interactivamente el modelo de forma visual:
```bash
uv run streamlit run app.py
```
* **Acceso web**: [http://localhost:8501](http://localhost:8501)

### 4. Lanzar las Pruebas Automatizadas
Ejecuta la suite de pruebas unitarias y de integración del sistema:
```bash
uv run pytest
```
