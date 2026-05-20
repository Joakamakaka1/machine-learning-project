# Hotel Booking Cancellation — ML Project

Predictor de cancelaciones de reservas hoteleras (`is_canceled`).
Clasificación binaria sobre el dataset `dataset_practica_final.csv`.

## Stack

| Capa        | Herramienta                             |
| ----------- | --------------------------------------- |
| Pipeline ML | scikit-learn, XGBoost, TensorFlow/Keras |
| Tracking    | MLflow                                  |
| API         | FastAPI + Uvicorn                       |
| UI (bonus)  | Streamlit                               |

## Setup

```bash
cp .env.example .env          # ajusta variables de entorno
uv sync                        # instala dependencias (crea .venv)
uv sync --extra dev            # + herramientas de desarrollo
uv sync --extra notebook       # + Jupyter
```

Coloca el dataset en `data/raw/dataset_practica_final.csv`.

## Estructura

```
machine-learning-project/
├── api/               # FastAPI: main, schemas, routers, dependencies
├── src/               # Pipeline ML: config, data, train, eval, predict
├── notebooks/         # EDA y experimentación
├── data/raw/          # Dataset original (gitignored)
├── models/            # Modelos serializados (gitignored)
├── outputs/           # Plots y reportes (gitignored)
├── .env.example       # Template de variables de entorno
└── pyproject.toml     # Dependencias y config del proyecto
```

## Ejecutar

```bash
# API
uv run uvicorn api.main:app --reload

# Streamlit (bonus)
uv run streamlit run app.py

# Tests
uv run pytest
```

## Roles del equipo

| Rol              | Responsabilidad                                             |
| ---------------- | ----------------------------------------------------------- |
| Data Engineer    | `src/data_loader.py` — carga, limpieza, feature engineering |
| ML Engineer      | `src/model_trainer.py` + `src/evaluator.py`                 |
| MLOps            | MLflow tracking + serialización                             |
| Backend          | `api/` — FastAPI + schemas                                  |
| Frontend (bonus) | `app.py` — Streamlit                                        |
