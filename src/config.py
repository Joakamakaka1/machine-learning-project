"""
- Archivo encargado de:
- Almacenar rutas
- Hiperparámetros del proyecto
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Cargar variables de entorno del archivo .env
load_dotenv()

# Directorios principales
BASE_DIR = Path(__file__).resolve().parent.parent

# Leer rutas de directorios desde el .env con fallback por defecto
env_data_dir = os.getenv("DATA_DIR", "data")
DATA_DIR = BASE_DIR / env_data_dir

# Si DATA_DIR en .env ya es "data/raw", el RAW_DATA_PATH se ajusta
if env_data_dir.endswith("raw"):
    RAW_DATA_PATH = DATA_DIR / "dataset_practica_final.csv"
    PROCESSED_DATA_DIR = DATA_DIR.parent / "processed"
else:
    RAW_DATA_PATH = DATA_DIR / "raw" / "dataset_practica_final.csv"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / os.getenv("MODEL_DIR", "models")
OUTPUTS_DIR = BASE_DIR / os.getenv("OUTPUT_DIR", "outputs")

# Crear carpetas si no existen
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)

# Variables a eliminar por fuga de datos o irrelevancia
COLS_TO_DROP = ["reservation_status", "reservation_status_date", "company"]
TARGET_COL = "is_canceled"
