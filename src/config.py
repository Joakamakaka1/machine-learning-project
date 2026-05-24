"""
- Archivo encargado de:
- Almacenar rutas
- Hiperparámetros del proyecto
"""

from pathlib import Path

# Directorios principales
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "dataset_practica_final.csv"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Crear carpetas si no existen
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)

# Variables a eliminar por fuga de datos o irrelevancia
COLS_TO_DROP = ["reservation_status", "reservation_status_date", "company"]
TARGET_COL = "is_canceled"

# Columnas numéricas del dataset — usadas en la capa de inferencia (predictor.py)
# Se usa para convertir de forma segura los valores nulos textuales ("NULL", "None") a NaN
NUM_COLS = [
    "lead_time",
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "agent",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]
