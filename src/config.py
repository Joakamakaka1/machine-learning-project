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
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Variables a eliminar por fuga de datos o irrelevancia
# Reservation_status es la etiqueta de cancelación codificada como texto.
# El modelo podría aprender a identificar literalmente.
# Reservation_status_date es la fecha de esa etiqueta, lo que también causaría fuga de datos.
# Company es un ID de empresa que hizo la reserva.
# No aporta información útil y podría causar sobreajuste.
COLS_TO_DROP = ["reservation_status", "reservation_status_date", "company"]

# Variable objetivo
TARGET_COL = "is_canceled"

# Tokens que el CSV codifica como valor ausente
NA_TOKENS = ["NULL", "NA", "NaN", ""]

# Columnas numéricas continuas/discretas.
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
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    # Features derivadas por engineer_features()
    "total_nights",
    "total_guests",
    "adr_per_night",
    "room_type_changed",
    "previous_cancel_rate",
    "is_new_guest",
]

# Columnas categóricas → OneHotEncoder
CAT_COLS = [
    "hotel",
    "arrival_date_month",
    "meal",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "customer_type",
    "country",
    "agent",
]

