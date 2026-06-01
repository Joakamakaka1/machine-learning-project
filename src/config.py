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


# Función para crear directorios si no existen
def setup_dirs() -> None:
    """Crea los directorios necesarios si no existen. Llamar explícitamente desde main()."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# Constantes de entrenamiento y reproducibilidad
RANDOM_STATE = 42
TEST_SIZE = 0.2
NN_EPOCHS = 50
NN_BATCH_SIZE = 256

# Nombres de los modelos — fuente única para evitar duplicación en trainer y main
MODEL_NAMES = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
    "neural_network",
]

# Variables a eliminar por fuga de datos o irrelevancia
# reservation_status: etiqueta de cancelación codificada como texto (leakage directo).
# reservation_status_date: fecha de esa etiqueta (leakage directo).
# company: ID de empresa con alta cardinalidad y bajo poder predictivo.
# required_car_parking_spaces: proxy de check-in (leakage — cancelados tienen 0.0%).
# arrival_date_year: no generaliza a años futuros (solo contiene 2015-2017).
# assigned_room_type: se asigna al check-in, no al momento de reserva (leakage).
COLS_TO_DROP = [
    "reservation_status",
    "reservation_status_date",
    "company",
    "required_car_parking_spaces",
    "arrival_date_year",
    "assigned_room_type",
]

# Variable objetivo
TARGET_COL = "is_canceled"

# Tokens que el CSV codifica como valor ausente
NA_TOKENS = ["NULL", "NA", "NaN", ""]

# Columnas numéricas continuas/discretas.
NUM_COLS = [
    "lead_time",
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
    "total_of_special_requests",
    # Features derivadas por FeatureEngineer
    "total_nights",
    "total_guests",
    "adr_per_night",
    "previous_cancel_rate",
    "is_new_guest",
    "is_domestic",
    "is_agent_booking",
    "log_lead_time",
]

# Columnas categóricas → OneHotEncoder
CAT_COLS = [
    "hotel",
    "arrival_date_month",
    "meal",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
    "country",
    "agent",
    # Features derivadas categóricas por FeatureEngineer
    "guest_composition",
    "stay_category",
]
