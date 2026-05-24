"""
Configuracion de la API: constantes del proyecto y variables de entorno.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

PROJECT_NAME = "Hotel Booking Cancellation Predictor API"
VERSION = "1.0.0"
API_V1_STR = "/api/v1"

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", 8000))
