"""
Configuracion de la API: constantes del proyecto y variables de entorno.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    PROJECT_NAME: str = "Hotel Booking Cancellation Predictor API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    @property
    def MODEL_PATH(self) -> Path:
        return Path(__file__).resolve().parent.parent / "models" / "best_model.pkl"


settings = Settings()

# Aliases de modulo para compatibilidad con el resto del codigo
PROJECT_NAME = settings.PROJECT_NAME
VERSION = settings.VERSION
API_V1_STR = settings.API_V1_STR
MODEL_PATH = settings.MODEL_PATH
API_HOST = settings.API_HOST
API_PORT = settings.API_PORT
