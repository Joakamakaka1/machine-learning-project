from pathlib import Path

from pydantic_settings import BaseSettings

# Directorio raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    PROJECT_NAME: str = "Hotel Booking Cancellation Predictor API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # API Host/Port dynamically loaded from .env
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Ruta del modelo guardado para producción
    MODEL_PATH: Path = BASE_DIR / "models" / "best_model.pkl"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
