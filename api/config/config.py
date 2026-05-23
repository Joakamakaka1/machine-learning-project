from pathlib import Path

from pydantic_settings import BaseSettings

# Directorio raíz del proyecto (tres niveles hacia arriba desde api/config/config.py)
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    PROJECT_NAME: str = "Hotel Booking Cancellation Predictor API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # API Host/Port cargados dinámicamente desde el .env
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Ruta del modelo guardado para producción
    MODEL_PATH: Path = BASE_DIR / "models" / "best_model.pkl"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
