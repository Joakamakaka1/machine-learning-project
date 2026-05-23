from fastapi import APIRouter

from api.config import settings
from api.services.predictor import prediction_service

router = APIRouter()


@router.get("/health")
def health_check():
    """Endpoint de salud para comprobar que la API y el modelo estén activos."""
    return {
        "status": "OK",
        "project_name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "model_loaded": prediction_service is not None,
    }
