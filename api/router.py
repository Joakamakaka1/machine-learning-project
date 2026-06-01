import logging

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.config import settings
from api.schemas import BookingCreate, PredictionResult
from src.predictor import load_model

logger = logging.getLogger(__name__)
router = APIRouter()

# Cargar el modelo una sola vez al importar el módulo
try:
    _pipeline = load_model()
except Exception as e:
    _pipeline = None
    logger.warning("No se pudo cargar el modelo: %s", e)


@router.get("/health")
def health_check():
    """Endpoint de salud para comprobar que la API y el modelo estén activos."""
    return {
        "status": "OK",
        "project_name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "model_loaded": _pipeline is not None,
    }


@router.post("/predict", response_model=PredictionResult)
def predict_cancellation(booking: BookingCreate):
    """Endpoint para predecir si una reserva hotelera será cancelada."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")

    try:
        df_input = pd.DataFrame([booking.model_dump()])
        pred = _pipeline.predict(df_input)[0]
        prob = _pipeline.predict_proba(df_input)[:, 1][0]

        label = "Cancelado" if pred == 1 else "No Cancelado"
        message = f"Probabilidad de cancelacion del {prob:.2%}"

        return PredictionResult(prediction=pred, probability=prob, label=label, message=message)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error al realizar la inferencia sobre los datos de entrada: {str(e)}",
        )
