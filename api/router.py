import pandas as pd
from fastapi import APIRouter

from api.config import PROJECT_NAME, VERSION
from api.exceptions import ModelNotLoadedError, PredictionError
from api.schemas import BookingCreate, PredictionResult
from src.predictor import BookingPredictor

router = APIRouter()

# Instanciar el predictor una sola vez al cargar el modulo
try:
    _predictor = BookingPredictor()
except Exception as e:
    _predictor = None
    print(f"Advertencia: No se pudo cargar el modelo. {e}")


@router.get("/health")
def health_check():
    """Endpoint de salud para comprobar que la API y el modelo esten activos."""
    return {
        "status": "OK",
        "project_name": PROJECT_NAME,
        "version": VERSION,
        "model_loaded": _predictor is not None,
    }


@router.post("/predict", response_model=PredictionResult)
def predict_cancellation(booking: BookingCreate):
    """Endpoint para predecir si una reserva hotelera sera cancelada."""
    if _predictor is None:
        raise ModelNotLoadedError()

    try:
        df_input = pd.DataFrame([booking.model_dump()])
        pred = _predictor.predict(df_input)[0]
        prob = _predictor.predict_proba(df_input)[0]

        label = "Cancelado" if pred == 1 else "No Cancelado"
        message = f"Probabilidad de cancelacion del {prob:.2%}"

        return PredictionResult(prediction=pred, probability=prob, label=label, message=message)
    except Exception as e:
        raise PredictionError(
            detail=f"Error al realizar la inferencia sobre los datos de entrada: {str(e)}"
        )
