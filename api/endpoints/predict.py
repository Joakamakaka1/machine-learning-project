from fastapi import APIRouter, HTTPException

from api.schemas.booking import BookingCreate, PredictionResult
from api.services.predictor import prediction_service

router = APIRouter()


@router.post("/predict", response_model=PredictionResult)
def predict_cancellation(booking: BookingCreate):
    """Endpoint para predecir si una reserva hotelera será cancelada."""
    if prediction_service is None:
        raise HTTPException(
            status_code=500,
            detail="El servicio de predicción no está disponible porque el modelo no se ha entrenado aún.",
        )

    try:
        # Extraemos los campos dict desde Pydantic y los pasamos al servicio
        booking_data = booking.model_dump()
        pred, prob = prediction_service.predict_booking(booking_data)

        label = "Cancelado" if pred == 1 else "No Cancelado"
        message = f"Probabilidad de cancelación del {prob:.2%}"

        return PredictionResult(prediction=pred, probability=prob, label=label, message=message)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error al realizar la inferencia sobre los datos de entrada: {str(e)}",
        )
