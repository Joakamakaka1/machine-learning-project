from pydantic import BaseModel, Field


class BookingCreate(BaseModel):
    hotel: str = Field(
        ..., example="Resort Hotel", description="Tipo de hotel (Resort Hotel o City Hotel)"
    )
    lead_time: int = Field(..., example=342, description="Días de anticipación de la reserva")
    arrival_date_year: int = Field(..., example=2015)
    arrival_date_month: str = Field(..., example="July")
    arrival_date_week_number: int = Field(..., example=27)
    arrival_date_day_of_month: int = Field(..., example=1)
    stays_in_weekend_nights: int = Field(..., example=0)
    stays_in_week_nights: int = Field(..., example=0)
    adults: int = Field(..., example=2)
    children: float = Field(..., example=0.0)
    babies: int = Field(..., example=0)
    meal: str = Field(..., example="BB")
    country: str = Field(..., example="PRT")
    market_segment: str = Field(..., example="Direct")
    distribution_channel: str = Field(..., example="Direct")
    is_repeated_guest: int = Field(..., example=0)
    previous_cancellations: int = Field(..., example=0)
    previous_bookings_not_canceled: int = Field(..., example=0)
    reserved_room_type: str = Field(..., example="C")
    assigned_room_type: str = Field(..., example="C")
    booking_changes: int = Field(..., example=3)
    deposit_type: str = Field(..., example="No Deposit")
    agent: str = Field(..., example="NULL")
    days_in_waiting_list: int = Field(..., example=0)
    customer_type: str = Field(..., example="Transient")
    adr: float = Field(..., example=0.0)
    required_car_parking_spaces: int = Field(..., example=0)
    total_of_special_requests: int = Field(..., example=0)


class PredictionResult(BaseModel):
    prediction: int = Field(..., description="Predicción binaria: 1 si cancela, 0 si no cancela")
    probability: float = Field(..., description="Probabilidad de cancelación entre 0.0 y 1.0")
    label: str = Field(
        ..., description="Etiqueta legible de la predicción ('Cancelado' o 'No Cancelado')"
    )
    message: str = Field(..., description="Mensaje explicativo de la predicción")
