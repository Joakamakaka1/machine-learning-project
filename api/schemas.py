from pydantic import BaseModel, Field


class BookingCreate(BaseModel):
    hotel: str = Field(
        ...,
        description="Tipo de hotel (Resort Hotel o City Hotel)",
        json_schema_extra={"example": "Resort Hotel"},
    )
    lead_time: int = Field(
        ..., description="Dias de anticipacion de la reserva", json_schema_extra={"example": 342}
    )
    arrival_date_year: int = Field(..., json_schema_extra={"example": 2015})
    arrival_date_month: str = Field(..., json_schema_extra={"example": "July"})
    arrival_date_week_number: int = Field(..., json_schema_extra={"example": 27})
    arrival_date_day_of_month: int = Field(..., json_schema_extra={"example": 1})
    stays_in_weekend_nights: int = Field(..., json_schema_extra={"example": 0})
    stays_in_week_nights: int = Field(..., json_schema_extra={"example": 0})
    adults: int = Field(..., json_schema_extra={"example": 2})
    children: float = Field(..., json_schema_extra={"example": 0.0})
    babies: int = Field(..., json_schema_extra={"example": 0})
    meal: str = Field(..., json_schema_extra={"example": "BB"})
    country: str = Field(..., json_schema_extra={"example": "PRT"})
    market_segment: str = Field(..., json_schema_extra={"example": "Direct"})
    distribution_channel: str = Field(..., json_schema_extra={"example": "Direct"})
    is_repeated_guest: int = Field(..., json_schema_extra={"example": 0})
    previous_cancellations: int = Field(..., json_schema_extra={"example": 0})
    previous_bookings_not_canceled: int = Field(..., json_schema_extra={"example": 0})
    reserved_room_type: str = Field(..., json_schema_extra={"example": "C"})
    assigned_room_type: str = Field(..., json_schema_extra={"example": "C"})
    booking_changes: int = Field(..., json_schema_extra={"example": 3})
    deposit_type: str = Field(..., json_schema_extra={"example": "No Deposit"})
    agent: str = Field(..., json_schema_extra={"example": "NULL"})
    days_in_waiting_list: int = Field(..., json_schema_extra={"example": 0})
    customer_type: str = Field(..., json_schema_extra={"example": "Transient"})
    adr: float = Field(..., json_schema_extra={"example": 0.0})
    required_car_parking_spaces: int = Field(..., json_schema_extra={"example": 0})
    total_of_special_requests: int = Field(..., json_schema_extra={"example": 0})


class PredictionResult(BaseModel):
    prediction: int = Field(..., description="Prediccion binaria: 1 si cancela, 0 si no cancela")
    probability: float = Field(..., description="Probabilidad de cancelacion entre 0.0 y 1.0")
    label: str = Field(
        ..., description="Etiqueta legible de la prediccion ('Cancelado' o 'No Cancelado')"
    )
    message: str = Field(..., description="Mensaje explicativo de la prediccion")
