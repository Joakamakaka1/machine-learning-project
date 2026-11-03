"""Request and response contracts of the API.

Columns and allowed values come from `schema.py`: nothing is repeated here. Validation is strict,
so a number written as text or a boolean used as a flag is an error, not a conversion.
"""

from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from hotel_cancellation import config
from hotel_cancellation.data import schema
from hotel_cancellation.predictor import Label, RiskBand

MAX_BATCH_SIZE = 1000

# Not a business rule: identifiers in the dataset have at most three digits.
_MAX_AGENT_LENGTH = 10

_RESTRICTED_CATEGORIES = tuple(
    column for column in schema.CATEGORICAL_FEATURES if column in schema.ALLOWED_VALUES
)


def _category(column: str, description: str) -> Any:
    allowed: list[JsonValue] = [*sorted(schema.ALLOWED_VALUES[column])]
    return Field(description=description, json_schema_extra={"enum": allowed})


def _count(description: str) -> Any:
    return Field(ge=0, le=schema.MAX_COUNT, description=description)


class Booking(BaseModel):
    """A booking at the moment of its confirmation, with the 19 input features."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        json_schema_extra={"example": dict(schema.EXAMPLE_BOOKING)},
    )

    hotel: str = _category("hotel", "Tipo de hotel.")
    lead_time: int = _count("Días entre la reserva y la llegada.")
    stays_in_weekend_nights: int = _count("Noches en fin de semana.")
    stays_in_week_nights: int = _count("Noches entre semana.")
    adults: int = _count("Adultos.")
    children: int = _count("Niños.")
    babies: int = _count("Bebés.")
    meal: str = _category("meal", "Régimen de comidas.")
    market_segment: str = _category("market_segment", "Segmento de mercado.")
    distribution_channel: str = _category("distribution_channel", "Canal de distribución.")
    is_repeated_guest: int = Field(ge=0, le=1, description="1 si es un cliente repetidor.")
    previous_cancellations: int = _count("Cancelaciones anteriores del cliente.")
    previous_bookings_not_canceled: int = _count("Reservas anteriores no canceladas.")
    reserved_room_type: str = _category("reserved_room_type", "Habitación reservada.")
    deposit_type: str = _category("deposit_type", "Tipo de depósito. No se debe leer como causa.")
    agent: str | None = Field(
        max_length=_MAX_AGENT_LENGTH,
        description="Identificador de la agencia, como texto. Nulo si no hay agencia.",
    )
    days_in_waiting_list: int = _count("Días en lista de espera.")
    customer_type: str = _category("customer_type", "Tipo de cliente.")
    adr: float = Field(
        ge=0,
        lt=config.ADR_UPPER_LIMIT,
        allow_inf_nan=False,
        description="Tarifa media diaria.",
    )

    @field_validator(*_RESTRICTED_CATEGORIES)
    @classmethod
    def _category_is_allowed(cls, value: str, info: ValidationInfo) -> str:
        allowed = schema.ALLOWED_VALUES[str(info.field_name)]
        if value not in allowed:
            raise ValueError(f"must be one of {sorted(allowed)}")
        return value

    @field_validator("agent")
    @classmethod
    def _agent_is_an_identifier(cls, value: str | None) -> str | None:
        if value is not None and not schema.ID_PATTERN.fullmatch(value):
            raise ValueError("must be an integer written as text")
        return value

    @model_validator(mode="after")
    def _booking_is_possible(self) -> Self:
        if self.adults + self.children + self.babies == 0:
            raise PydanticCustomError("no_guests", "a booking needs at least one guest")
        if self.stays_in_weekend_nights + self.stays_in_week_nights == 0:
            raise PydanticCustomError("no_nights", "a booking needs at least one night")
        return self


class BatchRequest(BaseModel):
    """Several bookings to predict together."""

    model_config = ConfigDict(strict=True, extra="forbid")

    bookings: list[Booking] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class PredictionResponse(BaseModel):
    """The outcome for one booking."""

    model_config = ConfigDict(protected_namespaces=())

    probability: float = Field(
        description=(
            "Probabilidad calibrada de cancelación. La calibración isotónica da valores en "
            "escalones, y algunos son exactamente 0 o 1."
        )
    )
    threshold: float = Field(description="Probabilidad a partir de la que se marca la reserva.")
    label: Label
    risk_band: RiskBand = Field(
        description=(
            "low: por debajo del umbral. medium: marcada, pero por debajo de 0.5. high: desde 0.5."
        )
    )
    model_version: str


class BatchResponse(BaseModel):
    """One prediction per booking, in the order received."""

    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    """State of the service."""

    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok"]
    model_loaded: bool
    model_version: str


class ModelSummary(BaseModel):
    """Identity of the served model."""

    name: str
    calibration: str
    version: str
    sha256: str
    created_at: str
    params: dict[str, Any]


class CostAssumptions(BaseModel):
    """Assumed costs behind the threshold. They are not data from the hotels."""

    are_assumptions: Literal[True]
    false_positive: float
    false_negative: float
    scenarios: list[dict[str, float]]


class ValidationSummary(BaseModel):
    """Mean and deviation across the validation windows of 2016. An undefined metric is null."""

    mean: dict[str, float | None]
    std: dict[str, float | None]


class TestSummary(BaseModel):
    """Saved result of the served model on 2017. The global ROC-AUC never comes alone."""

    __test__ = False

    executed_at: str
    subsets: list[dict[str, Any]]
    roc_auc_ci_95: list[float]


class ModelInfoResponse(BaseModel):
    """What identifies the served model and what is known about its performance."""

    model: ModelSummary
    threshold: float
    costs: CostAssumptions
    input_features: list[str]
    validation: ValidationSummary
    test: TestSummary | None


class ScenarioResponse(BaseModel):
    """The scenario with `total_of_special_requests`, a variable the served model does not use."""

    informative_only: Literal[True]
    rows: list[dict[str, Any]]


class EvaluationResponse(BaseModel):
    """The evaluation saved by the single execution, without any new computation."""

    record: dict[str, Any]
    threshold: float
    costs_are_assumptions: Literal[True]
    metrics: list[dict[str, Any]]
    segments: list[dict[str, Any]]
    bootstrap: list[dict[str, Any]]
    monthly: list[dict[str, Any]]
    psi: list[dict[str, Any]]
    special_requests_scenario: ScenarioResponse
