"""Fields of the prediction form.

This module separates what the model asks for from how the form draws it. Allowed values and the
example booking come from `schema.py`, so the form opens with a valid booking ready to send.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from hotel_cancellation import config
from hotel_cancellation.data import schema

_ADR_LARGEST_INPUT = config.ADR_UPPER_LIMIT - 0.01
_NO_AGENT = ""


class Kind(StrEnum):
    """How a field is drawn and sent."""

    CATEGORY = "category"
    COUNT = "count"
    FLAG = "flag"
    MONEY = "money"
    AGENT = "agent"


@dataclass(frozen=True)
class FormField:
    """One input of the form. `name` is the exact name the API expects."""

    name: str
    label: str
    kind: Kind
    help_text: str = ""


FORM_SECTIONS: Mapping[str, tuple[FormField, ...]] = {
    "Reserva": (
        FormField("hotel", "Tipo de hotel", Kind.CATEGORY, "Hotel urbano (City) o vacacional."),
        FormField(
            "lead_time",
            "Antelación (días)",
            Kind.COUNT,
            "Días entre la fecha de reserva y la de llegada.",
        ),
        FormField(
            "deposit_type",
            "Tipo de depósito",
            Kind.CATEGORY,
            "No Deposit: sin depósito. Non Refund: no reembolsable. Refundable: reembolsable. "
            "No se debe leer como causa de la cancelación.",
        ),
        FormField(
            "reserved_room_type",
            "Habitación reservada",
            Kind.CATEGORY,
            "Tipo de habitación, con un código anonimizado.",
        ),
        FormField("adr", "Tarifa media diaria (ADR)", Kind.MONEY, "Precio medio por noche."),
    ),
    "Estancia y ocupantes": (
        FormField("stays_in_weekend_nights", "Noches de fin de semana", Kind.COUNT),
        FormField("stays_in_week_nights", "Noches entre semana", Kind.COUNT),
        FormField("adults", "Adultos", Kind.COUNT),
        FormField("children", "Niños", Kind.COUNT),
        FormField("babies", "Bebés", Kind.COUNT),
        FormField(
            "meal",
            "Régimen de comidas",
            Kind.CATEGORY,
            "BB: desayuno. HB: media pensión. FB: pensión completa. SC: sin comidas.",
        ),
    ),
    "Cliente y canal": (
        FormField(
            "market_segment",
            "Segmento de mercado",
            Kind.CATEGORY,
            "TA: agencia de viajes. TO: turoperador.",
        ),
        FormField("distribution_channel", "Canal de distribución", Kind.CATEGORY),
        FormField(
            "customer_type",
            "Tipo de cliente",
            Kind.CATEGORY,
            "Transient: individual. Contract: con contrato. Group: en grupo. "
            "Transient-Party: individual ligada a otra reserva.",
        ),
        FormField(
            "agent",
            "Agencia (identificador)",
            Kind.AGENT,
            "Número de la agencia. Déjalo vacío si la reserva no pasa por una agencia.",
        ),
    ),
    "Historial": (
        FormField("is_repeated_guest", "¿Cliente repetidor?", Kind.FLAG),
        FormField("previous_cancellations", "Cancelaciones anteriores", Kind.COUNT),
        FormField("previous_bookings_not_canceled", "Reservas anteriores sin cancelar", Kind.COUNT),
        FormField(
            "days_in_waiting_list",
            "Días en lista de espera",
            Kind.COUNT,
        ),
    ),
}


def all_fields() -> list[FormField]:
    """List the fields in the order of the form."""
    return [field for fields in FORM_SECTIONS.values() for field in fields]


def options_of(field: FormField) -> list[str]:
    """Give the allowed values of a category."""
    return sorted(schema.ALLOWED_VALUES[field.name])


def limits_of(field: FormField) -> tuple[float, float]:
    """Give the lowest and highest values that the form lets through."""
    if field.kind is Kind.MONEY:
        return 0.0, _ADR_LARGEST_INPUT
    return 0, schema.MAX_COUNT


def default_of(field: FormField) -> Any:
    """Give the value of the example booking, as the form draws it."""
    value = schema.EXAMPLE_BOOKING[field.name]
    if field.kind is Kind.AGENT:
        return _NO_AGENT if value is None else value
    return value


def build_payload(values: Mapping[str, Any]) -> dict[str, Any]:
    """Build the body of the request, with each value in the type the API demands.

    Args:
        values: What the form holds, by field name.

    Returns:
        A booking. An empty agent means that the booking has no agency.
    """
    payload: dict[str, Any] = {}
    for field in all_fields():
        value = values[field.name]
        if field.kind in (Kind.COUNT, Kind.FLAG):
            payload[field.name] = int(value)
        elif field.kind is Kind.MONEY:
            payload[field.name] = float(value)
        elif field.kind is Kind.AGENT:
            text = "" if value is None else str(value).strip()
            payload[field.name] = None if text == _NO_AGENT else text
        else:
            payload[field.name] = str(value)
    return payload
