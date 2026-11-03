"""Single source of truth for columns and allowed values.

Data loading, the API and the prediction form all read from this module. The allowed values were
extracted from the raw dataset and a slow test checks them against it. Rows of 2017 contribute no
value that is absent before 2017.
"""

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

TARGET = "is_canceled"

# Built from the arrival year, month and day only to split the data. It never reaches a model.
ARRIVAL_DATE = "arrival_date"

NUMERIC_FEATURES: tuple[str, ...] = (
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "days_in_waiting_list",
    "adr",
)

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "hotel",
    "meal",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
    "agent",
)

INPUT_FEATURES: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Not a business rule: it keeps every count convertible to a float. The largest in the dataset,
# `lead_time`, is far below it.
MAX_COUNT = 10_000

# A valid booking that the API documents and the prediction form opens with.
EXAMPLE_BOOKING: Mapping[str, Any] = MappingProxyType(
    {
        "hotel": "City Hotel",
        "lead_time": 45,
        "stays_in_weekend_nights": 1,
        "stays_in_week_nights": 2,
        "adults": 2,
        "children": 0,
        "babies": 0,
        "meal": "BB",
        "market_segment": "Online TA",
        "distribution_channel": "TA/TO",
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "reserved_room_type": "A",
        "deposit_type": "No Deposit",
        "agent": "9",
        "days_in_waiting_list": 0,
        "customer_type": "Transient",
        "adr": 110.0,
    }
)

# Forbidden in every model that can win. It only enters the informative scenario of the protocol.
SPECIAL_REQUESTS = "total_of_special_requests"

_CALENDAR_REASON = "Calendar variable: it does not generalize and it adds variance across windows."

FORBIDDEN_FEATURES: Mapping[str, str] = MappingProxyType(
    {
        "reservation_status": "Final outcome of the booking: it restates the target.",
        "reservation_status_date": "Date of the final status: only known after the outcome.",
        "country": "Documented leakage (Antonio et al. 2019): it is corrected at check-in.",
        "required_car_parking_spaces": "Only known at check-in or later.",
        "assigned_room_type": "Only known at check-in or later.",
        "booking_changes": "Accumulates over the life of the booking.",
        "total_of_special_requests": "Accumulates over the life of the booking.",
        "arrival_date_year": _CALENDAR_REASON,
        "arrival_date_month": _CALENDAR_REASON,
        "arrival_date_week_number": _CALENDAR_REASON,
        "arrival_date_day_of_month": _CALENDAR_REASON,
        "company": "No effect on validation metrics, so the simpler model keeps it out.",
    }
)

# Column order of the raw CSV.
RAW_COLUMNS: tuple[str, ...] = (
    "hotel",
    "is_canceled",
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "assigned_room_type",
    "booking_changes",
    "deposit_type",
    "agent",
    "company",
    "days_in_waiting_list",
    "customer_type",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "reservation_status",
    "reservation_status_date",
)

ARRIVAL_MONTHS: tuple[str, ...] = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

_ROOM_TYPES_RESERVED = frozenset({"A", "B", "C", "D", "E", "F", "G", "H", "L", "P"})

ALLOWED_VALUES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "hotel": frozenset({"City Hotel", "Resort Hotel"}),
        "meal": frozenset({"BB", "FB", "HB", "SC", "Undefined"}),
        "market_segment": frozenset(
            {
                "Aviation",
                "Complementary",
                "Corporate",
                "Direct",
                "Groups",
                "Offline TA/TO",
                "Online TA",
                "Undefined",
            }
        ),
        "distribution_channel": frozenset({"Corporate", "Direct", "GDS", "TA/TO", "Undefined"}),
        "reserved_room_type": _ROOM_TYPES_RESERVED,
        "assigned_room_type": _ROOM_TYPES_RESERVED | {"I", "K"},
        "deposit_type": frozenset({"No Deposit", "Non Refund", "Refundable"}),
        "customer_type": frozenset({"Contract", "Group", "Transient", "Transient-Party"}),
        "reservation_status": frozenset({"Canceled", "Check-Out", "No-Show"}),
        "arrival_date_month": frozenset(ARRIVAL_MONTHS),
    }
)

# Agent and company identifiers are free integers (334 and 353 distinct values, counting NULL).
ID_COLUMNS: tuple[str, ...] = ("agent", "company")
ID_PATTERN = re.compile(r"\d+")

NULLABLE_COLUMNS: frozenset[str] = frozenset({"children", "country", "agent", "company"})
BINARY_COLUMNS: tuple[str, ...] = ("is_canceled", "is_repeated_guest")
NON_NEGATIVE_COLUMNS: tuple[str, ...] = (
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "required_car_parking_spaces",
    "total_of_special_requests",
)
