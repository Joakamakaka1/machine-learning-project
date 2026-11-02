"""Shared fixtures: synthetic bookings that never depend on the real CSV."""

from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest

BASE_BOOKING: dict[str, Any] = {
    "hotel": "City Hotel",
    "is_canceled": 0,
    "lead_time": 30,
    "arrival_date_year": 2016,
    "arrival_date_month": "March",
    "arrival_date_week_number": 11,
    "arrival_date_day_of_month": 15,
    "stays_in_weekend_nights": 1,
    "stays_in_week_nights": 2,
    "adults": 2,
    "children": 0.0,
    "babies": 0,
    "meal": "BB",
    "country": "PRT",
    "market_segment": "Online TA",
    "distribution_channel": "TA/TO",
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0,
    "reserved_room_type": "A",
    "assigned_room_type": "A",
    "booking_changes": 0,
    "deposit_type": "No Deposit",
    "agent": "9",
    "company": "40",
    "days_in_waiting_list": 0,
    "customer_type": "Transient",
    "adr": 100.0,
    "required_car_parking_spaces": 0,
    "total_of_special_requests": 0,
    "reservation_status": "Check-Out",
    "reservation_status_date": "2016-03-18",
}

RawFrameFactory = Callable[..., pd.DataFrame]


def _arrival_columns(day: str) -> dict[str, Any]:
    date = pd.Timestamp(day)
    return {
        "arrival_date_year": date.year,
        "arrival_date_month": date.month_name(),
        "arrival_date_week_number": date.isocalendar().week,
        "arrival_date_day_of_month": date.day,
    }


@pytest.fixture
def raw_frame() -> RawFrameFactory:
    """Build a raw-like frame with one row per dictionary of overrides.

    The `arrival` key expands to the arrival year, month, week and day columns.
    """

    def build(*overrides: dict[str, Any]) -> pd.DataFrame:
        rows = []
        for override in overrides:
            values = dict(override)
            arrival = values.pop("arrival", None)
            dates = {} if arrival is None else _arrival_columns(arrival)
            rows.append({**BASE_BOOKING, **dates, **values})
        return pd.DataFrame(rows)

    return build


@pytest.fixture
def raw_like_df(raw_frame: RawFrameFactory) -> pd.DataFrame:
    """Valid raw frame with the dirty rows that the cleaning must drop and the nulls it keeps."""
    return raw_frame(
        {"is_canceled": 1, "reservation_status": "Canceled"},
        {"hotel": "Resort Hotel", "lead_time": 10, "adr": 80.0},
        {"agent": None, "country": None, "company": None},
        {"children": None},
        {"adr": 0.0, "market_segment": "Complementary"},
        {"adr": 5399.99},
        {"adults": 0, "children": 0.0, "babies": 0},
        {"adults": 0, "children": None, "babies": 0},
        {"stays_in_week_nights": 0, "stays_in_weekend_nights": 0},
        {"adr": -6.38},
        {"adr": 5400.0},
    )


@pytest.fixture
def timeline_df(raw_frame: RawFrameFactory) -> pd.DataFrame:
    """Two bookings per month from July 2015 to August 2017, plus exact duplicates."""
    months = pd.date_range("2015-07-15", "2017-08-15", freq="MS") + pd.Timedelta(days=14)
    rows = [
        {"arrival": str(month.date()), "lead_time": lead_time}
        for month in months
        for lead_time in (5, 50)
    ]
    duplicates = [
        {"arrival": "2016-03-15", "lead_time": 7},
        {"arrival": "2016-03-15", "lead_time": 7},
        {"arrival": "2016-08-15", "lead_time": 8},
        {"arrival": "2016-08-15", "lead_time": 8},
    ]
    return raw_frame(*rows, *duplicates)
