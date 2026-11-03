"""Shared fixtures: synthetic bookings that never depend on the real CSV."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hotel_cancellation import data_loader

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


def synthetic_raw(rows_per_month: int = 120, seed: int = 0) -> pd.DataFrame:
    """Raw bookings from July 2015 to August 2017 with signal, in the shape of the real CSV.

    Cancellations depend on `lead_time`, `deposit_type` and `hotel`, which makes the expected
    behaviour of a trained model predictable. Forbidden columns copy the outcome, so that any leak
    shows up as a suspiciously good score.
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range("2015-07-01", "2017-08-01", freq="MS")
    arrivals = pd.DatetimeIndex(
        [
            month + pd.Timedelta(days=int(day))
            for month in months
            for day in rng.integers(0, 28, rows_per_month)
        ]
    )
    rows = len(arrivals)
    lead_time = rng.integers(0, 300, rows)
    hotel = rng.choice(["City Hotel", "Resort Hotel"], rows, p=[0.65, 0.35])
    deposit_type = rng.choice(
        ["No Deposit", "Non Refund", "Refundable"], rows, p=[0.85, 0.14, 0.01]
    )
    logit = (
        -1.5
        + 0.01 * lead_time
        + 3.0 * (deposit_type == "Non Refund")
        + 0.5 * (hotel == "City Hotel")
    )
    is_canceled = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    children = rng.integers(0, 2, rows).astype(float)
    children[rng.random(rows) < 0.01] = np.nan
    agent = rng.choice(["9", "240", "1", "14"], rows).astype(object)
    agent[rng.random(rows) < 0.1] = None
    return pd.DataFrame(
        {
            "hotel": hotel,
            "is_canceled": is_canceled,
            "lead_time": lead_time,
            "arrival_date_year": arrivals.year,
            "arrival_date_month": arrivals.month_name(),
            "arrival_date_week_number": arrivals.isocalendar().week.to_numpy(),
            "arrival_date_day_of_month": arrivals.day,
            "stays_in_weekend_nights": rng.integers(0, 3, rows),
            "stays_in_week_nights": rng.integers(1, 6, rows),
            "adults": rng.integers(1, 4, rows),
            "children": children,
            "babies": rng.integers(0, 2, rows),
            "meal": rng.choice(["BB", "HB", "SC"], rows),
            "country": rng.choice(["PRT", "GBR", "FRA"], rows),
            "market_segment": rng.choice(["Online TA", "Offline TA/TO", "Direct"], rows),
            "distribution_channel": rng.choice(["TA/TO", "Direct"], rows),
            "is_repeated_guest": rng.integers(0, 2, rows),
            "previous_cancellations": rng.integers(0, 2, rows),
            "previous_bookings_not_canceled": rng.integers(0, 2, rows),
            "reserved_room_type": rng.choice(["A", "D", "E"], rows),
            "assigned_room_type": rng.choice(["A", "D", "E"], rows),
            "booking_changes": is_canceled,
            "deposit_type": deposit_type,
            "agent": pd.Series(agent, dtype="str"),
            "company": pd.Series([None] * rows, dtype="str"),
            "days_in_waiting_list": rng.integers(0, 5, rows),
            "customer_type": rng.choice(["Transient", "Contract", "Group"], rows),
            "adr": rng.uniform(40, 250, rows).round(2),
            "required_car_parking_spaces": 1 - is_canceled,
            "total_of_special_requests": 1 - is_canceled,
            "reservation_status": np.where(is_canceled == 1, "Canceled", "Check-Out"),
            "reservation_status_date": (arrivals + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
        }
    )


@pytest.fixture(scope="session")
def synthetic_development() -> pd.DataFrame:
    """Clean development rows of `synthetic_raw`, in arrival order."""
    return data_loader.split_temporal(data_loader.clean(synthetic_raw())).development


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
