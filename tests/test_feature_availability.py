"""Tests of the accumulation diagnostic on synthetic bookings with a known behaviour."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from experiments import feature_availability
from hotel_cancellation.data import schema

ROWS = 4000


@pytest.fixture
def development() -> pd.DataFrame:
    """Canceled bookings with one variable that accumulates and others that do not."""
    rng = np.random.default_rng(0)
    arrival = pd.Timestamp("2016-06-01") + pd.to_timedelta(rng.integers(0, 120, ROWS), unit="D")
    lead_time = rng.integers(20, 200, ROWS)
    life = rng.uniform(0, 1, ROWS)
    days_to_cancel = np.floor(life * lead_time).astype(int)
    booking_date = arrival - pd.to_timedelta(lead_time, unit="D")
    cancel_date = booking_date + pd.to_timedelta(days_to_cancel, unit="D")
    return pd.DataFrame(
        {
            schema.ARRIVAL_DATE: arrival,
            "lead_time": lead_time,
            "is_canceled": 1,
            "reservation_status": "Canceled",
            "reservation_status_date": cancel_date.strftime("%Y-%m-%d"),
            "hotel": rng.choice(["City Hotel", "Resort Hotel"], ROWS),
            "market_segment": rng.choice(["Online TA", "Groups"], ROWS),
            "deposit_type": "No Deposit",
            "accumulates": rng.poisson(0.1 + 3 * days_to_cancel / lead_time),
            "fixed_number": rng.poisson(1.0, ROWS),
            "fixed_category": rng.choice(["x", "y", "z"], ROWS),
        }
    )


def _variables(frame: pd.DataFrame) -> dict[str, dict[Any, Any]]:
    table = feature_availability.diagnose(frame).table
    return {str(row["variable"]): row for row in table.to_dict("records")}


def test_accumulating_variable_grows_with_the_elapsed_life(development: pd.DataFrame) -> None:
    accumulates = _variables(development)["accumulates"]

    assert accumulates["rho_all"] > 0.4
    assert accumulates["rho_control"] > 0.4
    assert accumulates["rho_control_within_strata"] > 0.4
    assert accumulates["mean_75_100"] > accumulates["mean_0_5"] + 1


def test_a_variable_that_does_not_change_shows_no_trend(development: pd.DataFrame) -> None:
    variables = _variables(development)

    assert abs(variables["fixed_number"]["rho_all"]) < 0.05
    assert abs(variables["fixed_number"]["rho_control_within_strata"]) < 0.08
    assert variables["fixed_category"]["kind"] == "categorical"
    assert variables["fixed_category"]["cramers_v_control"] < 0.1


def test_one_day_cancellations_are_compared_with_the_rest(development: pd.DataFrame) -> None:
    variables = _variables(development)

    assert variables["accumulates"]["smd_one_day"] > 0.5
    assert abs(variables["fixed_number"]["smd_one_day"]) < 0.3


def test_the_elapsed_life_excludes_impossible_bookings(development: pd.DataFrame) -> None:
    broken = development.copy()
    broken.loc[broken.index[0], "lead_time"] = 0
    broken.loc[broken.index[1], "reservation_status_date"] = "2010-01-01"
    broken.loc[broken.index[2], "reservation_status_date"] = "2018-01-01"
    broken.loc[broken.index[3], "reservation_status"] = "No-Show"

    summary = feature_availability.diagnose(broken).summary

    assert summary["canceled_rows"] == ROWS - 1
    assert summary["excluded_zero_lead_time"] == 1
    assert summary["excluded_outside_life"] == 2
    assert summary["rows_used"] == ROWS - 4


def test_diagnostic_refuses_rows_of_the_test_period(development: pd.DataFrame) -> None:
    leaked = development.copy()
    leaked.loc[leaked.index[0], schema.ARRIVAL_DATE] = pd.Timestamp("2017-01-01")

    with pytest.raises(ValueError, match="2017"):
        feature_availability.diagnose(leaked)


def test_flags_mark_the_variables_at_or_above_the_control(development: pd.DataFrame) -> None:
    frame = development.assign(booking_changes=development["fixed_number"])

    result = feature_availability.diagnose(frame)

    assert "accumulates" in result.summary["flagged"]


def test_cohorts_repeat_the_measure_in_each_group_of_bookings(development: pd.DataFrame) -> None:
    frame = development.assign(booking_changes=development["fixed_number"])
    frame = pd.concat([frame, frame.head(500)], ignore_index=True)

    result = feature_availability.diagnose(frame)
    cohorts = {
        (str(row["variable"]), str(row["cohort"])): row for row in result.cohorts.to_dict("records")
    }

    assert {cohort for _, cohort in cohorts} == {
        "all",
        "unique_rows",
        "no_deposit_unique_rows",
        "online_ta",
        "online_ta_city",
        "online_ta_resort",
        "city",
        "resort",
    }
    assert cohorts["accumulates", "unique_rows"]["rows"] == ROWS
    assert cohorts["accumulates", "all"]["rows"] == ROWS + 500
    assert cohorts["accumulates", "online_ta_resort"]["rho"] > 0.4
    assert "accumulates" in result.summary["flagged_by_cohort"]["resort"]


def test_only_the_variables_of_the_frame_are_diagnosed(development: pd.DataFrame) -> None:
    assert set(_variables(development)) == {
        "lead_time",
        "hotel",
        "market_segment",
        "deposit_type",
        "accumulates",
        "fixed_number",
        "fixed_category",
    }


def test_a_count_that_is_almost_always_zero_still_gets_an_association(
    development: pd.DataFrame,
) -> None:
    rng = np.random.default_rng(1)
    life_proxy = development["accumulates"].to_numpy()
    rare_count = ((life_proxy >= 4) & (rng.uniform(size=ROWS) < 0.5)).astype(int)
    assert rare_count.mean() < 0.15
    frame = development.assign(rare_count=rare_count * rng.integers(1, 9, ROWS))

    assert _variables(frame)["rare_count"]["cramers_v_control"] > 0.05


def test_a_category_that_reaches_the_control_is_flagged(development: pd.DataFrame) -> None:
    rng = np.random.default_rng(2)
    days = pd.to_datetime(development["reservation_status_date"]) - (
        development[schema.ARRIVAL_DATE] - pd.to_timedelta(development["lead_time"], unit="D")
    )
    life = days.dt.days / development["lead_time"]
    frame = development.assign(
        booking_changes=rng.poisson(0.1 + life),
        drifting_category=(life > 0.5).map({True: "late", False: "early"}),
    )

    result = feature_availability.diagnose(frame)

    assert "drifting_category" in result.summary["flagged"]
    assert "fixed_category" not in result.summary["flagged"]
