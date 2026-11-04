"""The machinery of the second review on small synthetic rows. It needs the train extra."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

pytest.importorskip("xgboost")

from experiments.review import feature_tiers, harness
from hotel_cancellation import config
from hotel_cancellation.data import schema


def test_monthly_windows_cover_2016_without_reaching_the_test() -> None:
    assert len(harness.M12) == 12
    assert harness.M12[0].start == pd.Timestamp("2016-01-01")
    assert harness.M12[-1].end == config.TEST_START
    for window in (*harness.M12, *harness.Q4):
        assert window.train_end <= window.start < window.end <= config.TEST_START


def test_distinct_auc_counts_each_input_vector_once() -> None:
    actual = np.array([1, 1, 1, 0, 0])
    scores = np.array([0.9, 0.9, 0.9, 0.2, 0.1])
    hashes = np.array([1, 1, 1, 2, 3])
    assert harness.distinct_auc(actual, scores, hashes) == 1.0


def test_distinct_auc_does_not_depend_on_how_many_times_a_vector_repeats() -> None:
    actual = np.array([1, 0, 1, 0])
    scores = np.array([0.8, 0.3, 0.6, 0.7])
    hashes = np.array([1, 2, 3, 4])
    expected = harness.distinct_auc(actual, scores, hashes)
    repeated = harness.distinct_auc(
        np.r_[actual, [1, 1]], np.r_[scores, [0.8, 0.8]], np.r_[hashes, [1, 1]]
    )
    assert repeated == pytest.approx(expected)
    assert expected == pytest.approx(roc_auc_score(actual, scores))


def test_holm_scales_and_keeps_the_order() -> None:
    assert harness.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def _history_rows() -> pd.DataFrame:
    arrival = pd.to_datetime(["2016-01-10", "2016-02-10", "2016-03-10", "2016-04-10"])
    booking = pd.to_datetime(["2016-01-01", "2016-02-01", "2016-03-01", "2016-04-01"])
    status = pd.to_datetime(["2016-01-05", "2016-02-12", "2016-03-12", "2016-04-12"])
    return pd.DataFrame(
        {
            schema.ARRIVAL_DATE: arrival,
            "booking_date": booking,
            "status_date": status,
            "agent_key": ["9", "9", "9", "9"],
            schema.TARGET: [1, 0, 1, 0],
        }
    )


def test_history_ignores_outcomes_not_settled_yet() -> None:
    rows = _history_rows()
    history = feature_tiers.settled_history(rows, ("agent_key",))
    changed = rows.assign(**{schema.TARGET: [1, 0, 1, 1]})
    history_changed = feature_tiers.settled_history(changed, ("agent_key",))
    # Only the last row's own outcome changed, and nothing can read it.
    assert history.iloc[:, 0].iloc[-1] == pytest.approx(history_changed.iloc[:, 0].iloc[-1])


def test_history_waits_for_arrival_even_if_the_booking_was_canceled_earlier() -> None:
    rows = _history_rows()
    rows.loc[0, schema.ARRIVAL_DATE] = pd.Timestamp("2016-03-15")
    history = feature_tiers.settled_history(rows, ("agent_key",))
    counts = np.expm1(history.iloc[:, 1].to_numpy())
    # The first booking was canceled in January, but it only counts after its arrival in March.
    assert counts[1] == pytest.approx(0)
    assert counts[3] == pytest.approx(3)


def test_rows_without_any_history_do_not_read_the_future() -> None:
    rows = _history_rows()
    changed = rows.assign(**{schema.TARGET: [1, 1, 1, 1]})
    first = feature_tiers.settled_history(rows, ("agent_key",)).iloc[:, 0].to_numpy()[0]
    second = feature_tiers.settled_history(changed, ("agent_key",)).iloc[:, 0].to_numpy()[0]
    assert first == second


def test_a_cancellation_event_only_counts_canceled_bookings() -> None:
    day = pd.Timestamp("2016-01-05")
    rows = pd.DataFrame(
        {
            "hotel": ["City Hotel"] * 3,
            "status_date": [day] * 3,
            "agent_key": ["9"] * 3,
            "market_segment": ["Online TA"] * 3,
            "deposit_type": ["No Deposit"] * 3,
            "reservation_status": ["Canceled", "Canceled", "Check-Out"],
            "batch_id": [1, 2, 3],
        }
    )
    assert harness.cancellation_event_size(rows).tolist() == [2, 2, 0]
