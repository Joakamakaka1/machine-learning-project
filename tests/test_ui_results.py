"""Tests of the figures the interface states, against the saved predictions of 2017."""

import pandas as pd
import pytest

from hotel_cancellation.evaluation import evaluator

pytest.importorskip("streamlit")

from hotel_cancellation.serving.ui import results

SERVED = evaluator.DEFAULT_COST_SCENARIO


@pytest.fixture(scope="module")
def predictions() -> pd.DataFrame:
    return results.load_predictions()


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.0, "<1 %"),
        (0.004, "<1 %"),
        (0.01, "1.0 %"),
        (0.2504, "25.0 %"),
        (0.99, "99.0 %"),
        (0.996, ">99 %"),
        (1.0, ">99 %"),
    ],
)
def test_extreme_probabilities_are_not_shown_as_certainties(
    probability: float, expected: str
) -> None:
    assert results.format_probability(probability) == expected


def test_outcome_counts_a_small_known_example() -> None:
    frame = pd.DataFrame(
        {"champion": [0.1, 0.3, 0.6, 0.9], results.TARGET_COLUMN: [1, 0, 1, 1]},
    )
    outcome = results.outcome_at(frame, 0.25, evaluator.CostScenario(1.0, 3.0))
    assert outcome.flagged_share == 0.75
    assert outcome.recall == pytest.approx(2 / 3)
    assert outcome.precision == pytest.approx(2 / 3)
    assert outcome.cost == pytest.approx((1 * 1 + 3 * 1) / 4)


def test_outcome_at_the_served_threshold_matches_the_report(predictions: pd.DataFrame) -> None:
    outcome = results.outcome_at(predictions, 0.25, SERVED)
    assert outcome.flagged_share == pytest.approx(0.773, abs=5e-4)
    assert outcome.recall == pytest.approx(0.941, abs=5e-4)
    assert outcome.precision == pytest.approx(0.473, abs=5e-4)
    assert outcome.cost == pytest.approx(0.476, abs=5e-4)


def test_the_no_model_point_is_marking_every_booking(predictions: pd.DataFrame) -> None:
    assert results.cost_marking_all(predictions, SERVED) == pytest.approx(0.6117, abs=5e-5)
    outcome = results.outcome_at(predictions, 0.0, SERVED)
    assert outcome.flagged_share == 1.0
    assert outcome.cost == results.cost_marking_all(predictions, SERVED)


def test_marking_nothing_is_priced_for_comparison(predictions: pd.DataFrame) -> None:
    cost = results.cost_marking_none(predictions, evaluator.CostScenario(1.0, 1.0))
    assert cost == pytest.approx(0.3883, abs=5e-5)


def test_the_isotonic_calibrator_gives_a_few_steps(predictions: pd.DataFrame) -> None:
    steps = results.probability_steps(predictions)
    assert steps.distinct_values == 128
    assert steps.share_zero == pytest.approx(0.0046, abs=5e-5)
    assert steps.share_one == pytest.approx(0.1005, abs=5e-5)


def test_the_cost_curve_covers_the_grid_and_the_served_threshold(
    predictions: pd.DataFrame,
) -> None:
    curve = results.cost_curve(predictions, SERVED)
    assert curve["threshold"].is_monotonic_increasing
    assert curve["threshold"].iloc[0] == pytest.approx(0.01)
    assert curve["threshold"].iloc[-1] == pytest.approx(0.99)
    at_served = curve.loc[curve["threshold"] == 0.25, "cost"].iloc[0]
    assert at_served == pytest.approx(0.476, abs=5e-4)


def test_between_030_and_040_the_cost_would_be_about_043(predictions: pd.DataFrame) -> None:
    low, high = results.cost_range(predictions, 0.30, 0.40, SERVED)
    assert low == pytest.approx(0.4304, abs=5e-4)
    assert high == pytest.approx(0.4386, abs=5e-4)


def test_the_served_threshold_falls_right_before_a_step(predictions: pd.DataFrame) -> None:
    at_served = results.outcome_at(predictions, 0.25, SERVED).flagged_share
    next_point = results.outcome_at(predictions, 0.26, SERVED).flagged_share
    assert at_served - next_point == pytest.approx(0.141, abs=5e-3)


def test_the_champion_rows_never_carry_the_global_figure_alone() -> None:
    rows = results.champion_subsets()
    assert rows.set_index("subset")["roc_auc"].to_dict() == pytest.approx(
        {"all": 0.8175, "unique": 0.7608, "no_deposit": 0.7529, "non_refund": 0.9998}, abs=5e-5
    )
