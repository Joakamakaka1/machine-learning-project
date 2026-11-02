import math

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from hotel_cancellation import evaluator, schema


def test_f1_is_the_one_of_the_positive_class() -> None:
    # TP = 1, FP = 1, FN = 1 and TN = 3: F1 of the positive class is 0.5, the weighted one 0.667.
    y = np.array([1, 1, 0, 0, 0, 0])
    probabilities = np.array([0.9, 0.1, 0.8, 0.2, 0.2, 0.2])
    metrics = evaluator.classification_metrics(y, probabilities, threshold=0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["accuracy"] == pytest.approx(4 / 6)


def test_f1_keeps_the_positive_class_when_negatives_dominate() -> None:
    y = np.array([0] * 98 + [1, 1])
    never_positive = np.zeros(100)
    metrics = evaluator.classification_metrics(y, never_positive, threshold=0.5)
    assert metrics["accuracy"] == pytest.approx(0.98)
    assert metrics["f1"] == 0.0


def test_threshold_follows_the_costs() -> None:
    assert evaluator.optimal_threshold(evaluator.CostScenario(1, 3)) == pytest.approx(0.25)
    assert evaluator.optimal_threshold(evaluator.CostScenario(1, 2)) == pytest.approx(1 / 3)
    assert evaluator.optimal_threshold(evaluator.CostScenario(1, 5)) == pytest.approx(1 / 6)


def test_the_three_declared_scenarios_exist() -> None:
    ratios = [(s.false_positive, s.false_negative) for s in evaluator.COST_SCENARIOS]
    assert ratios == [(1.0, 2.0), (1.0, 3.0), (1.0, 5.0)]


def test_cost_counts_each_error_with_its_price() -> None:
    # One false positive and two false negatives on five bookings.
    y = np.array([1, 1, 1, 0, 0])
    probabilities = np.array([0.9, 0.1, 0.1, 0.9, 0.1])
    cost = evaluator.cost_per_booking(y, probabilities, 0.5, evaluator.CostScenario(1, 3))
    assert cost == pytest.approx((1 * 1 + 2 * 3) / 5)


def test_threshold_minimizes_the_cost_of_a_calibrated_model() -> None:
    rng = np.random.default_rng(0)
    probabilities = rng.uniform(0, 1, 200_000)
    y = rng.binomial(1, probabilities)
    scenario = evaluator.CostScenario(1, 3)
    best = evaluator.optimal_threshold(scenario)
    at_best = evaluator.cost_per_booking(y, probabilities, best, scenario)
    assert at_best < evaluator.cost_per_booking(y, probabilities, 0.5, scenario)
    assert at_best < evaluator.cost_per_booking(y, probabilities, 0.1, scenario)


def test_empirical_threshold_recovers_the_theoretical_one() -> None:
    rng = np.random.default_rng(0)
    probabilities = rng.uniform(0, 1, 100_000)
    y = rng.binomial(1, probabilities)
    threshold, _ = evaluator.empirical_best_threshold(
        y, probabilities, evaluator.CostScenario(1, 3)
    )
    assert threshold == pytest.approx(0.25, abs=0.03)


def test_calibration_slope_detects_overconfidence() -> None:
    rng = np.random.default_rng(0)
    true_logit = rng.normal(0, 1, 50_000)
    y = rng.binomial(1, 1 / (1 + np.exp(-true_logit)))
    calibrated = 1 / (1 + np.exp(-true_logit))
    overconfident = 1 / (1 + np.exp(-2 * true_logit))
    assert evaluator.calibration_slope(y, calibrated) == pytest.approx(1.0, abs=0.05)
    assert evaluator.calibration_slope(y, overconfident) == pytest.approx(0.5, abs=0.05)


def test_calibration_slope_is_undefined_for_a_constant_model() -> None:
    assert math.isnan(evaluator.calibration_slope(np.array([0, 1, 1]), np.full(3, 0.5)))


def _validation_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lead_time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 7.0, 8.0],
            schema.TARGET: [0, 0, 1, 0, 1, 1, 1, 1, 1],
            "deposit_type": ["No Deposit"] * 7 + ["Non Refund"] * 2,
        }
    )


def test_validation_scorer_reports_every_metric_and_every_subset() -> None:
    rows = _validation_rows()
    X, y = rows[["lead_time", "deposit_type"]], rows[schema.TARGET]
    model = _LeadTimeModel().fit(X, y)
    scores = evaluator.validation_scorer(model, X, y)
    assert set(scores) == set(evaluator.VALIDATION_METRICS)
    # Pairs (positive, negative) ranked by hand; the sixth and seventh rows are identical.
    assert scores["roc_auc"] == pytest.approx(17 / 18)
    assert scores["roc_auc_unique"] == pytest.approx(14 / 15)
    assert scores["roc_auc_no_deposit"] == pytest.approx(11 / 12)
    assert math.isnan(scores["roc_auc_non_refund"])


def test_validation_scorer_survives_a_baseline_without_signal() -> None:
    rows = _validation_rows()
    X, y = rows[["lead_time", "deposit_type"]], rows[schema.TARGET]
    baseline = DummyClassifier(strategy="prior").fit(X, y)
    scores = evaluator.validation_scorer(baseline, X, y)
    assert scores["roc_auc"] == pytest.approx(0.5)
    assert math.isnan(scores["calibration_slope"])
    assert scores["brier"] == pytest.approx(y.mean() * (1 - y.mean()))


class _LeadTimeModel(LogisticRegression):  # type: ignore[misc]
    """Scores each booking by its lead time, so the expected ranks can be worked out by hand."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_LeadTimeModel":
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        positive = X["lead_time"].to_numpy() / 10
        return np.column_stack([1 - positive, positive])
