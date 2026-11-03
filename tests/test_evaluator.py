import math

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from hotel_cancellation.data import schema
from hotel_cancellation.evaluation import evaluator


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


class TestUndefinedMetrics:
    def test_slope_is_undefined_with_a_single_class(self) -> None:
        assert math.isnan(evaluator.calibration_slope(np.zeros(3), np.array([0.1, 0.2, 0.3])))
        assert math.isnan(evaluator.calibration_slope(np.ones(3), np.array([0.1, 0.2, 0.3])))

    def test_metrics_of_a_subset_with_a_single_class_are_nan_instead_of_failing(self) -> None:
        rows = pd.DataFrame({"deposit_type": ["Refundable"] * 3, "lead_time": [1.0, 2.0, 3.0]})
        scores = evaluator.validation_metrics(pd.Series([0, 0, 0]), np.array([0.1, 0.2, 0.3]), rows)
        for metric in ("roc_auc", "pr_auc", "calibration_slope", "recall", "precision", "f1"):
            assert math.isnan(scores[metric]), metric
        assert scores["brier"] == pytest.approx((0.01 + 0.04 + 0.09) / 3)
        assert scores["accuracy"] == 1.0

    def test_an_empty_subset_gives_nan_in_every_metric(self) -> None:
        rows = pd.DataFrame({"deposit_type": pd.Series([], dtype="str")})
        scores = evaluator.validation_metrics(pd.Series([], dtype=int), np.array([]), rows)
        assert all(math.isnan(value) for value in scores.values())

    def test_precision_without_positive_predictions_is_undefined(self) -> None:
        metrics = evaluator.classification_metrics(np.array([0, 1]), np.zeros(2), threshold=0.5)
        assert math.isnan(metrics["precision"])
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0


class TestCalibrationDiagnostics:
    @pytest.fixture
    def stepped(self) -> tuple[np.ndarray, np.ndarray]:
        """Isotonic-like probabilities: some rows reach exactly 0 or 1, and not all come true."""
        rng = np.random.default_rng(0)
        values = rng.choice([0.0, 0.2, 0.5, 0.8, 1.0], 5000)
        outcomes = rng.binomial(1, np.clip(values, 0.1, 0.9))
        return outcomes, values

    def test_the_definition_clips_at_one_in_a_thousand(
        self, stepped: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y, p = stepped
        clipped = np.clip(p, 1e-3, 1 - 1e-3)
        assert evaluator.calibration_slope(y, p) == pytest.approx(
            evaluator.calibration_slope(y, clipped)
        )

    def test_it_reports_the_sensitivity_to_the_clip_and_the_extreme_rows(
        self, stepped: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y, p = stepped
        diagnostics = evaluator.calibration_diagnostics(y, p)
        assert diagnostics["calibration_slope"] == evaluator.calibration_slope(y, p)
        assert diagnostics["calibration_slope_clip_1e6"] == pytest.approx(
            evaluator.calibration_slope(y, p, clip=1e-6)
        )
        assert diagnostics["calibration_slope_clip_1e2"] == pytest.approx(
            evaluator.calibration_slope(y, p, clip=1e-2)
        )
        assert diagnostics["calibration_slope_clip_1e6"] < diagnostics["calibration_slope"]
        assert diagnostics["calibration_slope"] < diagnostics["calibration_slope_clip_1e2"]
        assert diagnostics["share_probability_zero"] == pytest.approx((p == 0).mean())
        assert diagnostics["share_probability_one"] == pytest.approx((p == 1).mean())


class TestCostsAtTheServedThreshold:
    Y = np.array([1, 1, 1, 0, 0])
    P = np.array([0.9, 0.2, 0.1, 0.9, 0.1])

    def test_each_scenario_is_priced_at_the_served_threshold(self) -> None:
        costs = evaluator.costs_at_threshold(self.Y, self.P, 0.25)
        # One false positive and two false negatives on five bookings.
        assert costs["cost_1_2_at_threshold"] == pytest.approx((1 + 2 * 2) / 5)
        assert costs["cost_1_3_at_threshold"] == pytest.approx((1 + 2 * 3) / 5)
        assert costs["cost_1_5_at_threshold"] == pytest.approx((1 + 2 * 5) / 5)

    def test_each_scenario_is_priced_at_its_own_threshold(self) -> None:
        costs = evaluator.costs_at_own_threshold(self.Y, self.P)
        # At 1/6 the row with 0.2 stops being a false negative, which the served threshold misses.
        assert costs["cost_1_5"] == pytest.approx((1 + 1 * 5) / 5)
        assert costs["cost_1_3"] == pytest.approx((1 + 2 * 3) / 5)
