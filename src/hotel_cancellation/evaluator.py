"""Metrics, costs and decision thresholds. F1, precision and recall are of the positive class."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

_DEFAULT_THRESHOLD = 0.5
_THRESHOLD_GRID = np.linspace(0.01, 0.99, 99)
_LOGIT_CLIP = 1e-6


@dataclass(frozen=True)
class CostScenario:
    """Assumed cost of each error: a false positive and a false negative."""

    false_positive: float
    false_negative: float

    @property
    def name(self) -> str:
        """Short label, such as `cost_1_3`."""
        return f"cost_{self.false_positive:g}_{self.false_negative:g}"


# Assumptions, declared as such in the protocol. The API and the report use the central one.
COST_SCENARIOS: tuple[CostScenario, ...] = (
    CostScenario(1.0, 2.0),
    CostScenario(1.0, 3.0),
    CostScenario(1.0, 5.0),
)
DEFAULT_COST_SCENARIO = COST_SCENARIOS[1]

VALIDATION_METRICS: tuple[str, ...] = (
    "roc_auc",
    "pr_auc",
    "brier",
    "calibration_slope",
    "accuracy",
    "precision",
    "recall",
    "f1",
    *(scenario.name for scenario in COST_SCENARIOS),
    "roc_auc_unique",
    "roc_auc_no_deposit",
    "roc_auc_non_refund",
)


def optimal_threshold(scenario: CostScenario) -> float:
    """Return `C_FP / (C_FP + C_FN)`, the threshold that minimizes a calibrated model's cost."""
    return scenario.false_positive / (scenario.false_positive + scenario.false_negative)


def cost_per_booking(
    y_true: ArrayLike, probabilities: ArrayLike, threshold: float, scenario: CostScenario
) -> float:
    """Average the cost of the errors made at `threshold` over all bookings."""
    actual = np.asarray(y_true)
    predicted = np.asarray(probabilities) >= threshold
    false_positives = np.count_nonzero(predicted & (actual == 0))
    false_negatives = np.count_nonzero(~predicted & (actual == 1))
    total = scenario.false_positive * false_positives + scenario.false_negative * false_negatives
    return float(total / actual.size)


def empirical_best_threshold(
    y_true: ArrayLike, probabilities: ArrayLike, scenario: CostScenario
) -> tuple[float, float]:
    """Find the threshold with the lowest observed cost.

    It contrasts the theoretical threshold with the data. It is not used to decide anything.

    Returns:
        The threshold on a grid of hundredths, and its cost per booking.
    """
    costs = [
        cost_per_booking(y_true, probabilities, float(threshold), scenario)
        for threshold in _THRESHOLD_GRID
    ]
    best = int(np.argmin(costs))
    return float(_THRESHOLD_GRID[best]), costs[best]


def calibration_slope(y_true: ArrayLike, probabilities: ArrayLike) -> float:
    """Regress the outcome on the logit of the probability: 1 is calibrated, below 1 overconfident.

    Returns:
        The slope, or `NaN` if the model gives a constant probability.
    """
    values = np.asarray(probabilities, dtype=float)
    if np.ptp(values) == 0:
        return float("nan")
    clipped = np.clip(values, _LOGIT_CLIP, 1 - _LOGIT_CLIP)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    regression = LogisticRegression(C=np.inf, max_iter=1000)
    return float(regression.fit(logits, np.asarray(y_true)).coef_[0, 0])


def classification_metrics(
    y_true: ArrayLike, probabilities: ArrayLike, threshold: float = _DEFAULT_THRESHOLD
) -> dict[str, float]:
    """Measure the decisions taken at `threshold`.

    Returns:
        Accuracy, and the precision, recall and F1 of the positive class.
    """
    predicted = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, pos_label=1, zero_division=0)),
    }


def _roc_auc(y_true: NDArray[Any], probabilities: NDArray[Any]) -> float:
    """Return the ROC-AUC, or `NaN` when a subset holds a single class."""
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, probabilities))


def validation_metrics(
    y_true: pd.Series, probabilities: ArrayLike, features: pd.DataFrame
) -> dict[str, float]:
    """Compute every metric in `VALIDATION_METRICS` for a set of predictions.

    The ROC-AUC never goes alone: it comes with unique rows, `No Deposit` and `Non Refund`.
    Classification metrics use a threshold of 0.5, and each cost uses the threshold of its scenario.

    Args:
        y_true: Observed outcomes.
        probabilities: Predicted probability of cancellation.
        features: Input rows aligned with the outcomes, with `deposit_type`.

    Returns:
        A value per metric. It is `NaN` where a subset holds a single class.
    """
    actual = np.asarray(y_true)
    scores = np.asarray(probabilities, dtype=float)
    is_unique = ~features.assign(__target__=actual).duplicated().to_numpy()
    deposit = features["deposit_type"].to_numpy()
    no_deposit = deposit == "No Deposit"
    non_refund = deposit == "Non Refund"
    metrics = {
        "roc_auc": _roc_auc(actual, scores),
        "pr_auc": float(average_precision_score(actual, scores)),
        "brier": float(brier_score_loss(actual, scores)),
        "calibration_slope": calibration_slope(actual, scores),
        "roc_auc_unique": _roc_auc(actual[is_unique], scores[is_unique]),
        "roc_auc_no_deposit": _roc_auc(actual[no_deposit], scores[no_deposit]),
        "roc_auc_non_refund": _roc_auc(actual[non_refund], scores[non_refund]),
        **classification_metrics(actual, scores),
    }
    for scenario in COST_SCENARIOS:
        threshold = optimal_threshold(scenario)
        metrics[scenario.name] = cost_per_booking(actual, scores, threshold, scenario)
    return {name: metrics[name] for name in VALIDATION_METRICS}


def validation_scorer(estimator: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:  # noqa: N803
    """Score a fitted model: the callable that `RandomizedSearchCV` receives as `scoring`."""
    return validation_metrics(y, estimator.predict_proba(X)[:, 1], X)
