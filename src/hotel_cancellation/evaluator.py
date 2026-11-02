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
# The slope of an isotonic model depends on this clip: see `calibration_diagnostics`.
_LOGIT_CLIP = 1e-3
_SENSITIVITY_CLIPS = {"calibration_slope_clip_1e6": 1e-6, "calibration_slope_clip_1e2": 1e-2}
_PSI_BINS = 10
_PSI_FLOOR = 1e-4
_MISSING_LABEL = "missing"

BOOTSTRAP_REPLICATES = 2000


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
    if actual.size == 0:
        return float("nan")
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


def costs_at_threshold(
    y_true: ArrayLike, probabilities: ArrayLike, threshold: float
) -> dict[str, float]:
    """Price the decisions taken at one threshold under every cost scenario."""
    return {
        f"{scenario.name}_at_threshold": cost_per_booking(
            y_true, probabilities, threshold, scenario
        )
        for scenario in COST_SCENARIOS
    }


def costs_at_own_threshold(y_true: ArrayLike, probabilities: ArrayLike) -> dict[str, float]:
    """Price each cost scenario at its own optimal threshold."""
    return {
        scenario.name: cost_per_booking(
            y_true, probabilities, optimal_threshold(scenario), scenario
        )
        for scenario in COST_SCENARIOS
    }


def calibration_slope(
    y_true: ArrayLike, probabilities: ArrayLike, *, clip: float = _LOGIT_CLIP
) -> float:
    """Regress the outcome on the logit of the probability: 1 is calibrated, below 1 overconfident.

    Args:
        y_true: Observed outcomes.
        probabilities: Predicted probability of cancellation.
        clip: Probabilities are limited to `[clip, 1 - clip]`, because the logit of 0 is infinite.

    Returns:
        The slope, or `NaN` if there is a single class or the probability is constant.
    """
    actual = np.asarray(y_true)
    clipped = np.clip(np.asarray(probabilities, dtype=float), clip, 1 - clip)
    if np.unique(actual).size < 2 or np.ptp(clipped) == 0:
        return float("nan")
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    regression = LogisticRegression(C=np.inf, max_iter=1000)
    return float(regression.fit(logits, actual).coef_[0, 0])


def calibration_diagnostics(y_true: ArrayLike, probabilities: ArrayLike) -> dict[str, float]:
    """Give the calibration slope with what it depends on.

    An isotonic calibrator gives some rows a probability of exactly 0 or 1, so the slope depends on
    the clip of those rows. The share of such rows says how much that matters.

    Returns:
        The slope, the slope with each alternative clip, and the share of extreme rows.
    """
    values = np.asarray(probabilities, dtype=float)
    empty = values.size == 0
    return {
        "calibration_slope": calibration_slope(y_true, values),
        **{
            name: calibration_slope(y_true, values, clip=clip)
            for name, clip in _SENSITIVITY_CLIPS.items()
        },
        "share_probability_zero": float("nan") if empty else float((values == 0).mean()),
        "share_probability_one": float("nan") if empty else float((values == 1).mean()),
    }


def classification_metrics(
    y_true: ArrayLike, probabilities: ArrayLike, threshold: float = _DEFAULT_THRESHOLD
) -> dict[str, float]:
    """Measure the decisions taken at `threshold`.

    Returns:
        Accuracy, and the precision, recall and F1 of the positive class. A metric that divides by
        zero (no positive predictions for the precision, no positive rows for the recall) is `NaN`.
    """
    actual = np.asarray(y_true)
    if actual.size == 0:
        return dict.fromkeys(("accuracy", "precision", "recall", "f1"), float("nan"))
    predicted = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, pos_label=1, zero_division=np.nan)),
        "recall": float(recall_score(actual, predicted, pos_label=1, zero_division=np.nan)),
        "f1": float(f1_score(actual, predicted, pos_label=1, zero_division=np.nan)),
    }


def _roc_auc(y_true: NDArray[Any], probabilities: NDArray[Any]) -> float:
    """Return the ROC-AUC, or `NaN` when a subset holds a single class."""
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, probabilities))


def _pr_auc(y_true: NDArray[Any], probabilities: NDArray[Any]) -> float:
    """Return the area under the PR curve, or `NaN` when a subset holds a single class."""
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, probabilities))


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
        "pr_auc": _pr_auc(actual, scores),
        "brier": float(brier_score_loss(actual, scores)) if actual.size else float("nan"),
        "calibration_slope": calibration_slope(actual, scores),
        "roc_auc_unique": _roc_auc(actual[is_unique], scores[is_unique]),
        "roc_auc_no_deposit": _roc_auc(actual[no_deposit], scores[no_deposit]),
        "roc_auc_non_refund": _roc_auc(actual[non_refund], scores[non_refund]),
        **classification_metrics(actual, scores),
        **costs_at_own_threshold(actual, scores),
    }
    return {name: metrics[name] for name in VALIDATION_METRICS}


def _numeric_labels(
    reference: pd.Series, current: pd.Series, bins: int
) -> tuple[pd.Series, pd.Series]:
    """Label each number with its bin: a bin per value that is tied, and quantiles for the rest.

    Quantiles alone collapse into a single bin in a variable that is mostly zero, which would hide
    any shift. A value holding at least a share of `1 / bins` of the reference is tied.
    """
    shares = reference.value_counts(normalize=True)
    tied = shares.index[shares >= 1 / bins] if len(shares) > bins else shares.index
    untied = reference[reference.notna() & ~reference.isin(tied)]
    inner: list[float] = []
    if not untied.empty:
        quantiles = np.quantile(untied.to_numpy(dtype=float), np.linspace(0, 1, bins + 1))
        inner = [float(edge) for edge in np.unique(quantiles)[1:-1]]
    edges = [-np.inf, *inner, np.inf]

    def label(values: pd.Series) -> pd.Series:
        binned = pd.cut(values, edges, include_lowest=True).astype("str")
        # In a discrete variable, a value new to the reference gets a bin of its own.
        exact = values.isin(tied) | (untied.empty & values.notna())
        return binned.where(~exact, values.astype("str")).where(values.notna(), other=None)

    return label(reference), label(current)


def _bin_shares(
    reference: pd.Series, current: pd.Series, bins: int
) -> tuple[NDArray[Any], NDArray[Any]]:
    if pd.api.types.is_numeric_dtype(reference):
        first, second = _numeric_labels(reference, current, bins)
    else:
        first, second = reference, current
    first, second = first.fillna(_MISSING_LABEL), second.fillna(_MISSING_LABEL)
    categories = sorted(set(first) | set(second))
    shares = [
        values.value_counts(normalize=True).reindex(categories, fill_value=0.0).to_numpy()
        for values in (first, second)
    ]
    return shares[0], shares[1]


def population_stability_index(
    reference: pd.Series, current: pd.Series, *, bins: int = _PSI_BINS
) -> float:
    """Measure how far a variable moved: above 0.25 is usually read as a large shift.

    Args:
        reference: Values the model learned from.
        current: Values to compare with them.
        bins: Quantile bins of a numeric variable, apart from its tied values. Categories are
            compared as they are.

    Returns:
        The index, with a small floor so that a new category gives a finite value.
    """
    expected, actual = _bin_shares(reference, current, bins)
    expected = np.clip(expected, _PSI_FLOOR, None)
    actual = np.clip(actual, _PSI_FLOOR, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _quantile(samples: NDArray[Any], level: float) -> float:
    return float(np.quantile(samples, level)) if samples.size else float("nan")


def block_bootstrap_auc(
    y_true: pd.Series,
    probabilities: pd.DataFrame,
    blocks: pd.Series,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = 0,
    against: str | None = None,
) -> pd.DataFrame:
    """Compute a percentile interval for the ROC-AUC, resampling whole blocks.

    Bookings of a block are not independent: identical rows share their date. All the models are
    measured on the same replicates, so the gaps between them are paired. The interval does not
    cover temporal drift, which is published apart.

    Args:
        y_true: Observed outcomes.
        probabilities: One column of predicted probabilities per model.
        blocks: Block of each row.
        replicates: Bootstrap replicates.
        seed: Seed of the resampling.
        against: Model to compare every other model with.

    Returns:
        A row per model. With `against`, it includes the gap and its interval.
    """
    actual = y_true.to_numpy()
    block_codes, block_names = pd.factorize(blocks)
    block_count = len(block_names)
    rng = np.random.default_rng(seed)
    per_model: dict[str, list[float]] = {name: [] for name in probabilities.columns}
    used = 0
    for _ in range(replicates):
        counts = np.bincount(rng.integers(0, block_count, block_count), minlength=block_count)
        weights = counts[block_codes]
        if np.unique(actual[weights > 0]).size < 2:
            continue
        used += 1
        for name in probabilities.columns:
            per_model[name].append(
                float(roc_auc_score(actual, probabilities[name], sample_weight=weights))
            )

    rows = []
    for name in probabilities.columns:
        samples = np.asarray(per_model[name])
        row: dict[str, Any] = {
            "model": name,
            "roc_auc": _roc_auc(actual, probabilities[name].to_numpy()),
            "ci_low": _quantile(samples, 0.025),
            "ci_high": _quantile(samples, 0.975),
            "replicates_used": used,
        }
        if against is not None:
            gaps = samples - np.asarray(per_model[against])
            row["gap_ci_low"] = _quantile(gaps, 0.025)
            row["gap_ci_high"] = _quantile(gaps, 0.975)
        rows.append(row)
    return pd.DataFrame(rows)


def validation_scorer(estimator: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:  # noqa: N803
    """Score a fitted model: the callable that `RandomizedSearchCV` receives as `scoring`."""
    return validation_metrics(y, estimator.predict_proba(X)[:, 1], X)
