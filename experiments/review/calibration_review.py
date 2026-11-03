"""Calibration and threshold across three origins, instead of the single pair of the protocol.

Each origin fits a base model, calibrates it on the next quarter and measures on the one after.
Run it with `uv run python -m experiments.review.calibration_review`.
"""

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

from experiments import common
from experiments.review import harness
from hotel_cancellation import evaluator
from hotel_cancellation.data import data_loader, schema

_COST = evaluator.DEFAULT_COST_SCENARIO
_THRESHOLD = evaluator.optimal_threshold(_COST)
_GRID = np.round(np.arange(0.10, 0.61, 0.01), 2)
_RESAMPLES = 200
_ORIGINS: tuple[tuple[str, str, str, str], ...] = (
    ("2016-01-01", "2016-04-01", "2016-07-01", "fit_q1_measure_q2"),
    ("2016-04-01", "2016-07-01", "2016-10-01", "fit_q2_measure_q3"),
    ("2016-07-01", "2016-10-01", "2017-01-01", "fit_q3_measure_q4"),
)


def expected_calibration_error(actual: NDArray[Any], scores: NDArray[Any], bins: int = 10) -> float:
    """Compute the mean gap between predicted and observed rates over equal-frequency bins."""
    frame = pd.DataFrame({"y": actual, "p": scores})
    labels = pd.qcut(frame["p"], bins, duplicates="drop")
    grouped = frame.groupby(labels, observed=True).agg(
        y=("y", "mean"), p=("p", "mean"), n=("y", "size")
    )
    return float(np.average((grouped["y"] - grouped["p"]).abs(), weights=grouped["n"]))


def decision_row(actual: NDArray[Any], scores: NDArray[Any]) -> dict[str, float]:
    """Measure the fixed cost threshold and the price of not having the best one."""
    best_threshold, best_cost = evaluator.empirical_best_threshold(actual, scores, _COST)
    costs = [evaluator.cost_per_booking(actual, scores, float(t), _COST) for t in _GRID]
    flags = [float((scores >= t).mean()) for t in _GRID]
    around = (_GRID >= _THRESHOLD - 0.05) & (_GRID <= _THRESHOLD + 0.05)
    return {
        "brier": float(brier_score_loss(actual, scores)),
        "log_loss": float(log_loss(actual, np.clip(scores, 1e-6, 1 - 1e-6))),
        "slope_clip_1e3": evaluator.calibration_slope(actual, scores),
        "ece": expected_calibration_error(actual, scores),
        "share_probability_zero": float((scores == 0).mean()),
        "share_probability_one": float((scores == 1).mean()),
        "flagged_at_threshold": float((scores >= _THRESHOLD).mean()),
        "recall_at_threshold": float((scores[actual == 1] >= _THRESHOLD).mean()),
        "cost_at_threshold": evaluator.cost_per_booking(actual, scores, _THRESHOLD, _COST),
        "best_threshold": best_threshold,
        "cost_at_best_threshold": best_cost,
        "flag_swing_within_0_05": float(np.ptp(np.array(flags)[around])),
        "cost_swing_within_0_05": float(np.ptp(np.array(costs)[around])),
    }


def _fit_base(name: str, rows: pd.DataFrame) -> Any:
    pipeline = common.tuned_pipeline(name)
    pipeline.set_params(model__n_jobs=-1)
    return pipeline.fit(data_loader.coerce_features(rows), rows[schema.TARGET])


def _windows(rows: pd.DataFrame, first: str, second: str) -> pd.DataFrame:
    dates = rows[schema.ARRIVAL_DATE]
    return rows[(dates >= first) & (dates < second)]


def compare_methods(rows: pd.DataFrame) -> pd.DataFrame:
    """Compare calibration methods across origins and base models."""
    results = []
    for base_end, calibration_end, measure_end, origin in _ORIGINS:
        base_rows = rows[rows[schema.ARRIVAL_DATE] < base_end]
        fit_rows = _windows(rows, base_end, calibration_end)
        measure_rows = _windows(rows, calibration_end, measure_end)
        x_fit = data_loader.coerce_features(fit_rows)
        x_measure = data_loader.coerce_features(measure_rows)
        actual = measure_rows[schema.TARGET].to_numpy()
        for name in ("random_forest", "xgboost"):
            base = _fit_base(name, base_rows)
            scores = {"none": base.predict_proba(x_measure)[:, 1]}
            for method in ("sigmoid", "isotonic"):
                calibrated = CalibratedClassifierCV(FrozenEstimator(base), method=method)
                calibrated.fit(x_fit, fit_rows[schema.TARGET])
                scores[method] = calibrated.predict_proba(x_measure)[:, 1]
            for method, values in scores.items():
                results.append(
                    {
                        "model": name,
                        "origin": origin,
                        "method": method,
                        "fit_rows": len(fit_rows),
                        "fit_prevalence": float(fit_rows[schema.TARGET].mean()),
                        "measure_prevalence": float(actual.mean()),
                        **decision_row(actual, values),
                    }
                )
                print(f"{origin} | {name} | {method}: {results[-1]['brier']:.4f}", flush=True)
    return pd.DataFrame(results)


def threshold_stability(rows: pd.DataFrame) -> pd.DataFrame:
    """Resample the calibration rows and see how the share of flagged bookings moves.

    The base model is trained before July, calibrated on 2016T3 and measured on 2016T4, one quarter
    earlier than the champion. It is not refitted, so this isolates the instability of the
    calibrator. Batches are resampled whole.
    """
    base_rows = rows[rows[schema.ARRIVAL_DATE] < "2016-07-01"]
    fit_rows = _windows(rows, "2016-07-01", "2016-10-01")
    measure_rows = _windows(rows, "2016-10-01", "2017-01-01")
    base = _fit_base("random_forest", base_rows)
    fit_scores = base.predict_proba(data_loader.coerce_features(fit_rows))[:, 1]
    measure_scores = base.predict_proba(data_loader.coerce_features(measure_rows))[:, 1]
    actual_fit = fit_rows[schema.TARGET].to_numpy()
    actual = measure_rows[schema.TARGET].to_numpy()
    codes, names = pd.factorize(fit_rows["batch_id"])
    rng = np.random.default_rng(common.SEED)
    samples = []
    for draw in range(_RESAMPLES):
        weights = np.bincount(rng.integers(0, names.size, names.size), minlength=names.size)[codes]
        isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        isotonic.fit(fit_scores, actual_fit, sample_weight=weights)
        scores = isotonic.predict(measure_scores)
        samples.append(
            {
                "draw": draw,
                "steps": len(isotonic.X_thresholds_),
                "flagged_at_threshold": float((scores >= _THRESHOLD).mean()),
                "cost_at_threshold": evaluator.cost_per_booking(actual, scores, _THRESHOLD, _COST),
                "brier": float(brier_score_loss(actual, scores)),
            }
        )
    return pd.DataFrame(samples)


def main() -> None:
    """Write the comparison and the stability of the threshold."""
    rows = harness.load_rows()
    pd.set_option("display.width", 260, "display.max_columns", 40)
    methods = compare_methods(rows)
    harness.save("calibration_methods", methods)
    print(methods.round(4).to_string(index=False))
    stability = threshold_stability(rows)
    harness.save("calibration_threshold_stability", stability)
    print(stability.describe().round(4).to_string())


if __name__ == "__main__":
    main()
