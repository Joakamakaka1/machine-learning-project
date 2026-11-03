"""Measure what retraining more or less often is worth, on the same validation months.

The served model is never retrained, while the windows of the protocol retrain every quarter, so
the validation figure may flatter it. Every month of 2016 is scored by models whose training ends
at the start of that month, quarter, half year or year.

Run it with `uv run python -m experiments.review.cadence`.
"""

from typing import Any

import pandas as pd
from scipy import stats  # type: ignore[import-untyped]

from experiments import common
from experiments.review import harness
from hotel_cancellation.data import data_loader, schema

_CADENCES: tuple[str, ...] = ("monthly", "quarterly", "half_yearly", "frozen_at_2016_start")


def _last_origin(month: pd.Timestamp, cadence: str) -> pd.Timestamp:
    if cadence == "monthly":
        return month
    if cadence == "quarterly":
        return pd.Timestamp(month.year, 3 * ((month.month - 1) // 3) + 1, 1)
    if cadence == "half_yearly":
        return pd.Timestamp(month.year, 1 if month.month <= 6 else 7, 1)
    return pd.Timestamp("2016-01-01")


def _forest() -> Any:
    return common.tuned_pipeline("random_forest").set_params(model__n_jobs=-1)


def main() -> None:
    """Score each month with each cadence and compare every cadence with the quarterly one."""
    rows = harness.load_rows()
    dates = rows[schema.ARRIVAL_DATE]
    features = data_loader.coerce_features(rows)
    fitted: dict[pd.Timestamp, Any] = {}
    results = []
    for window in harness.M12:
        valid = (dates >= window.start) & (dates < window.end)
        for cadence in _CADENCES:
            origin = _last_origin(window.start, cadence)
            if origin not in fitted:
                train = dates < origin
                fitted[origin] = _forest().fit(features[train], rows.loc[train, schema.TARGET])
                print(f"fitted until {origin.date()}", flush=True)
            scores = fitted[origin].predict_proba(features[valid])[:, 1]
            results.append(
                {
                    "cadence": cadence,
                    "window": window.name,
                    "months_since_training": (window.start.year - origin.year) * 12
                    + window.start.month
                    - origin.month,
                    **harness.score_window(rows[valid], scores),
                }
            )
    by_window = pd.DataFrame(results).rename(columns={"cadence": "variant"})
    table = harness.add_holm(harness.compare_to_reference(by_window, "quarterly"))
    harness.save("cadence_by_window", by_window)
    harness.save("cadence", table)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(table.round(4).T.to_string())
    frozen = by_window[by_window["variant"] == "frozen_at_2016_start"]
    slope = stats.linregress(frozen["months_since_training"], frozen["roc_auc"])
    print(f"Frozen model: slope {slope.slope:.5f} per month, p {slope.pvalue:.3f}")
    print(
        frozen[["window", "prevalence", "roc_auc", "roc_auc_no_deposit"]]
        .round(4)
        .to_string(index=False)
    )
    print(
        "mean months since training:",
        by_window.groupby("variant")["months_since_training"].mean().round(1).to_dict(),
    )


if __name__ == "__main__":
    main()
