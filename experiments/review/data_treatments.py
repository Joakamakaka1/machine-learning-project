"""One change per experiment to what the model is trained on, keeping the validation rows intact.

Validation rows never change: only the training set or its weights do, so the gaps are paired.
Run it with `uv run python -m experiments.review.data_treatments`.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from experiments import common
from experiments.review import harness
from hotel_cancellation import data_loader, features, schema

_CITY_START = pd.Timestamp("2015-09-01")
_LOW_COVERAGE_MONTHS = ("2015-07", "2015-08", "2015-11", "2015-12", "2016-01")

Treatment = Callable[[pd.DataFrame], tuple[pd.DataFrame, NDArray[np.float64] | None]]


def _as_is(train: pd.DataFrame) -> tuple[pd.DataFrame, None]:
    return train, None


def _without_early_city(train: pd.DataFrame) -> tuple[pd.DataFrame, None]:
    """Drop City Hotel arrivals before September 2015, where stays look under-recorded."""
    keep = ~((train["hotel"] == "City Hotel") & (train[schema.ARRIVAL_DATE] < _CITY_START))
    return train[keep], None


def _without_low_coverage_city(train: pd.DataFrame) -> tuple[pd.DataFrame, None]:
    """Drop every City Hotel month whose occupied rooms look far below capacity."""
    month = train[schema.ARRIVAL_DATE].dt.to_period("M").astype("str")
    keep = ~((train["hotel"] == "City Hotel") & month.isin(_LOW_COVERAGE_MONTHS))
    return train[keep], None


def _last_months(months: int) -> Treatment:
    def treat(train: pd.DataFrame) -> tuple[pd.DataFrame, None]:
        start = train[schema.ARRIVAL_DATE].max() - pd.DateOffset(months=months)
        return train[train[schema.ARRIVAL_DATE] >= start], None

    return treat


def _recency_weights(half_life_days: float) -> Treatment:
    def treat(train: pd.DataFrame) -> tuple[pd.DataFrame, NDArray[np.float64]]:
        age = (train[schema.ARRIVAL_DATE].max() - train[schema.ARRIVAL_DATE]).dt.days
        return train, np.asarray(0.5 ** (age / half_life_days), dtype=float)

    return treat


def _batch_weights(power: float) -> Treatment:
    def treat(train: pd.DataFrame) -> tuple[pd.DataFrame, NDArray[np.float64]]:
        sizes = train.groupby("batch_id")["batch_id"].transform("size")
        return train, np.asarray(1.0 / sizes**power, dtype=float)

    return treat


def _without_non_refund(train: pd.DataFrame) -> tuple[pd.DataFrame, None]:
    """Train on the bookings where there is something to learn, and let the rest fall back."""
    return train[train["deposit_type"] != "Non Refund"], None


def _without_batches(min_size: int) -> Treatment:
    def treat(train: pd.DataFrame) -> tuple[pd.DataFrame, None]:
        sizes = train.groupby("batch_id")["batch_id"].transform("size")
        return train[sizes < min_size], None

    return treat


def _without_previous_cancellation_rows(train: pd.DataFrame) -> tuple[pd.DataFrame, None]:
    keep = ~((train["previous_cancellations"] > 0) & (train[schema.ARRIVAL_DATE] < "2016-01-01"))
    return train[keep], None


def _clip_outliers(train: pd.DataFrame) -> tuple[pd.DataFrame, None]:
    """Drop absurd party sizes and very long stays from training, kept in validation."""
    keep = (
        (train["adults"] <= 4)
        & (train["nights"] <= 30)
        & (train["babies"] <= 2)
        & (train["children"].fillna(0) <= 3)
    )
    return train[keep], None


def _treated(make_model: Callable[[], Any], treatment: Treatment) -> harness.FitPredict:
    def fit_predict(train: pd.DataFrame, valid: pd.DataFrame) -> NDArray[np.float64]:
        subset, weights = treatment(train)
        pipeline = make_model()
        pipeline.fit(
            data_loader.coerce_features(subset),
            subset[schema.TARGET],
            **({} if weights is None else {f"{features.MODEL_STEP}__sample_weight": weights}),
        )
        return np.asarray(pipeline.predict_proba(data_loader.coerce_features(valid))[:, 1])

    return fit_predict


def _xgboost_pipeline() -> Any:
    return features.build_pipeline(
        features.build_preprocessor(features.Family.UNSCALED), common.reference_xgboost()
    )


TREATMENTS: tuple[tuple[str, Treatment], ...] = (
    ("as_is", _as_is),
    ("without_city_before_2015_09", _without_early_city),
    ("without_low_coverage_city_months", _without_low_coverage_city),
    ("last_12_months", _last_months(12)),
    ("last_9_months", _last_months(9)),
    ("recency_half_life_180", _recency_weights(180.0)),
    ("recency_half_life_365", _recency_weights(365.0)),
    ("batch_weight_inverse", _batch_weights(1.0)),
    ("batch_weight_inverse_sqrt", _batch_weights(0.5)),
    ("without_batches_10_or_more", _without_batches(10)),
    ("without_non_refund", _without_non_refund),
    ("without_2015_previous_cancellations", _without_previous_cancellation_rows),
    ("without_outliers", _clip_outliers),
)


def main() -> None:
    """Run every treatment on both schemes with the reference XGBoost."""
    rows = harness.load_rows()
    pd.set_option("display.width", 240, "display.max_columns", 40)
    variants = [(name, _treated(_xgboost_pipeline, treat)) for name, treat in TREATMENTS]
    for scheme, windows in (("m12", harness.M12), ("q4", harness.Q4)):
        by_window = harness.run_variants(f"treatments {scheme}", rows, variants, windows)
        table = harness.add_holm(harness.compare_to_reference(by_window, "as_is"))
        harness.save(f"treatments_{scheme}_by_window", by_window)
        harness.save(f"treatments_{scheme}", table)
        print(
            table[
                [
                    "variant",
                    "roc_auc",
                    "roc_auc_gap",
                    "roc_auc_low",
                    "roc_auc_high",
                    "roc_auc_no_deposit",
                    "roc_auc_no_deposit_gap",
                    "roc_auc_no_deposit_low",
                    "roc_auc_no_deposit_high",
                    "log_loss_gap",
                ]
            ]
            .round(4)
            .to_string(index=False),
            flush=True,
        )


if __name__ == "__main__":
    main()
