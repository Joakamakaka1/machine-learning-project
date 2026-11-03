"""Variables: what each one contributes, grouped by what the dataset paper says about its timing.

Antonio et al. (2019, Data in Brief) rebuild the values relative to the day before arrival, so an
input is not proven to be what was known at confirmation. The tiers follow their Table 1:

- `by_definition`: defined at the creation or confirmation of the booking.
- `amendable`: read from the booking and its change log, so they may hold a later value.
- `settled_later`: `deposit_type`, computed from payments before arrival or cancellation.

New variables use the past only: a booking counts as history once its outcome is settled, that is,
once both its arrival and its last status are before the booking date of the row.

Run it with `uv run python -m experiments.review.feature_tiers`.
"""

import sys
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from experiments import common
from experiments.review import harness
from hotel_cancellation.data import data_loader, features, schema

BY_DEFINITION: tuple[str, ...] = (
    "hotel",
    "lead_time",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "days_in_waiting_list",
)
_SMOOTHING = 20.0
# Rate of the rows booked before any booking had settled: it must not come from the target.
_RATE_WITHOUT_HISTORY = 0.5
_RECENT_DAYS = 180


def settled_history(
    rows: pd.DataFrame, keys: Sequence[str], *, days: int | None = None
) -> pd.DataFrame:
    """Compute, per row, the cancellation rate of its group among bookings already settled.

    Args:
        rows: Development rows with `booking_date`, `status_date` and the arrival date.
        keys: Columns that define the group.
        days: Only count bookings settled within this many days before the booking date.

    Returns:
        The smoothed rate and the logarithm of the count of settled bookings, aligned with `rows`.
    """
    settled = rows[[schema.ARRIVAL_DATE, "status_date"]].max(axis=1).to_numpy("datetime64[D]")
    booking = rows["booking_date"].to_numpy("datetime64[D]")
    target = rows[schema.TARGET].to_numpy(dtype=float)

    def cumulative(
        indices: NDArray[np.intp], query: NDArray[np.intp]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        order = indices[np.argsort(settled[indices], kind="stable")]
        sorted_settled = settled[order]
        running = np.r_[0.0, np.cumsum(target[order])]
        upper = np.searchsorted(sorted_settled, booking[query], side="left")
        if days is None:
            return upper.astype(float), running[upper]
        lower = np.searchsorted(
            sorted_settled, booking[query] - np.timedelta64(days, "D"), side="left"
        )
        return (upper - lower).astype(float), running[upper] - running[lower]

    everything = np.arange(len(rows))
    total_count, total_sum = cumulative(everything, everything)
    prior = np.where(
        total_count > 0, total_sum / np.maximum(total_count, 1.0), _RATE_WITHOUT_HISTORY
    )
    rate = np.zeros(len(rows))
    count = np.zeros(len(rows))
    key_values = rows[list(keys)].astype("str").agg("|".join, axis=1)
    for _, group in key_values.groupby(key_values, sort=False):
        positions = rows.index.get_indexer(group.index)
        group_count, group_sum = cumulative(positions, positions)
        rate[positions] = (group_sum + _SMOOTHING * prior[positions]) / (group_count + _SMOOTHING)
        count[positions] = group_count
    name = "_".join(keys) + (f"_{days}d" if days else "")
    return pd.DataFrame(
        {f"history_rate_{name}": rate, f"history_count_{name}": np.log1p(count)}, index=rows.index
    )


def _select(columns: Sequence[str]) -> tuple[list[str], list[str]]:
    numeric = [column for column in columns if column in schema.NUMERIC_FEATURES]
    categorical = [column for column in columns if column in schema.CATEGORICAL_FEATURES]
    return numeric, categorical


def predictor(
    columns: Sequence[str],
    *,
    extra: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    extra_categorical: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    make_model: Callable[[], object] = common.reference_xgboost,
    min_frequency: int = features.MIN_CATEGORY_FREQUENCY,
) -> harness.FitPredict:
    """Build a variant on some input columns plus computed ones, with the standard preprocessing."""

    def frame(rows: pd.DataFrame) -> pd.DataFrame:
        result = data_loader.coerce_features(rows)
        result.index = rows.index
        if extra is not None:
            result = pd.concat([result, extra(rows)], axis=1)
        if extra_categorical is not None:
            result = pd.concat([result, extra_categorical(rows)], axis=1)
        return result

    def fit_predict(train: pd.DataFrame, valid: pd.DataFrame) -> NDArray[np.float64]:
        numeric, categorical = _select(columns)
        if extra is not None:
            numeric = [*numeric, *extra(train.head(1)).columns]
        if extra_categorical is not None:
            categorical = [*categorical, *extra_categorical(train.head(1)).columns]
        pipeline = features.build_pipeline(
            features.build_preprocessor(
                features.Family.UNSCALED,
                numeric=numeric,
                categorical=categorical,
                min_frequency=min_frequency,
            ),
            make_model(),
        )
        pipeline.fit(frame(train), train[schema.TARGET])
        return np.asarray(pipeline.predict_proba(frame(valid))[:, 1])

    return fit_predict


def tier_variants() -> list[tuple[str, harness.FitPredict]]:
    """Compare the published inputs with smaller sets that depend less on the snapshot."""
    everything = schema.INPUT_FEATURES
    without_deposit = tuple(column for column in everything if column != "deposit_type")
    without_batch_proxies = tuple(
        column
        for column in without_deposit
        if column not in ("agent", "market_segment", "distribution_channel", "customer_type")
    )
    return [
        ("all_19", predictor(everything)),
        ("by_definition_only", predictor(BY_DEFINITION)),
        (
            "by_definition_without_previous",
            predictor(("hotel", "lead_time", "is_repeated_guest", "days_in_waiting_list")),
        ),
        ("without_deposit", predictor(without_deposit)),
        ("without_deposit_and_channel", predictor(without_batch_proxies)),
    ]


def ablation_variants() -> list[tuple[str, harness.FitPredict]]:
    """Remove one input at a time."""
    variants: list[tuple[str, harness.FitPredict]] = [("all_19", predictor(schema.INPUT_FEATURES))]
    for removed in schema.INPUT_FEATURES:
        kept = tuple(column for column in schema.INPUT_FEATURES if column != removed)
        variants.append((f"without_{removed}", predictor(kept)))
    return variants


def _derived(rows: pd.DataFrame) -> pd.DataFrame:
    guests = rows[["adults", "children", "babies"]].fillna(0).sum(axis=1).clip(lower=1)
    return pd.DataFrame(
        {
            "total_nights": rows["nights"].astype(float),
            "total_guests": guests.astype(float),
            "adr_per_guest": rows["adr"] / guests,
            "has_agent": rows["agent"].notna().astype(float),
        },
        index=rows.index,
    )


def _month(rows: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {"arrival_month": rows["arrival_date_month"].astype("str")}, index=rows.index
    )


def _month_cyclic(rows: pd.DataFrame) -> pd.DataFrame:
    angle = 2 * np.pi * rows[schema.ARRIVAL_DATE].dt.dayofyear / 366
    return pd.DataFrame(
        {"season_sin": np.sin(angle), "season_cos": np.cos(angle)}, index=rows.index
    )


def _previous_binary(rows: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {"any_previous_cancellation": (rows["previous_cancellations"] > 0).astype(float)},
        index=rows.index,
    )


def _columns_from(table: pd.DataFrame) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def take(rows: pd.DataFrame) -> pd.DataFrame:
        return table.loc[rows.index]

    return take


def engineered_variants(rows: pd.DataFrame) -> list[tuple[str, harness.FitPredict]]:
    """Add one group of new variables at a time, computed without looking ahead."""
    everything = schema.INPUT_FEATURES
    no_agent = tuple(column for column in everything if column != "agent")
    no_previous_count = tuple(column for column in everything if column != "previous_cancellations")
    agent_history = settled_history(rows, ("agent_key",))
    agent_recent = settled_history(rows, ("agent_key",), days=_RECENT_DAYS)
    segment_history = settled_history(rows, ("hotel", "market_segment"))
    both = pd.concat([agent_history, segment_history], axis=1)
    return [
        ("all_19", predictor(everything)),
        ("plus_agent_history", predictor(everything, extra=_columns_from(agent_history))),
        ("plus_agent_recent_history", predictor(everything, extra=_columns_from(agent_recent))),
        ("agent_history_instead_of_agent", predictor(no_agent, extra=_columns_from(agent_history))),
        ("plus_segment_history", predictor(everything, extra=_columns_from(segment_history))),
        ("plus_agent_and_segment_history", predictor(everything, extra=_columns_from(both))),
        ("plus_derived_ratios", predictor(everything, extra=_derived)),
        ("plus_arrival_month", predictor(everything, extra_categorical=_month)),
        ("plus_season_cyclic", predictor(everything, extra=_month_cyclic)),
        ("binary_previous_cancellation", predictor(no_previous_count, extra=_previous_binary)),
        ("min_frequency_200", predictor(everything, min_frequency=200)),
        ("min_frequency_10", predictor(everything, min_frequency=10)),
    ]


def _report(table: pd.DataFrame) -> None:
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
                "roc_auc_distinct",
                "log_loss_gap",
            ]
        ]
        .round(4)
        .to_string(index=False),
        flush=True,
    )


def main() -> None:
    """Run the families named on the command line, or all of them, on both schemes."""
    rows = harness.load_rows()
    pd.set_option("display.width", 260, "display.max_columns", 40)
    families = {
        "tiers": tier_variants(),
        "ablation": ablation_variants(),
        "engineered": engineered_variants(rows),
    }
    chosen = set(sys.argv[1:])
    for family, variants in families.items():
        if chosen and family not in chosen:
            continue
        for scheme, windows in (("m12", harness.M12), ("q4", harness.Q4)):
            by_window = harness.run_variants(f"{family} {scheme}", rows, variants, windows)
            table = harness.add_holm(harness.compare_to_reference(by_window, "all_19"))
            harness.save(f"{family}_{scheme}_by_window", by_window)
            harness.save(f"{family}_{scheme}", table)
            print(f"== {family} {scheme} ==")
            _report(table)


if __name__ == "__main__":
    main()
