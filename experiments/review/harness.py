"""Shared machinery of the second review: windows, metrics, paired comparisons and outputs.

Every function here works on development rows only. See `docs/revision_protocolo.md`.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats  # type: ignore[import-untyped]
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from hotel_cancellation import config
from hotel_cancellation.data import data_loader, schema

SEEDS: tuple[int, ...] = (42, 7, 123, 2024, 31337)
MIN_GAIN = 0.002

BATCH_KEYS: tuple[str, ...] = (
    "hotel",
    "booking_date",
    "agent_key",
    "market_segment",
    "deposit_type",
    "customer_type",
    "reserved_room_type",
)

EVENT_KEYS: tuple[str, ...] = (
    "hotel",
    "status_date",
    "agent_key",
    "market_segment",
    "deposit_type",
)

FitPredict = Callable[[pd.DataFrame, pd.DataFrame], ArrayLike]


@dataclass(frozen=True)
class Window:
    """Validates on arrivals in `[start, end)` and trains on arrivals before `train_end`."""

    name: str
    train_end: pd.Timestamp
    start: pd.Timestamp
    end: pd.Timestamp


def _quarterly() -> tuple[Window, ...]:
    return tuple(
        Window(window.name, window.start, window.start, window.end)
        for window in config.EXPANDING_WINDOWS
    )


def _monthly() -> tuple[Window, ...]:
    starts = pd.date_range("2016-01-01", "2016-12-01", freq="MS")
    return tuple(
        Window(start.strftime("%Y-%m"), start, start, start + pd.offsets.MonthBegin(1))
        for start in starts
    )


Q4: tuple[Window, ...] = _quarterly()
M12: tuple[Window, ...] = _monthly()


def load_rows() -> pd.DataFrame:
    """Load development rows with the derived columns that the analysis uses, never the models."""
    rows = data_loader.load_development()
    if (rows[schema.ARRIVAL_DATE] >= config.TEST_START).any():
        raise ValueError("The review only accepts development rows.")
    return add_derived(rows)


def add_derived(rows: pd.DataFrame) -> pd.DataFrame:
    """Add the booking date, the status date, the nights and the booking batch of each row."""
    booking_date = rows[schema.ARRIVAL_DATE] - pd.to_timedelta(rows["lead_time"], unit="D")
    frame = rows.assign(
        booking_date=booking_date,
        status_date=pd.to_datetime(rows["reservation_status_date"]),
        nights=rows["stays_in_week_nights"] + rows["stays_in_weekend_nights"],
        agent_key=rows["agent"].fillna("none"),
    )
    batch = frame.groupby(list(BATCH_KEYS), sort=False).ngroup()
    return frame.assign(batch_id=batch, batch_size=batch.map(batch.value_counts()))


def cancellation_event_size(rows: pd.DataFrame) -> pd.Series:
    """Count the bookings canceled together with each row, and give 0 to the rest."""
    canceled = rows[rows["reservation_status"] == "Canceled"]
    sizes = canceled.groupby(list(EVENT_KEYS), sort=False)["batch_id"].transform("size")
    return sizes.reindex(rows.index, fill_value=0)


def input_hash(rows: pd.DataFrame) -> pd.Series:
    """Hash the 19 input features of each row."""
    return pd.util.hash_pandas_object(data_loader.coerce_features(rows), index=False)


def distinct_auc(y_true: NDArray[Any], scores: NDArray[Any], hashes: NDArray[Any]) -> float:
    """Compute the ROC-AUC with one observation per distinct input vector.

    The key does not hold the outcome, so the subset does not depend on it. Rows with the same
    inputs have the same score, and each group counts once, split by its share of cancellations.
    """
    frame = pd.DataFrame({"y": y_true, "score": scores, "hash": hashes})
    grouped = frame.groupby("hash", sort=False).agg(rate=("y", "mean"), score=("score", "first"))
    labels = np.r_[np.ones(len(grouped)), np.zeros(len(grouped))]
    weights = np.r_[grouped["rate"].to_numpy(), 1 - grouped["rate"].to_numpy()]
    doubled = np.r_[grouped["score"].to_numpy(), grouped["score"].to_numpy()]
    keep = weights > 0
    if np.unique(labels[keep]).size < 2:
        return float("nan")
    return float(roc_auc_score(labels[keep], doubled[keep], sample_weight=weights[keep]))


def safe_auc(y_true: NDArray[Any], scores: NDArray[Any]) -> float:
    """Return the ROC-AUC, or `NaN` when a subset holds a single class."""
    return float(roc_auc_score(y_true, scores)) if np.unique(y_true).size == 2 else float("nan")


def score_window(rows: pd.DataFrame, probabilities: ArrayLike) -> dict[str, float]:
    """Compute the metrics of the review on the validation rows of a window."""
    actual = rows[schema.TARGET].to_numpy()
    scores = np.asarray(probabilities, dtype=float)
    deposit = rows["deposit_type"].to_numpy()
    hashes = input_hash(rows).to_numpy()
    no_deposit = deposit == "No Deposit"
    return {
        "rows": float(len(rows)),
        "prevalence": float(actual.mean()),
        "roc_auc": safe_auc(actual, scores),
        "roc_auc_no_deposit": safe_auc(actual[no_deposit], scores[no_deposit]),
        "roc_auc_distinct": distinct_auc(actual, scores, hashes),
        "pr_auc": float(average_precision_score(actual, scores)),
        "log_loss": float(log_loss(actual, np.clip(scores, 1e-6, 1 - 1e-6))),
        "brier": float(brier_score_loss(actual, scores)),
    }


def run_variants(
    name: str,
    rows: pd.DataFrame,
    variants: Sequence[tuple[str, FitPredict]],
    windows: Iterable[Window],
) -> pd.DataFrame:
    """Run each variant on each window and return one row per variant and window."""
    results = []
    for window in windows:
        train = rows[rows[schema.ARRIVAL_DATE] < window.train_end]
        dates = rows[schema.ARRIVAL_DATE]
        valid = rows[(dates >= window.start) & (dates < window.end)]
        for variant, fit_predict in variants:
            scores = score_window(valid, fit_predict(train, valid))
            results.append({"variant": variant, "window": window.name, **scores})
            print(f"{name} | {window.name} | {variant}: {scores['roc_auc']:.4f}", flush=True)
    return pd.DataFrame(results)


def compare_to_reference(
    by_window: pd.DataFrame,
    reference: str,
    metrics: Sequence[str] = ("roc_auc", "roc_auc_no_deposit", "roc_auc_distinct", "log_loss"),
) -> pd.DataFrame:
    """Give the mean of each variant and its paired gap with the reference, with a t interval."""
    rows = []
    for variant, group in by_window.groupby("variant", sort=False):
        row: dict[str, Any] = {"variant": variant}
        for metric in metrics:
            table = by_window.pivot_table(index="window", columns="variant", values=metric)
            gaps = (table[variant] - table[reference]).dropna().to_numpy()
            row[metric] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
            if variant == reference:
                continue
            mean = float(gaps.mean())
            error = float(gaps.std(ddof=1) / np.sqrt(gaps.size))
            margin = float(stats.t.ppf(0.975, gaps.size - 1) * error)
            row[f"{metric}_gap"] = mean
            row[f"{metric}_low"] = mean - margin
            row[f"{metric}_high"] = mean + margin
            row[f"{metric}_p"] = (
                float(stats.ttest_1samp(gaps, 0.0).pvalue) if error > 0 else float("nan")
            )
            row[f"{metric}_wins"] = int((gaps > 0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def holm(p_values: Sequence[float]) -> list[float]:
    """Adjust p-values with Holm's step-down method."""
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, (values.size - position) * values[index])
        adjusted[index] = min(running, 1.0)
    return [float(value) for value in adjusted]


def add_holm(
    table: pd.DataFrame, metrics: Sequence[str] = ("roc_auc", "roc_auc_no_deposit")
) -> pd.DataFrame:
    """Add Holm-adjusted p-values within one family of comparisons."""
    result = table.copy()
    for metric in metrics:
        column = f"{metric}_p"
        if column not in result:
            continue
        mask = result[column].notna()
        result.loc[mask, f"{metric}_p_holm"] = holm(result.loc[mask, column].tolist())
    return result


def save(name: str, table: pd.DataFrame) -> None:
    """Store a table in the folder of the review, rounded so that the diffs stay readable."""
    folder = config.get_settings().review_dir
    folder.mkdir(parents=True, exist_ok=True)
    table.round(5).to_csv(folder / f"{name}.csv", index=False, lineterminator="\n")
