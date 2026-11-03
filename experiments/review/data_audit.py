"""Audit of the development rows: quality, labels, batches, leakage and stationarity.

It trains no complex model. Run it with `uv run python -m experiments.review.data_audit`.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiments.review import harness
from hotel_cancellation import schema

_LATE_START = pd.Timestamp("2016-10-01")
_MIN_EVENT = 10


def label_coherence(rows: pd.DataFrame) -> dict[str, Any]:
    """Check the target against the status and its date."""
    canceled = rows[rows["reservation_status"] == "Canceled"]
    no_show = rows[rows["reservation_status"] == "No-Show"]
    checkout = rows[rows["reservation_status"] == "Check-Out"]
    departure = checkout[schema.ARRIVAL_DATE] + pd.to_timedelta(checkout["nights"], unit="D")
    return {
        "target_1_is_canceled_or_no_show": bool(
            ((rows[schema.TARGET] == 1) == (rows["reservation_status"] != "Check-Out")).all()
        ),
        "canceled_rows": len(canceled),
        "no_show_rows": len(no_show),
        "no_show_share_of_positives": len(no_show) / int(rows[schema.TARGET].sum()),
        "canceled_after_arrival": int(
            (canceled["status_date"] > canceled[schema.ARRIVAL_DATE]).sum()
        ),
        "canceled_before_booking": int((canceled["status_date"] < canceled["booking_date"]).sum()),
        "canceled_status_before_data_start": int(
            (canceled["status_date"] < pd.Timestamp("2015-07-01")).sum()
        ),
        "checkout_status_equals_departure": float((checkout["status_date"] == departure).mean()),
    }


def date_coherence(rows: pd.DataFrame) -> pd.DataFrame:
    """Find which pair of weekdays explains `stays_in_weekend_nights`, using every row."""
    arrival_weekday = rows[schema.ARRIVAL_DATE].dt.weekday.to_numpy()
    nights = rows["nights"].to_numpy()
    observed = rows["stays_in_weekend_nights"].to_numpy()
    table = []
    for first in range(7):
        pair = {first, (first + 1) % 7}
        counts = np.zeros(7, dtype=int)
        for offset in range(7):
            counts[offset] = sum((offset + day) % 7 in pair for day in range(7))
        full_weeks, rest = np.divmod(nights, 7)
        expected = full_weeks * 2
        expected = expected + np.array(
            [
                sum((weekday + day) % 7 in pair for day in range(int(extra)))
                for weekday, extra in zip(arrival_weekday, rest, strict=True)
            ]
        )
        table.append(
            {
                "first_weekend_day_in_dataset_dates": first,
                "weekday_name": pd.Timestamp("2024-01-01").day_name()
                if first == 0
                else (pd.Timestamp("2024-01-01") + pd.Timedelta(days=first)).day_name(),
                "rows_coherent": float((expected == observed).mean()),
            }
        )
    return pd.DataFrame(table)


def monthly_coverage(rows: pd.DataFrame) -> pd.DataFrame:
    """Count rows and stays per month and hotel, plus the rooms occupied per night."""
    table = (
        rows.assign(month=rows[schema.ARRIVAL_DATE].dt.to_period("M").astype("str"))
        .groupby(["hotel", "month"])
        .agg(
            rows=(schema.TARGET, "size"),
            cancellation_rate=(schema.TARGET, "mean"),
            stays=(schema.TARGET, lambda values: int((values == 0).sum())),
            non_refund_share=(
                "deposit_type",
                lambda values: float((values == "Non Refund").mean()),
            ),
        )
        .reset_index()
    )
    stayed = rows[rows[schema.TARGET] == 0]
    occupied = []
    for hotel, group in stayed.groupby("hotel"):
        nights = np.repeat(group[schema.ARRIVAL_DATE].to_numpy(), group["nights"].to_numpy())
        offsets = np.concatenate([np.arange(count) for count in group["nights"].to_numpy()])
        dates = pd.to_datetime(nights) + pd.to_timedelta(offsets, unit="D")
        per_day = pd.Series(1, index=dates).groupby(level=0).sum()
        days = pd.DatetimeIndex(per_day.index)
        kept = per_day[(days >= pd.Timestamp("2015-07-01")) & (days < pd.Timestamp("2017-01-01"))]
        months = pd.DatetimeIndex(kept.index).to_period("M").astype("str")
        by_month = kept.groupby(months).agg(["mean", "max"])
        by_month = by_month.rename(columns={"mean": "occupied_mean", "max": "occupied_max"})
        occupied.append(by_month.reset_index(names="month").assign(hotel=hotel))
    return table.merge(pd.concat(occupied), on=["hotel", "month"], how="left")


def batch_structure(rows: pd.DataFrame) -> dict[str, Any]:
    """Describe the booking batches and the cancellation events."""
    batches = rows.groupby("batch_id").agg(
        size=(schema.TARGET, "size"), rate=(schema.TARGET, "mean")
    )
    canceled = rows[rows["reservation_status"] == "Canceled"]
    keys = ["hotel", "status_date", "agent_key", "market_segment", "deposit_type"]
    events = canceled.groupby(keys, sort=False).size().rename("rows")
    big_events = events[events >= _MIN_EVENT]
    by_deposit = (
        big_events.groupby(level="deposit_type").sum() / canceled.groupby("deposit_type").size()
    )
    purity = batches["rate"].where(batches["rate"] > 0.5, 1 - batches["rate"])
    large = batches["size"] >= _MIN_EVENT
    sizes = batches["size"].to_numpy(dtype=float)
    # One-way ANOVA estimator: singleton batches add no evidence of agreement inside a batch.
    total, groups = sizes.sum(), sizes.size
    grand_mean = float(rows[schema.TARGET].mean())
    between = float((sizes * (batches["rate"].to_numpy() - grand_mean) ** 2).sum() / (groups - 1))
    within = float(
        (sizes * batches["rate"].to_numpy() * (1 - batches["rate"].to_numpy())).sum()
        / (total - groups)
    )
    typical_size = (total - (sizes**2).sum() / total) / (groups - 1)
    icc = (between - within) / (between + (typical_size - 1) * within)
    design_effect = 1 + (float((sizes**2).sum() / total) - 1) * icc
    return {
        "batches": len(batches),
        "rows": len(rows),
        "rows_in_batches_5_or_more": float(
            batches.loc[batches["size"] >= 5, "size"].sum() / len(rows)
        ),
        "rows_in_batches_10_or_more": float(batches.loc[large, "size"].sum() / len(rows)),
        "rows_in_batches_30_or_more": float(
            batches.loc[batches["size"] >= 30, "size"].sum() / len(rows)
        ),
        "largest_batch": int(batches["size"].max()),
        "label_purity_batches_10_or_more": float(
            np.average(purity[large], weights=batches.loc[large, "size"])
        ),
        "intraclass_correlation": float(icc),
        "mean_batch_size_seen_by_a_row": float((sizes**2).sum() / total),
        "design_effect": float(design_effect),
        "effective_rows": float(total / design_effect),
        "cancellation_events": len(events),
        "cancellations_in_events_10_or_more": float(big_events.sum() / len(canceled)),
        "cancellations_in_events_50_or_more": float(events[events >= 50].sum() / len(canceled)),
        "by_deposit_share_in_events_10_or_more": {
            str(key): float(value) for key, value in by_deposit.items()
        },
        "largest_events": [
            {"key": str(key), "rows": int(value)}
            for key, value in events.sort_values(ascending=False).head(6).items()
        ],
    }


def previous_cancellations(rows: pd.DataFrame) -> pd.DataFrame:
    """Compare `previous_cancellations` by quarter, deposit and batch size."""
    flagged = rows.assign(
        quarter=rows[schema.ARRIVAL_DATE].dt.to_period("Q").astype("str"),
        any_previous=rows["previous_cancellations"] > 0,
        in_batch=np.where(rows["batch_size"] >= _MIN_EVENT, "batch_10_or_more", "smaller"),
        deposit=np.where(rows["deposit_type"] == "Non Refund", "Non Refund", "other"),
    )
    return (
        flagged.groupby(["quarter", "deposit", "in_batch", "any_previous"])
        .agg(rows=(schema.TARGET, "size"), cancellation_rate=(schema.TARGET, "mean"))
        .reset_index()
    )


def _encode_as_target_rate(train: pd.Series, y_train: pd.Series, values: pd.Series) -> pd.Series:
    prior = float(y_train.mean())
    rates = y_train.groupby(train.astype("str")).mean()
    return values.astype("str").map(rates).fillna(prior).astype(float)


def univariate_power(rows: pd.DataFrame) -> pd.DataFrame:
    """Measure what each raw column alone says about the target on 2016Q4, learned before it.

    Numbers are used as they are, in whichever direction training says. Categories get the
    cancellation rate seen in training.
    """
    train = rows[rows[schema.ARRIVAL_DATE] < _LATE_START]
    valid = rows[rows[schema.ARRIVAL_DATE] >= _LATE_START]
    y_train, y_valid = train[schema.TARGET], valid[schema.TARGET]
    no_deposit = (valid["deposit_type"] == "No Deposit").to_numpy()
    table = []
    for column in schema.RAW_COLUMNS:
        if column in (schema.TARGET, "reservation_status_date"):
            continue
        if pd.api.types.is_numeric_dtype(rows[column]) and rows[column].nunique() > 40:
            filled_train = train[column].fillna(train[column].median())
            sign = np.sign(np.corrcoef(filled_train, y_train)[0, 1]) or 1.0
            scores = (valid[column].fillna(train[column].median()) * sign).to_numpy()
        else:
            scores = _encode_as_target_rate(train[column], y_train, valid[column]).to_numpy()
        table.append(
            {
                "column": column,
                "role": "input"
                if column in schema.INPUT_FEATURES
                else "forbidden"
                if column in schema.FORBIDDEN_FEATURES
                else "other",
                "auc_2016q4": float(roc_auc_score(y_valid, scores)),
                "auc_2016q4_no_deposit": float(
                    roc_auc_score(y_valid[no_deposit], scores[no_deposit])
                ),
            }
        )
    return pd.DataFrame(table).sort_values("auc_2016q4", ascending=False)


def fold_overlap(rows: pd.DataFrame) -> pd.DataFrame:
    """Measure, per window, what validation rows share with training."""
    hashes = harness.input_hash(rows)
    table = []
    for window in harness.Q4:
        dates = rows[schema.ARRIVAL_DATE]
        train = dates < window.train_end
        valid = (dates >= window.start) & (dates < window.end)
        train_batches = set(rows.loc[train, "batch_id"])
        booked_before = rows.loc[valid, "booking_date"] < window.start
        table.append(
            {
                "window": window.name,
                "train_rows": int(train.sum()),
                "valid_rows": int(valid.sum()),
                "same_inputs_in_train": float(hashes[valid].isin(hashes[train]).mean()),
                "same_batch_in_train": float(
                    rows.loc[valid, "batch_id"].isin(train_batches).mean()
                ),
                "booked_before_cut": float(booked_before.mean()),
                "train_rows_booked_after_valid_median_booking": float(
                    (
                        rows.loc[train, "booking_date"] > rows.loc[valid, "booking_date"].median()
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(table)


def label_conflicts(rows: pd.DataFrame) -> dict[str, float]:
    """Give the error that no model on these 19 variables can avoid among repeated inputs."""
    hashes = harness.input_hash(rows)
    groups = rows.assign(hash=hashes.to_numpy()).groupby("hash")[schema.TARGET].agg(["size", "sum"])
    repeated = groups[groups["size"] >= 2]
    rate = repeated["sum"] / repeated["size"]
    deduped = rows.assign(hash=hashes.to_numpy())
    with_target = deduped.drop_duplicates(["hash", schema.TARGET])
    inputs_only = deduped.drop_duplicates("hash")
    return {
        "distinct_input_vectors": float(len(groups)),
        "share_of_rows_in_repeated_vectors": float(repeated["size"].sum() / len(rows)),
        "share_of_rows_in_vectors_with_both_labels": float(
            repeated.loc[(rate > 0) & (rate < 1), "size"].sum() / len(rows)
        ),
        "majority_vote_error_within_repeated": float(
            np.minimum(repeated["sum"], repeated["size"] - repeated["sum"]).sum()
            / repeated["size"].sum()
        ),
        "rows_unique_with_target_in_key": float(len(with_target)),
        "rows_unique_inputs_only": float(len(inputs_only)),
        "prevalence_unique_with_target_in_key": float(with_target[schema.TARGET].mean()),
        "prevalence_unique_inputs_only": float(inputs_only[schema.TARGET].mean()),
    }


def snapshot_fingerprints(rows: pd.DataFrame) -> pd.DataFrame:
    """Compare inputs that could be final values between stays and cancellations, by hotel."""
    no_deposit = rows[rows["deposit_type"] == "No Deposit"]
    cents = (no_deposit["adr"] * 100).round().astype(int) % 100
    decimals = np.where(cents == 0, 0, np.where(cents % 10 == 0, 1, 2))
    table = []
    for hotel, group in no_deposit.assign(decimals=decimals).groupby("hotel"):
        actual = group[schema.TARGET]
        for column in ("decimals", "adr", "nights", "adults", "children", "lead_time"):
            values = group[column].fillna(0)
            table.append(
                {
                    "hotel": hotel,
                    "column": column,
                    "mean_stayed": float(values[actual == 0].mean()),
                    "mean_canceled": float(values[actual == 1].mean()),
                    "auc_as_score": float(roc_auc_score(actual, values)),
                }
            )
    return pd.DataFrame(table)


def main() -> None:
    """Write every table of the audit to `outputs/review/`."""
    rows = harness.load_rows()
    coherence = date_coherence(rows)
    monthly = monthly_coverage(rows)
    previous = previous_cancellations(rows)
    power = univariate_power(rows)
    overlap = fold_overlap(rows)
    fingerprints = snapshot_fingerprints(rows)
    harness.save("audit_date_coherence", coherence)
    harness.save("audit_monthly_coverage", monthly)
    harness.save("audit_previous_cancellations", previous)
    harness.save("audit_univariate_power", power)
    harness.save("audit_fold_overlap", overlap)
    harness.save("audit_snapshot_fingerprints", fingerprints)

    import json

    summary = {
        "labels": label_coherence(rows),
        "batches": batch_structure(rows),
        "conflicts": label_conflicts(rows),
    }
    (harness.OUTPUT_DIR / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
    )
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 200)
    print(json.dumps(summary, indent=1, default=str))
    print(coherence.round(4).to_string(index=False))
    print(monthly.round(3).to_string(index=False))
    print(power.round(4).to_string(index=False))
    print(overlap.round(4).to_string(index=False))
    print(fingerprints.round(4).to_string(index=False))
    print(previous[previous["any_previous"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
