"""Availability diagnostic: does a variable keep changing after the booking is confirmed?

The CSV stores the final value of each booking, not its history. A canceled booking stops changing
when it is canceled, so a variable that accumulates must grow with the share of the booking's life
that had elapsed by then. `booking_changes` works as the positive control, because it counts changes
by definition.

The diagnostic is observational: whoever cancels right away may also be a different kind of
customer. `rho_control_within_strata` reduces that risk by comparing bookings of the same hotel,
arrival month and lead time band, but it does not remove it.

Only arrivals before 2017 are used. Run it with `uv run python -m experiments.feature_availability`.
"""

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from hotel_cancellation import config, data_loader, schema

CONTROL_SEGMENT = "Online TA"
CONTROL_VARIABLE = "booking_changes"
ONE_DAY = 1
MIN_CATEGORY_ROWS = 30
QUANTILE_BINS = 5

LIFE_EDGES = (0.0, 0.05, 0.25, 0.5, 0.75, 1.0)
LIFE_LABELS = ("0_5", "5_25", "25_50", "50_75", "75_100")
FIRST_LIFE_BIN, LAST_LIFE_BIN = LIFE_LABELS[0], LIFE_LABELS[-1]

_NOT_VARIABLES = {
    schema.TARGET,
    schema.ARRIVAL_DATE,
    "reservation_status",
    "reservation_status_date",
}


@dataclass(frozen=True)
class Diagnostic:
    """Result of the diagnostic: per variable, per life bin, per cohort, and the context."""

    table: pd.DataFrame
    bins: pd.DataFrame
    cohorts: pd.DataFrame
    summary: dict[str, Any]


def diagnose(development: pd.DataFrame) -> Diagnostic:
    """Measure how each variable changes with the elapsed life of canceled bookings.

    Args:
        development: Cleaned rows with arrivals before the test period.

    Returns:
        The diagnostic of every variable in the frame.

    Raises:
        ValueError: If any row belongs to the test period.
    """
    if (development[schema.ARRIVAL_DATE] >= config.TEST_START).any():
        raise ValueError(f"The diagnostic only accepts arrivals before {config.TEST_START.date()}.")

    canceled = development[development["reservation_status"] == "Canceled"]
    positive_lead = canceled[canceled["lead_time"] > 0]
    with_life = _with_elapsed_life(positive_lead)

    is_control = with_life["market_segment"] == CONTROL_SEGMENT
    is_one_day = with_life["days_to_cancel"] <= ONE_DAY
    strata = _control_strata(with_life[is_control])
    masks = _cohort_masks(with_life)

    variables = [column for column in development.columns if column not in _NOT_VARIABLES]

    rows = []
    bin_rows = []
    cohort_rows = []
    for variable in variables:
        row: dict[str, Any] = {
            "variable": variable,
            "role": _role(variable),
            "kind": _kind(with_life[variable]),
        }
        if row["kind"] == "numeric":
            row.update(_numeric_statistics(with_life, variable, is_control, is_one_day, strata))
            cohort_rows.extend(_cohort_rows(with_life, variable, masks))
        else:
            row.update(_categorical_statistics(with_life, variable, is_control, is_one_day))
        row["cramers_v_all"] = _cramers_v(_as_groups(with_life[variable]), with_life["life_bin"])
        row["cramers_v_control"] = _cramers_v(
            _as_groups(with_life.loc[is_control, variable]), with_life.loc[is_control, "life_bin"]
        )
        rows.append(row)
        bin_rows.extend(_bin_rows(with_life, variable, "all"))
        bin_rows.extend(_bin_rows(with_life[is_control], variable, "control"))

    table = pd.DataFrame(rows)
    cohorts = pd.DataFrame(cohort_rows)
    return Diagnostic(
        table=table,
        bins=pd.DataFrame(bin_rows),
        cohorts=cohorts,
        summary={
            "development_rows": len(development),
            "first_arrival": str(development[schema.ARRIVAL_DATE].min().date()),
            "last_arrival": str(development[schema.ARRIVAL_DATE].max().date()),
            "canceled_rows": len(canceled),
            "excluded_zero_lead_time": len(canceled) - len(positive_lead),
            "excluded_outside_life": len(positive_lead) - len(with_life),
            "rows_used": len(with_life),
            "one_day_rows": int(is_one_day.sum()),
            "control_segment": CONTROL_SEGMENT,
            "control_rows": int(is_control.sum()),
            "control_variable": CONTROL_VARIABLE,
            "flagged": _flagged(table),
            "flagged_by_cohort": _flagged_by_cohort(cohorts),
        },
    )


def _cohort_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Groups in which the measure is repeated, because the effect is not uniform.

    The non-refundable batches are mostly exact duplicates and they cancel together, which hides
    the trend when every canceled row counts. That is why unique rows have their own cohorts.
    """
    unique = ~df.duplicated()
    online = df["market_segment"] == CONTROL_SEGMENT
    city = df["hotel"] == "City Hotel"
    return {
        "all": pd.Series(data=True, index=df.index),
        "unique_rows": unique,
        "no_deposit_unique_rows": unique & (df["deposit_type"] == "No Deposit"),
        "online_ta": online,
        "online_ta_city": online & city,
        "online_ta_resort": online & ~city,
        "city": city,
        "resort": ~city,
    }


def _cohort_rows(
    df: pd.DataFrame, variable: str, masks: dict[str, pd.Series]
) -> list[dict[str, Any]]:
    rows = []
    for cohort, mask in masks.items():
        part = df[mask]
        values = part[variable].astype("float64")
        first = values[part["life_bin"] == FIRST_LIFE_BIN]
        last = values[part["life_bin"] == LAST_LIFE_BIN]
        rows.append(
            {
                "variable": variable,
                "role": _role(variable),
                "cohort": cohort,
                "rows": len(part),
                "rho": _spearman(values, part["life"]),
                f"mean_{FIRST_LIFE_BIN}": float(first.mean()),
                f"mean_{LAST_LIFE_BIN}": float(last.mean()),
                f"nonzero_{FIRST_LIFE_BIN}": float((first > 0).mean()),
                f"nonzero_{LAST_LIFE_BIN}": float((last > 0).mean()),
            }
        )
    return rows


def _flagged_by_cohort(cohorts: pd.DataFrame) -> dict[str, list[str]]:
    """List, per cohort, the variables whose correlation reaches that of the positive control."""
    flagged: dict[str, list[str]] = {}
    for cohort, part in cohorts.groupby("cohort"):
        control = part.loc[part["variable"] == CONTROL_VARIABLE, "rho"]
        others = part[part["variable"] != CONTROL_VARIABLE]
        reaching = [] if control.empty else others[others["rho"] >= control.iloc[0]]["variable"]
        flagged[str(cohort)] = [str(variable) for variable in reaching]
    return flagged


def _with_elapsed_life(canceled: pd.DataFrame) -> pd.DataFrame:
    """Add the days from booking to cancellation and the share of life they represent.

    Rows whose cancellation falls before the booking or after the arrival are dropped.
    """
    lead_days = pd.to_timedelta(canceled["lead_time"], unit="D")
    booking_date = canceled[schema.ARRIVAL_DATE] - lead_days
    cancel_date = pd.to_datetime(canceled["reservation_status_date"], format="%Y-%m-%d")
    days_to_cancel = (cancel_date - booking_date).dt.days
    life = days_to_cancel / canceled["lead_time"]
    inside = life.between(0, 1)
    selected = canceled.loc[inside].assign(
        days_to_cancel=days_to_cancel[inside],
        life=life[inside],
    )
    return selected.assign(
        life_bin=pd.cut(
            selected["life"],
            bins=list(LIFE_EDGES),
            labels=list(LIFE_LABELS),
            include_lowest=True,
            right=False,
        )
        .astype("object")
        .where(selected["life"] < 1, LAST_LIFE_BIN)
        .astype("str")
    ).reset_index(drop=True)


def _control_strata(control: pd.DataFrame) -> pd.Series:
    """Group bookings of the same hotel, arrival month and lead time band."""
    lead_band = pd.qcut(control["lead_time"], QUANTILE_BINS, labels=False, duplicates="drop")
    month = control[schema.ARRIVAL_DATE].dt.month
    return control.groupby([control["hotel"], month, lead_band], observed=True).ngroup()


def _role(variable: str) -> str:
    if variable in schema.INPUT_FEATURES:
        return "input"
    if variable in schema.FORBIDDEN_FEATURES:
        return "forbidden"
    return "other"


def _kind(values: pd.Series) -> str:
    return "numeric" if pd.api.types.is_numeric_dtype(values) else "categorical"


def _spearman(left: pd.Series, right: pd.Series) -> float:
    if left.nunique() < 2 or right.nunique() < 2:
        return float("nan")
    return float(left.rank().corr(right.rank()))


def _spearman_within_strata(values: pd.Series, life: pd.Series, strata: pd.Series) -> float:
    if values.nunique() < 2:
        return float("nan")
    left = values.rank()
    right = life.rank()
    left_residual = left - left.groupby(strata).transform("mean")
    right_residual = right - right.groupby(strata).transform("mean")
    return float(left_residual.corr(right_residual))


def _standardized_difference(later: pd.Series, early: pd.Series) -> float:
    pooled = np.sqrt((later.var() + early.var()) / 2)
    return float((later.mean() - early.mean()) / pooled) if pooled > 0 else 0.0


def _numeric_statistics(
    df: pd.DataFrame,
    variable: str,
    is_control: pd.Series,
    is_one_day: pd.Series,
    strata: pd.Series,
) -> dict[str, float]:
    values = df[variable].astype("float64")
    first = values[df["life_bin"] == FIRST_LIFE_BIN]
    last = values[df["life_bin"] == LAST_LIFE_BIN]
    control = df[is_control]
    return {
        "rho_all": _spearman(values, df["life"]),
        "rho_control": _spearman(values[is_control], control["life"]),
        "rho_control_within_strata": _spearman_within_strata(
            values[is_control], control["life"], strata
        ),
        f"mean_{FIRST_LIFE_BIN}": float(first.mean()),
        f"mean_{LAST_LIFE_BIN}": float(last.mean()),
        f"nonzero_{FIRST_LIFE_BIN}": float((first > 0).mean()),
        f"nonzero_{LAST_LIFE_BIN}": float((last > 0).mean()),
        "mean_one_day": float(values[is_one_day].mean()),
        "mean_after_one_day": float(values[~is_one_day].mean()),
        "nonzero_one_day": float((values[is_one_day] > 0).mean()),
        "nonzero_after_one_day": float((values[~is_one_day] > 0).mean()),
        "smd_one_day": _standardized_difference(values[~is_one_day], values[is_one_day]),
        "smd_one_day_control": _standardized_difference(
            values[is_control & ~is_one_day], values[is_control & is_one_day]
        ),
    }


def _categorical_statistics(
    df: pd.DataFrame, variable: str, is_control: pd.Series, is_one_day: pd.Series
) -> dict[str, float]:
    groups = _as_groups(df[variable])
    is_first = df["life_bin"] == FIRST_LIFE_BIN
    is_last = df["life_bin"] == LAST_LIFE_BIN
    return {
        "tvd_first_last_life": _total_variation(groups[is_first], groups[is_last]),
        "tvd_first_last_life_control": _total_variation(
            groups[is_control & is_first], groups[is_control & is_last]
        ),
        "tvd_one_day": _total_variation(groups[~is_one_day], groups[is_one_day]),
        "tvd_one_day_control": _total_variation(
            groups[is_control & ~is_one_day], groups[is_control & is_one_day]
        ),
    }


def _as_groups(values: pd.Series) -> pd.Series:
    """Turn a variable into a few groups: quantiles for numbers, rare categories together.

    A count that is almost always zero has no useful quantiles, so it becomes zero or not.
    """
    if pd.api.types.is_numeric_dtype(values) and values.nunique() > QUANTILE_BINS + 1:
        quantiles = pd.qcut(values, QUANTILE_BINS, duplicates="drop")
        if quantiles.nunique() < 2:
            return (values > 0).astype("str")
        return quantiles.astype("str").fillna("missing").astype("str")
    text = values.astype("str").fillna("missing")
    counts = text.value_counts()
    rare = counts.index[counts < MIN_CATEGORY_ROWS]
    return text.where(~text.isin(list(rare)), "other")


def _total_variation(left: pd.Series, right: pd.Series) -> float:
    left_shares = left.value_counts(normalize=True)
    right_shares = right.value_counts(normalize=True)
    groups = left_shares.index.union(right_shares.index)
    difference = left_shares.reindex(groups, fill_value=0) - right_shares.reindex(
        groups, fill_value=0
    )
    return float(np.abs(difference.to_numpy()).sum() / 2)


def _cramers_v(left: pd.Series, right: pd.Series) -> float:
    counts = pd.crosstab(left, right).to_numpy(dtype="float64")
    smallest_side = min(counts.shape) - 1
    if smallest_side < 1:
        return float("nan")
    total = counts.sum()
    expected = np.outer(counts.sum(axis=1), counts.sum(axis=0)) / total
    chi_squared = ((counts - expected) ** 2 / expected).sum()
    return float(np.sqrt(chi_squared / (total * smallest_side)))


def _bin_rows(df: pd.DataFrame, variable: str, scope: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    numeric = _kind(df[variable]) == "numeric"
    for label in LIFE_LABELS:
        values = df.loc[df["life_bin"] == label, variable]
        row: dict[str, Any] = {
            "variable": variable,
            "scope": scope,
            "life_bin": label,
            "rows": len(values),
        }
        if numeric:
            row["mean"] = float(values.mean())
            row["nonzero_share"] = float((values > 0).mean())
        else:
            modes = values.value_counts(normalize=True)
            row["modal_category"] = None if modes.empty else str(modes.index[0])
            row["modal_share"] = None if modes.empty else float(modes.iloc[0])
        rows.append(row)
    return rows


def _flagged(table: pd.DataFrame) -> list[str]:
    """List the variables whose signal reaches that of the positive control."""
    control = table[table["variable"] == CONTROL_VARIABLE]
    if control.empty:
        return []
    reference = control.iloc[0]
    numeric = table["kind"] == "numeric"
    reaches_control = (
        numeric
        & (
            (table["rho_control"] >= reference["rho_control"])
            | (table["rho_control_within_strata"] >= reference["rho_control_within_strata"])
        )
    ) | (~numeric & (table["cramers_v_control"] >= reference["cramers_v_control"]))
    flagged = table.loc[reaches_control & (table["variable"] != CONTROL_VARIABLE), "variable"]
    return [str(variable) for variable in flagged]


def main() -> None:
    """Run the diagnostic on the real development rows and store the results."""
    raw = data_loader.load_raw()
    data_loader.validate_raw(raw)
    development = data_loader.split_temporal(data_loader.clean(raw)).development
    result = diagnose(development)

    output_dir = config.get_settings().experiments_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    table = result.table.sort_values("rho_control", ascending=False, na_position="last")
    table.round(4).to_csv(output_dir / "feature_availability.csv", index=False, lineterminator="\n")
    result.bins.round(4).to_csv(
        output_dir / "feature_availability_bins.csv", index=False, lineterminator="\n"
    )
    result.cohorts.round(4).to_csv(
        output_dir / "feature_availability_cohorts.csv", index=False, lineterminator="\n"
    )
    summary = {**result.summary, "dataset_sha256": config.RAW_DATASET_SHA256}
    (output_dir / "feature_availability_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    columns = [
        "variable",
        "role",
        "rho_all",
        "rho_control",
        "rho_control_within_strata",
        "nonzero_0_5",
        "nonzero_75_100",
        "smd_one_day",
        "smd_one_day_control",
        "cramers_v_control",
        "tvd_first_last_life_control",
        "tvd_one_day_control",
    ]
    print(table[columns].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
