"""Raw data loading, contract, cleaning, coercion and temporal splits."""

import gzip
import hashlib
import io
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hotel_cancellation import config
from hotel_cancellation.data import schema

logger = logging.getLogger(__name__)

# Only the markers the CSV really uses: pandas' default list would also swallow other words.
_MISSING_MARKERS = ("NULL", "NA")

_NUMERIC_COLUMNS: tuple[str, ...] = (
    *schema.BINARY_COLUMNS,
    *schema.NON_NEGATIVE_COLUMNS,
    "adr",
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
)
_ARRIVAL_PARTS = {"arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"}
_EXAMPLES_SHOWN = 5


class DatasetIntegrityError(ValueError):
    """The dataset on disk is not the one the project was built on."""


def load_raw(
    path: Path | None = None,
    *,
    expected_sha256: str = config.RAW_DATASET_SHA256,
) -> pd.DataFrame:
    """Read the compressed raw dataset after checking its SHA-256.

    The hash covers the uncompressed CSV, so recompressing the file does not invalidate it.
    `agent` and `company` are read as text so that an identifier never turns into `9.0`.

    Args:
        path: Compressed CSV. Defaults to the location in the settings.
        expected_sha256: Hash of the uncompressed CSV.

    Returns:
        The raw rows, unchanged.

    Raises:
        DatasetIntegrityError: If the content does not match the expected hash.
    """
    source = config.get_settings().raw_dataset_path if path is None else path
    content = gzip.decompress(source.read_bytes())
    actual_sha256 = hashlib.sha256(content).hexdigest()
    if actual_sha256 != expected_sha256:
        raise DatasetIntegrityError(
            f"{source} has SHA-256 {actual_sha256}, but {expected_sha256} was expected."
        )
    return pd.read_csv(
        io.BytesIO(content),
        dtype=dict.fromkeys(schema.ID_COLUMNS, "str"),
        na_values=list(_MISSING_MARKERS),
        keep_default_na=False,
    )


class DataContractError(ValueError):
    """The raw data breaks the contract. `violations` lists every broken rule."""

    def __init__(self, violations: Sequence[str]) -> None:
        """Build the message from every broken rule."""
        self.violations = tuple(violations)
        super().__init__("Data contract violated:\n- " + "\n- ".join(self.violations))


def validate_raw(df: pd.DataFrame) -> None:
    """Check that the raw rows honour the contract, without changing them.

    The rows that `clean` removes (no guests, no nights, extreme `adr`) are valid here: they are
    part of the source. The contract rejects whatever the cleaning could not interpret.

    Args:
        df: Rows as `load_raw` returns them.

    Raises:
        DataContractError: Listing every broken rule at once.
    """
    violations = [
        *_column_violations(df),
        *_missing_value_violations(df),
        *_numeric_violations(df),
        *_category_violations(df),
        *_date_violations(df),
    ]
    if violations:
        raise DataContractError(violations)


def _examples(values: pd.Series) -> str:
    shown = sorted({str(value) for value in values})
    return ", ".join(shown[:_EXAMPLES_SHOWN])


def _column_violations(df: pd.DataFrame) -> list[str]:
    violations = []
    missing = [column for column in schema.RAW_COLUMNS if column not in df.columns]
    if missing:
        violations.append(f"missing columns: {', '.join(missing)}")
    unexpected = [column for column in df.columns if column not in schema.RAW_COLUMNS]
    if unexpected:
        violations.append(f"unexpected columns: {', '.join(map(str, unexpected))}")
    return violations


def _missing_value_violations(df: pd.DataFrame) -> list[str]:
    required = [
        column
        for column in schema.RAW_COLUMNS
        if column in df.columns and column not in schema.NULLABLE_COLUMNS
    ]
    counts = df[required].isna().sum()
    return [f"{column}: {count} missing values" for column, count in counts[counts > 0].items()]


def _numeric_violations(df: pd.DataFrame) -> list[str]:
    violations = []
    for column in _NUMERIC_COLUMNS:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        not_numeric = df.loc[values.isna() & df[column].notna(), column]
        if not not_numeric.empty:
            violations.append(f"{column}: non-numeric values: {_examples(not_numeric)}")
        if column in schema.NON_NEGATIVE_COLUMNS and (values < 0).any():
            violations.append(f"{column}: negative values: {_examples(values[values < 0])}")
        if column in schema.BINARY_COLUMNS and not values.dropna().isin([0, 1]).all():
            other = values[values.notna() & ~values.isin([0, 1])]
            violations.append(f"{column}: values other than 0 or 1: {_examples(other)}")
    return violations


def _category_violations(df: pd.DataFrame) -> list[str]:
    violations = []
    for column, allowed in schema.ALLOWED_VALUES.items():
        if column not in df.columns:
            continue
        values = df[column].dropna()
        unknown = values[~values.isin(list(allowed))]
        if not unknown.empty:
            violations.append(f"{column}: values not allowed: {_examples(unknown)}")
    for column in schema.ID_COLUMNS:
        if column not in df.columns:
            continue
        values = df[column].dropna().astype("str")
        malformed = values[~values.str.fullmatch(schema.ID_PATTERN.pattern)]
        if not malformed.empty:
            violations.append(f"{column}: identifiers must be integers: {_examples(malformed)}")
    return violations


def _date_violations(df: pd.DataFrame) -> list[str]:
    violations = []
    if _ARRIVAL_PARTS.issubset(df.columns):
        invalid = int(_arrival_dates(df).isna().sum())
        if invalid:
            violations.append(f"{invalid} rows do not form a valid arrival date")
    if "reservation_status_date" in df.columns:
        values = df["reservation_status_date"]
        dates = pd.to_datetime(values, format="%Y-%m-%d", errors="coerce")
        invalid_dates = values[dates.isna() & values.notna()]
        if not invalid_dates.empty:
            violations.append(f"reservation_status_date: invalid: {_examples(invalid_dates)}")
    return violations


def _arrival_dates(df: pd.DataFrame) -> pd.Series:
    """Build the arrival date, with `NaT` where the year, month and day do not form a date."""
    months = {name: number for number, name in enumerate(schema.ARRIVAL_MONTHS, start=1)}
    return pd.to_datetime(
        pd.DataFrame(
            {
                "year": df["arrival_date_year"],
                "month": df["arrival_date_month"].map(months),
                "day": df["arrival_date_day_of_month"],
            }
        ),
        errors="coerce",
    )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove impossible bookings and add the arrival date used to split the data.

    A booking is impossible when it has no guests or no nights, or when its `adr` is negative or
    reaches `config.ADR_UPPER_LIMIT`. A missing `children` counts as no children. The raw columns
    stay, forbidden ones included, because the diagnostics need them: only `coerce_features`
    decides what reaches a model.

    Args:
        df: Rows that passed `validate_raw`.

    Returns:
        A new frame with the surviving rows and the `arrival_date` column.
    """
    guests = df[["adults", "children", "babies"]].fillna(0).sum(axis=1)
    nights = df["stays_in_week_nights"] + df["stays_in_weekend_nights"]
    plausible_adr = (df["adr"] >= 0) & (df["adr"] < config.ADR_UPPER_LIMIT)
    cleaned = df.loc[(guests > 0) & (nights > 0) & plausible_adr].reset_index(drop=True)
    logger.info("Cleaning kept %d of %d rows", len(cleaned), len(df))
    return cleaned.assign(**{schema.ARRIVAL_DATE: _arrival_dates(cleaned)})


def coerce_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the input features only, in a fixed order and with fixed types.

    This is the only way into a model, both when training and when predicting. Numbers become
    `float64` and categories become `str`; `None` becomes `NaN` in both.

    Args:
        df: Any frame holding the input features, extra columns included.

    Returns:
        A new frame with exactly `schema.INPUT_FEATURES`.

    Raises:
        DataContractError: If an input is missing or a numeric input holds text.
    """
    missing = [column for column in schema.INPUT_FEATURES if column not in df.columns]
    if missing:
        raise DataContractError([f"missing input features: {', '.join(missing)}"])

    coerced = {}
    violations = []
    for column in schema.NUMERIC_FEATURES:
        values = pd.to_numeric(df[column], errors="coerce")
        not_numeric = df.loc[values.isna() & df[column].notna(), column]
        if not not_numeric.empty:
            violations.append(f"{column}: non-numeric values: {_examples(not_numeric)}")
        coerced[column] = values.astype("float64")
    if violations:
        raise DataContractError(violations)

    for column in schema.CATEGORICAL_FEATURES:
        coerced[column] = _as_text(df[column])
    return pd.DataFrame(coerced, columns=list(schema.INPUT_FEATURES))


def _as_text(values: pd.Series) -> pd.Series:
    if isinstance(values.dtype, pd.StringDtype):
        return values.astype("str")
    return values.map(_text_or_missing).astype("str")


def _text_or_missing(value: object) -> str | float:
    """Write a value as text. A whole float is an identifier read as a number: 9.0 becomes 9."""
    if pd.isna(value):  # type: ignore[call-overload]
        return float("nan")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


class TemporalSplit(NamedTuple):
    """Development and test rows, each ordered by arrival date."""

    development: pd.DataFrame
    test: pd.DataFrame


def split_temporal(df: pd.DataFrame) -> TemporalSplit:
    """Separate the development arrivals from the test period (January to August 2017).

    Rows are ordered by arrival because the last rows must be the most recent ones: Keras takes
    its `validation_split` from the end.

    Args:
        df: Rows returned by `clean`.

    Returns:
        Both parts with every input row, in stable arrival order and with a fresh index.

    Raises:
        DataContractError: If there is no arrival date or a row arrives after the test period.
    """
    dates = _arrival_date_column(df)
    if (dates >= config.TEST_END).any():
        raise DataContractError(
            [
                f"arrival on {dates.max().date()} is after the test period, "
                f"which ends before {config.TEST_END.date()}"
            ]
        )
    ordered = df.sort_values(schema.ARRIVAL_DATE, kind="stable")
    is_test = ordered[schema.ARRIVAL_DATE] >= config.TEST_START
    return TemporalSplit(
        development=ordered.loc[~is_test].reset_index(drop=True),
        test=ordered.loc[is_test].reset_index(drop=True),
    )


def expanding_windows(df: pd.DataFrame) -> list[tuple[NDArray[np.intp], NDArray[np.intp]]]:
    """Build the expanding validation folds as row positions, ready for `cv=` in a search.

    Each quarter of 2016 validates and trains with every earlier arrival. Rows identical in every
    column share their arrival date, so they can never end up on both sides of a fold. Rows with
    the same 19 inputs can, because the arrival date is not one of them.

    Args:
        df: Rows with an arrival date, in any order. Test rows are ignored.

    Returns:
        One `(train, validation)` pair of positions per window in `config.EXPANDING_WINDOWS`.

    Raises:
        DataContractError: If there is no arrival date or a fold would have no rows.
    """
    dates = _arrival_date_column(df)
    folds = []
    for window in config.EXPANDING_WINDOWS:
        train = np.flatnonzero(dates < window.start)
        validation = np.flatnonzero((dates >= window.start) & (dates < window.end))
        if train.size == 0 or validation.size == 0:
            raise DataContractError([f"window {window.name} has no rows on one of its sides"])
        folds.append((train, validation))
    return folds


def _arrival_date_column(df: pd.DataFrame) -> pd.Series:
    if schema.ARRIVAL_DATE not in df.columns:
        raise DataContractError([f"missing column {schema.ARRIVAL_DATE}: run `clean` first"])
    return df[schema.ARRIVAL_DATE]
