"""Tests of raw data loading, contract, cleaning, coercion and temporal splits."""

import gzip
import hashlib
import itertools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hotel_cancellation import data_loader, schema

RawFrameFactory = Callable[..., pd.DataFrame]


def _write_gzip_csv(df: pd.DataFrame, path: Path, *, level: int = 9) -> str:
    """Write the frame like the real dataset and return the hash of the uncompressed CSV."""
    content = df.to_csv(index=False, na_rep="NULL", lineterminator="\n").encode()
    path.write_bytes(gzip.compress(content, compresslevel=level))
    return hashlib.sha256(content).hexdigest()


def test_load_raw_returns_the_rows_of_a_file_with_the_expected_hash(
    raw_like_df: pd.DataFrame, tmp_path: Path
) -> None:
    path = tmp_path / "raw.csv.gz"
    digest = _write_gzip_csv(raw_like_df, path)

    loaded = data_loader.load_raw(path, expected_sha256=digest)

    assert loaded.shape == raw_like_df.shape
    assert list(loaded.columns) == list(raw_like_df.columns)


def test_load_raw_hashes_the_content_and_not_the_compression(
    raw_like_df: pd.DataFrame, tmp_path: Path
) -> None:
    path = tmp_path / "raw.csv.gz"
    digest = _write_gzip_csv(raw_like_df, path, level=1)
    assert _write_gzip_csv(raw_like_df, path, level=9) == digest

    assert len(data_loader.load_raw(path, expected_sha256=digest)) == len(raw_like_df)


def test_load_raw_rejects_a_file_with_another_hash(
    raw_like_df: pd.DataFrame, tmp_path: Path
) -> None:
    path = tmp_path / "raw.csv.gz"
    digest = _write_gzip_csv(raw_like_df, path)

    with pytest.raises(data_loader.DatasetIntegrityError) as error:
        data_loader.load_raw(path, expected_sha256="0" * 64)

    assert digest in str(error.value)
    assert "0" * 64 in str(error.value)
    assert "raw.csv.gz" in str(error.value)


def test_load_raw_reads_agent_as_text_and_null_as_missing(
    raw_like_df: pd.DataFrame, tmp_path: Path
) -> None:
    path = tmp_path / "raw.csv.gz"
    digest = _write_gzip_csv(raw_like_df, path)

    loaded = data_loader.load_raw(path, expected_sha256=digest)

    assert pd.api.types.is_string_dtype(loaded["agent"])
    assert loaded["agent"].iloc[0] == "9"
    assert pd.isna(loaded["agent"].iloc[2])
    assert pd.isna(loaded["company"].iloc[2])
    assert pd.isna(loaded["country"].iloc[2])
    assert "9.0" not in set(loaded["agent"].dropna())


def test_load_raw_reads_children_na_as_missing_number(tmp_path: Path) -> None:
    content = b"children,hotel\nNA,City Hotel\n2,Resort Hotel\n"
    path = tmp_path / "raw.csv.gz"
    path.write_bytes(gzip.compress(content))

    loaded = data_loader.load_raw(path, expected_sha256=hashlib.sha256(content).hexdigest())

    assert pd.api.types.is_float_dtype(loaded["children"])
    assert pd.isna(loaded["children"].iloc[0])
    assert loaded["children"].iloc[1] == 2.0


def test_validate_raw_accepts_dirty_rows_that_the_cleaning_handles(
    raw_like_df: pd.DataFrame,
) -> None:
    data_loader.validate_raw(raw_like_df)


@pytest.mark.parametrize(
    ("override", "expected_in_message"),
    [
        ({"hotel": "Motel"}, ["hotel", "Motel"]),
        ({"hotel": None}, ["hotel", "missing"]),
        ({"reservation_status": "Unknown"}, ["reservation_status", "Unknown"]),
        ({"is_canceled": 2}, ["is_canceled", "0 or 1"]),
        ({"lead_time": "soon"}, ["lead_time", "soon"]),
        ({"adults": -1}, ["adults", "negative"]),
        ({"agent": "9.0"}, ["agent", "9.0"]),
        ({"company": "ACME"}, ["company", "ACME"]),
        ({"arrival_date_month": "Smarch"}, ["arrival_date_month", "Smarch"]),
        ({"arrival_date_month": "February", "arrival_date_day_of_month": 31}, ["arrival date"]),
        ({"reservation_status_date": "yesterday"}, ["reservation_status_date", "yesterday"]),
    ],
)
def test_validate_raw_explains_each_broken_rule(
    raw_frame: RawFrameFactory, override: dict[str, Any], expected_in_message: list[str]
) -> None:
    with pytest.raises(data_loader.DataContractError) as error:
        data_loader.validate_raw(raw_frame({}, override))

    for fragment in expected_in_message:
        assert fragment in str(error.value)


def test_validate_raw_names_missing_and_unexpected_columns(raw_like_df: pd.DataFrame) -> None:
    broken = raw_like_df.drop(columns="lead_time").assign(surprise=1)

    with pytest.raises(data_loader.DataContractError) as error:
        data_loader.validate_raw(broken)

    assert "lead_time" in str(error.value)
    assert "surprise" in str(error.value)


def test_validate_raw_reports_every_violation_at_once(raw_frame: RawFrameFactory) -> None:
    with pytest.raises(data_loader.DataContractError) as error:
        data_loader.validate_raw(raw_frame({"hotel": "Motel", "adults": -1, "is_canceled": 7}))

    assert len(error.value.violations) == 3


def test_validate_raw_accepts_the_columns_declared_as_nullable(raw_frame: RawFrameFactory) -> None:
    data_loader.validate_raw(raw_frame({}, {"children": None, "country": None, "agent": None}))


def test_clean_keeps_only_bookings_with_guests_nights_and_a_plausible_adr(
    raw_like_df: pd.DataFrame,
) -> None:
    cleaned = data_loader.clean(raw_like_df)

    assert list(cleaned["adr"]) == [100.0, 80.0, 100.0, 100.0, 0.0, 5399.99]
    assert list(cleaned.index) == list(range(6))


@pytest.mark.parametrize(
    "guests",
    [
        {"adults": 0, "children": 2.0, "babies": 0},
        {"adults": 0, "children": 0.0, "babies": 1},
        {"adults": 1, "children": None, "babies": 0},
    ],
)
def test_clean_keeps_a_booking_with_any_kind_of_guest(
    raw_frame: RawFrameFactory, guests: dict[str, Any]
) -> None:
    assert len(data_loader.clean(raw_frame(guests))) == 1


def test_clean_counts_missing_children_as_no_guests(raw_frame: RawFrameFactory) -> None:
    assert data_loader.clean(raw_frame({"adults": 0, "children": None, "babies": 0})).empty


def test_clean_keeps_a_stay_of_only_weekend_nights(raw_frame: RawFrameFactory) -> None:
    only_weekend = {"stays_in_week_nights": 0, "stays_in_weekend_nights": 2}
    assert len(data_loader.clean(raw_frame(only_weekend))) == 1


def test_clean_builds_the_arrival_date_from_year_month_and_day(
    raw_frame: RawFrameFactory,
) -> None:
    cleaned = data_loader.clean(raw_frame({"arrival": "2016-02-29"}))

    assert cleaned["arrival_date"].iloc[0] == pd.Timestamp("2016-02-29")
    assert pd.api.types.is_datetime64_any_dtype(cleaned["arrival_date"])


def test_clean_keeps_the_raw_columns_and_does_not_change_its_input(
    raw_like_df: pd.DataFrame,
) -> None:
    before = raw_like_df.copy()

    cleaned = data_loader.clean(raw_like_df)

    pd.testing.assert_frame_equal(raw_like_df, before)
    assert set(cleaned.columns) == {*raw_like_df.columns, "arrival_date"}


def test_coerce_features_returns_only_the_inputs_in_schema_order(
    raw_like_df: pd.DataFrame,
) -> None:
    shuffled = raw_like_df[list(reversed(raw_like_df.columns))]

    coerced = data_loader.coerce_features(shuffled)

    assert tuple(coerced.columns) == schema.INPUT_FEATURES
    assert not set(coerced.columns) & set(schema.FORBIDDEN_FEATURES)
    assert schema.TARGET not in coerced.columns


def test_coerce_features_fixes_numbers_as_float_and_categories_as_text(
    raw_like_df: pd.DataFrame,
) -> None:
    coerced = data_loader.coerce_features(raw_like_df)

    for column in schema.NUMERIC_FEATURES:
        assert coerced[column].dtype == "float64", column
    for column in schema.CATEGORICAL_FEATURES:
        assert coerced[column].dtype == "str", column


def test_coerce_features_turns_none_into_nan_in_a_single_record(
    raw_frame: RawFrameFactory,
) -> None:
    record = raw_frame({})[list(schema.INPUT_FEATURES)].iloc[0].to_dict()
    record.update(agent=None, children=None)

    coerced = data_loader.coerce_features(pd.DataFrame([record]))

    assert coerced["agent"].iloc[0] is not None
    assert pd.isna(coerced["agent"].iloc[0])
    assert np.isnan(coerced["children"].iloc[0])
    assert coerced["adults"].iloc[0] == 2.0


def test_coerce_features_writes_identifiers_without_decimals(raw_frame: RawFrameFactory) -> None:
    frame = raw_frame({"agent": 9.0}, {"agent": 240}, {"agent": "13"}, {"agent": None})

    agents = data_loader.coerce_features(frame)["agent"]

    assert list(agents.iloc[:3]) == ["9", "240", "13"]
    assert pd.isna(agents.iloc[3])


def test_coerce_features_gives_the_same_result_twice_and_keeps_its_input(
    raw_like_df: pd.DataFrame,
) -> None:
    before = raw_like_df.copy()

    once = data_loader.coerce_features(raw_like_df)
    twice = data_loader.coerce_features(once)

    pd.testing.assert_frame_equal(once, twice)
    pd.testing.assert_frame_equal(raw_like_df, before)


def test_coerce_features_names_the_missing_inputs(raw_like_df: pd.DataFrame) -> None:
    with pytest.raises(data_loader.DataContractError, match="lead_time"):
        data_loader.coerce_features(raw_like_df.drop(columns="lead_time"))


def test_coerce_features_rejects_text_in_a_numeric_input(raw_frame: RawFrameFactory) -> None:
    with pytest.raises(data_loader.DataContractError, match=r"lead_time.*soon"):
        data_loader.coerce_features(raw_frame({"lead_time": "soon"}))


def test_split_temporal_cuts_at_the_start_of_2017(raw_frame: RawFrameFactory) -> None:
    cleaned = data_loader.clean(
        raw_frame(
            {"arrival": "2016-12-31"},
            {"arrival": "2017-01-01"},
            {"arrival": "2015-07-01"},
            {"arrival": "2017-08-31"},
        )
    )

    split = data_loader.split_temporal(cleaned)

    assert list(split.development["arrival_date"]) == [
        pd.Timestamp("2015-07-01"),
        pd.Timestamp("2016-12-31"),
    ]
    assert list(split.test["arrival_date"]) == [
        pd.Timestamp("2017-01-01"),
        pd.Timestamp("2017-08-31"),
    ]


def test_split_temporal_keeps_every_row_and_orders_by_arrival(timeline_df: pd.DataFrame) -> None:
    shuffled = data_loader.clean(timeline_df).sample(frac=1, random_state=0)

    split = data_loader.split_temporal(shuffled)

    assert len(split.development) + len(split.test) == len(shuffled)
    for part in split:
        assert part["arrival_date"].is_monotonic_increasing
        assert list(part.index) == list(range(len(part)))


def test_split_temporal_keeps_the_order_of_rows_with_the_same_arrival(
    raw_frame: RawFrameFactory,
) -> None:
    rows = [{"arrival": "2016-05-05", "lead_time": lead_time} for lead_time in (3, 1, 2)]

    split = data_loader.split_temporal(data_loader.clean(raw_frame(*rows)))

    assert list(split.development["lead_time"]) == [3, 1, 2]


def test_split_temporal_rejects_arrivals_after_the_test_period(raw_frame: RawFrameFactory) -> None:
    cleaned = data_loader.clean(raw_frame({"arrival": "2017-09-01"}))

    with pytest.raises(data_loader.DataContractError, match="2017-09-01"):
        data_loader.split_temporal(cleaned)


def test_split_temporal_asks_for_the_arrival_date(raw_like_df: pd.DataFrame) -> None:
    with pytest.raises(data_loader.DataContractError, match="arrival_date"):
        data_loader.split_temporal(raw_like_df)


@pytest.fixture
def development(timeline_df: pd.DataFrame) -> pd.DataFrame:
    return data_loader.split_temporal(data_loader.clean(timeline_df)).development


def test_windows_train_on_the_past_and_validate_on_each_quarter_of_2016(
    development: pd.DataFrame,
) -> None:
    quarters = [
        ("2016-01-01", "2016-04-01"),
        ("2016-04-01", "2016-07-01"),
        ("2016-07-01", "2016-10-01"),
        ("2016-10-01", "2017-01-01"),
    ]
    dates = development["arrival_date"]

    folds = data_loader.expanding_windows(development)

    assert len(folds) == len(quarters)
    for (train, validation), (start, end) in zip(folds, quarters, strict=True):
        assert set(train) == set(np.flatnonzero(dates < start))
        assert set(validation) == set(np.flatnonzero((dates >= start) & (dates < end)))


def test_windows_have_no_overlap_inside_a_fold_or_between_validations(
    development: pd.DataFrame,
) -> None:
    folds = data_loader.expanding_windows(development)

    for train, validation in folds:
        assert not set(train) & set(validation)
    validations = [set(validation) for _, validation in folds]
    assert sum(len(part) for part in validations) == len(set().union(*validations))


def test_windows_expand_with_everything_validated_before(development: pd.DataFrame) -> None:
    folds = data_loader.expanding_windows(development)

    for (train, validation), (next_train, _) in itertools.pairwise(folds):
        assert set(train) | set(validation) == set(next_train)


def test_identical_rows_never_fall_on_both_sides_of_a_window(development: pd.DataFrame) -> None:
    assert development.duplicated().any()
    columns = list(development.columns)

    for train, validation in data_loader.expanding_windows(development):
        shared = development.iloc[train].merge(development.iloc[validation], on=columns)
        assert shared.empty


def test_windows_are_positions_whatever_the_row_order_or_labels(
    development: pd.DataFrame,
) -> None:
    shuffled = development.sample(frac=1, random_state=0)
    scrambled = shuffled.set_axis(range(1000, 1000 + len(shuffled)))
    dates = scrambled["arrival_date"].to_numpy()

    train, validation = data_loader.expanding_windows(scrambled)[0]

    assert (dates[train] < np.datetime64("2016-01-01")).all()
    assert (dates[validation] >= np.datetime64("2016-01-01")).all()
    assert (dates[validation] < np.datetime64("2016-04-01")).all()


def test_windows_never_contain_rows_of_the_test_period(timeline_df: pd.DataFrame) -> None:
    cleaned = data_loader.clean(timeline_df)
    assert (cleaned["arrival_date"] >= "2017-01-01").any()

    folds = data_loader.expanding_windows(cleaned)

    positions = np.concatenate([np.concatenate(fold) for fold in folds])
    assert (cleaned["arrival_date"].iloc[positions] < "2017-01-01").all()


def test_windows_name_the_fold_without_rows(raw_frame: RawFrameFactory) -> None:
    cleaned = data_loader.clean(raw_frame({"arrival": "2016-05-05"}, {"arrival": "2016-11-05"}))

    with pytest.raises(data_loader.DataContractError, match="2016Q1"):
        data_loader.expanding_windows(cleaned)
