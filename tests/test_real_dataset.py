"""Checks on the real dataset. Rows of 2017 are only seen by the contract and the row counts."""

import pandas as pd
import pytest

from hotel_cancellation.data import data_loader, schema

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return data_loader.load_raw()


@pytest.fixture(scope="module")
def cleaned(raw: pd.DataFrame) -> pd.DataFrame:
    return data_loader.clean(raw)


def test_real_dataset_honours_the_contract(raw: pd.DataFrame) -> None:
    data_loader.validate_raw(raw)
    assert raw.shape == (119_390, 32)
    assert tuple(raw.columns) == schema.RAW_COLUMNS


def test_cleaning_leaves_the_expected_rows(cleaned: pd.DataFrame) -> None:
    assert len(cleaned) == 118_563


def test_allowed_values_are_exactly_the_values_in_the_data(raw: pd.DataFrame) -> None:
    for column, allowed in schema.ALLOWED_VALUES.items():
        assert set(raw[column].dropna()) == allowed, column


def test_every_allowed_value_already_appears_before_2017(raw: pd.DataFrame) -> None:
    before_2017 = raw[raw["arrival_date_year"] < 2017]
    for column, allowed in schema.ALLOWED_VALUES.items():
        assert set(before_2017[column].dropna()) == allowed, column


def test_only_the_declared_columns_hold_missing_values(raw: pd.DataFrame) -> None:
    with_nulls = set(raw.columns[raw.isna().any()])
    assert with_nulls == schema.NULLABLE_COLUMNS


def test_identifiers_stay_integers_written_as_text(cleaned: pd.DataFrame) -> None:
    agents = data_loader.coerce_features(cleaned)["agent"].dropna()
    assert agents.str.fullmatch(schema.ID_PATTERN.pattern).all()
    assert "9" in set(agents)
    assert "9.0" not in set(agents)


def test_split_and_windows_cover_the_clean_rows(cleaned: pd.DataFrame) -> None:
    split = data_loader.split_temporal(cleaned)

    assert len(split.development) + len(split.test) == len(cleaned)
    assert split.development["arrival_date"].max() < pd.Timestamp("2017-01-01")
    assert split.test["arrival_date"].min() == pd.Timestamp("2017-01-01")
    assert split.test["arrival_date"].max() == pd.Timestamp("2017-08-31")
    assert len(data_loader.expanding_windows(split.development)) == 4
