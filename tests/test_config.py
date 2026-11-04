"""Tests of paths, protocol dates and validation windows."""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from hotel_cancellation import config


def _timestamps(*days: str) -> list[pd.Timestamp]:
    return [pd.Timestamp(day) for day in days]


def test_importing_config_creates_nothing_on_disk(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "HOTEL_ML_DATA_DIR": str(tmp_path / "data"),
        "HOTEL_ML_MODELS_DIR": str(tmp_path / "models"),
        "HOTEL_ML_OUTPUTS_DIR": str(tmp_path / "outputs"),
    }
    code = "from hotel_cancellation import config; config.get_settings()"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=tmp_path, env=env)  # noqa: S603
    assert list(tmp_path.iterdir()) == []


def test_settings_read_paths_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOTEL_ML_DATA_DIR", str(tmp_path))
    settings = config.get_settings()
    assert settings.data_dir == tmp_path
    assert settings.raw_dataset_path == tmp_path / "raw" / "dataset_practica_final.csv.gz"


def test_default_paths_point_to_the_repository(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = config.get_settings()
    root = Path(__file__).resolve().parents[1]
    assert settings.data_dir == root / "data"
    assert settings.models_dir == root / "models"
    assert settings.outputs_dir == root / "outputs"
    assert settings.experiments_dir == root / "outputs" / "experiments"
    assert settings.review_dir == root / "outputs" / "review"


def test_test_period_is_january_to_august_2017() -> None:
    period = [config.TEST_START, config.TEST_END]
    assert period == _timestamps("2017-01-01", "2017-09-01")


def test_expanding_windows_validate_each_quarter_of_2016() -> None:
    assert [window.name for window in config.EXPANDING_WINDOWS] == [
        "2016Q1",
        "2016Q2",
        "2016Q3",
        "2016Q4",
    ]
    assert [window.start for window in config.EXPANDING_WINDOWS] == _timestamps(
        "2016-01-01", "2016-04-01", "2016-07-01", "2016-10-01"
    )
    assert [window.end for window in config.EXPANDING_WINDOWS] == _timestamps(
        "2016-04-01", "2016-07-01", "2016-10-01", "2017-01-01"
    )


def test_no_window_reaches_the_test_period() -> None:
    assert all(window.end <= config.TEST_START for window in config.EXPANDING_WINDOWS)
    assert config.CALIBRATION_SELECTION.evaluation_end == config.TEST_START
    assert config.CHAMPION_CUT.calibration_end == config.TEST_START


def test_calibration_choice_fits_on_third_quarter_and_measures_on_fourth() -> None:
    cut = config.CALIBRATION_SELECTION
    assert cut.base_end == pd.Timestamp("2016-07-01")
    assert cut.calibration_end == pd.Timestamp("2016-10-01")
    assert cut.evaluation_end == pd.Timestamp("2017-01-01")


def test_champion_trains_before_october_and_calibrates_on_fourth_quarter() -> None:
    cut = config.CHAMPION_CUT
    assert cut.base_end == pd.Timestamp("2016-10-01")
    assert cut.calibration_end == pd.Timestamp("2017-01-01")
    assert cut.evaluation_end is None
