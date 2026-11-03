"""Full rehearsal of `evaluate` on the real models, with a window of 2016 as a pseudo test.

It only needs arrivals before 2017. The figures do not count: the calibrator of the champion was
fitted on 2016T4. It proves that the single execution will not break halfway on real rows.
"""

from pathlib import Path

import pandas as pd
import pytest

from hotel_cancellation import champion, cli, config
from hotel_cancellation.data import schema
from hotel_cancellation.training import pipeline

pytestmark = pytest.mark.slow

TABLES = (
    "metrics.csv",
    "segments.csv",
    "monthly.csv",
    "bootstrap.csv",
    "psi.csv",
    "scenario_special_requests.csv",
    "predictions.csv.gz",
    "record.json",
)


@pytest.fixture
def real_settings() -> config.Settings:
    settings = config.get_settings()
    if not settings.candidates_registry_path.exists() or not settings.candidates_dir.exists():
        pytest.skip("The candidates are not versioned: this machine does not have them.")
    return settings


@pytest.mark.parametrize("window", ["2016Q4", "2016Q2"])
def test_rehearsal_runs_on_real_rows_and_never_reads_the_test(
    window: str, real_settings: config.Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail() -> None:
        raise AssertionError("the rehearsal read the test rows")

    monkeypatch.setattr(pipeline, "load_test", fail)
    champion_before = (real_settings.champion_dir / champion.MODEL_FILE).read_bytes()
    record = real_settings.final_evaluation_record
    record_before = record.read_bytes() if record.exists() else None

    folder = pipeline.rehearse(
        settings=real_settings, window=window, output_dir=tmp_path, replicates=20
    )

    assert all((folder / table).exists() for table in TABLES)
    assert (record.read_bytes() if record.exists() else None) == record_before
    assert (real_settings.champion_dir / champion.MODEL_FILE).read_bytes() == champion_before
    predictions = pd.read_csv(folder / "predictions.csv.gz")
    assert predictions[schema.ARRIVAL_DATE].max() < str(config.TEST_START.date())
    metrics = pd.read_csv(folder / "metrics.csv")
    assert metrics.loc[metrics["subset"] == "all", "rows"].min() > 0
    psi = pd.read_csv(folder / "psi.csv").set_index("feature")["psi"]
    assert set(psi.index) == set(schema.INPUT_FEATURES)


def test_rehearsal_refuses_a_window_that_is_not_of_the_protocol(
    real_settings: config.Settings, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="2015Q1"):
        pipeline.rehearse(settings=real_settings, window="2015Q1", output_dir=tmp_path)


def test_rehearsal_command_writes_where_it_is_told(
    real_settings: config.Settings, tmp_path: Path
) -> None:
    arguments = ["rehearse", "--replicates", "5", "--output", str(tmp_path)]
    assert cli.main(arguments) == 0
    assert (tmp_path / "final_evaluation" / "record.json").exists()
