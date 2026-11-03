"""Paths, protocol dates and validation windows.

Importing this module has no side effects: it creates no folders and reads no environment
variables until `get_settings` is called. Every period is half open, `[start, end)`.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RAW_DATASET_FILENAME = "dataset_practica_final.csv.gz"
# Hash of the uncompressed CSV, so it does not depend on how the file was compressed.
RAW_DATASET_SHA256 = "d10cdeb07de9309007f528d6499a2c0467ae7a6b287b80ffd759e11e61801639"

ADR_UPPER_LIMIT = 5400.0


class Settings(BaseSettings):
    """Filesystem locations, which `HOTEL_ML_*` environment variables can override."""

    model_config = SettingsConfigDict(env_prefix="HOTEL_ML_", env_file=".env", extra="ignore")

    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = DEFAULT_MODELS_DIR
    outputs_dir: Path = DEFAULT_OUTPUTS_DIR
    mlflow_tracking_uri: str = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    mlflow_artifacts_dir: Path = PROJECT_ROOT / "mlartifacts"
    api_url: str = "http://127.0.0.1:8000"

    @property
    def raw_dataset_path(self) -> Path:
        """Location of the compressed raw dataset."""
        return self.data_dir / "raw" / RAW_DATASET_FILENAME

    @property
    def experiments_dir(self) -> Path:
        """Folder where the experiments write their results."""
        return self.outputs_dir / "experiments"

    @property
    def validation_dir(self) -> Path:
        """Folder with the validation metrics of the search and the final training."""
        return self.outputs_dir / "validation"

    @property
    def final_evaluation_dir(self) -> Path:
        """Results of the single execution on the test."""
        return self.outputs_dir / "final_evaluation"

    @property
    def final_evaluation_record(self) -> Path:
        """Record of when the test was executed. Its existence stops a second execution."""
        return self.final_evaluation_dir / "record.json"

    @property
    def figures_dir(self) -> Path:
        """Figures of the report."""
        return self.outputs_dir / "figures"

    @property
    def interpretation_dir(self) -> Path:
        """Importance of the variables."""
        return self.outputs_dir / "interpretation"

    @property
    def audit_dir(self) -> Path:
        """Audit of the inflated figures."""
        return self.outputs_dir / "audit"

    @property
    def business_dir(self) -> Path:
        """Value of the champion under the assumed costs."""
        return self.outputs_dir / "business"

    @property
    def best_params_path(self) -> Path:
        """Versioned result of the search, which training reads."""
        return self.models_dir / "best_params.json"

    @property
    def champion_dir(self) -> Path:
        """Versioned package of the served model."""
        return self.models_dir / "champion"

    @property
    def candidates_dir(self) -> Path:
        """Every final model, kept out of git, to compare them on the test."""
        return self.models_dir / "candidates"

    @property
    def scenario_model_path(self) -> Path:
        """Model of the informative scenario, kept out of git."""
        return self.models_dir / "scenario" / "special_requests.joblib"

    @property
    def candidates_registry_path(self) -> Path:
        """Versioned hashes of the models that git does not keep."""
        return self.models_dir / "candidates.json"


def get_settings() -> Settings:
    """Read the settings from the environment."""
    return Settings()


@dataclass(frozen=True)
class ValidationWindow:
    """Validates on `[start, end)` and trains on every arrival before `start`."""

    name: str
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class TemporalCut:
    """Trains the base model before `base_end` and the calibrator on `[base_end, calibration_end)`.

    With `evaluation_end`, the calibration is measured on `[calibration_end, evaluation_end)`.
    """

    base_end: pd.Timestamp
    calibration_end: pd.Timestamp
    evaluation_end: pd.Timestamp | None = None


TEST_START = pd.Timestamp("2017-01-01")
TEST_END = pd.Timestamp("2017-09-01")

EXPANDING_WINDOWS: tuple[ValidationWindow, ...] = (
    ValidationWindow("2016Q1", pd.Timestamp("2016-01-01"), pd.Timestamp("2016-04-01")),
    ValidationWindow("2016Q2", pd.Timestamp("2016-04-01"), pd.Timestamp("2016-07-01")),
    ValidationWindow("2016Q3", pd.Timestamp("2016-07-01"), pd.Timestamp("2016-10-01")),
    ValidationWindow("2016Q4", pd.Timestamp("2016-10-01"), TEST_START),
)

CALIBRATION_SELECTION = TemporalCut(
    base_end=pd.Timestamp("2016-07-01"),
    calibration_end=pd.Timestamp("2016-10-01"),
    evaluation_end=TEST_START,
)

CHAMPION_CUT = TemporalCut(
    base_end=pd.Timestamp("2016-10-01"),
    calibration_end=TEST_START,
)
