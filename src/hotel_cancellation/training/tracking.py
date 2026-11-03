"""Experiment tracking. MLflow is optional: the API image and the fast tests run without it."""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from hotel_cancellation.training import model_trainer

REGISTERED_MODEL = "hotel-cancellation"
CHAMPION_ALIAS = "champion"
DEFAULT_EXPERIMENT = "hotel-cancellation"

# MLflow stores scikit-learn models with skops, which refuses every type it does not know. This is
# what the models of the catalog need, the calibration wrapper included. The MLP cannot be stored
# with skops because it holds a Keras model, so it goes with cloudpickle.
_SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.calibration._CalibratedClassifier",
    "sklearn.calibration._SigmoidCalibration",
    "sklearn.tree._tree.Tree",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
]
_CLOUDPICKLE_MODELS = {"mlp"}


class Tracker(Protocol):
    """Where the pipeline reports what it does."""

    def log_search(
        self, outcomes: Mapping[str, model_trainer.SearchOutcome], *, profile: str
    ) -> None:
        """Record the search of each model."""

    def log_training(
        self,
        selection: model_trainer.Selection,
        candidates: Mapping[str, model_trainer.Candidate],
        *,
        champion_dir: Path,
        manifest: Mapping[str, Any],
        sample: pd.DataFrame,
    ) -> None:
        """Record the final training and point the alias at the champion."""


class NullTracker:
    """Tracks nothing."""

    def log_search(
        self, outcomes: Mapping[str, model_trainer.SearchOutcome], *, profile: str
    ) -> None:
        """Do nothing."""

    def log_training(
        self,
        selection: model_trainer.Selection,
        candidates: Mapping[str, model_trainer.Candidate],
        *,
        champion_dir: Path,
        manifest: Mapping[str, Any],
        sample: pd.DataFrame,
    ) -> None:
        """Do nothing."""


class MlflowTracker:
    """Local MLflow on SQLite, without a server.

    Args:
        tracking_uri: Store of the runs, usually `sqlite:///mlflow.db`.
        artifacts_dir: Folder of the artifacts of a new experiment.
        experiment: Experiment name.
    """

    def __init__(
        self, tracking_uri: str, artifacts_dir: Path, experiment: str = DEFAULT_EXPERIMENT
    ) -> None:
        """Keep the location; the first use opens the store, which can take a few seconds."""
        self.tracking_uri = tracking_uri
        self.artifacts_dir = artifacts_dir
        self.experiment = experiment

    def _start(self) -> Any:
        # Everything stays on this machine. MLflow reads this variable when it starts tracking.
        os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
        import mlflow

        mlflow.set_tracking_uri(self.tracking_uri)
        if mlflow.get_experiment_by_name(self.experiment) is None:
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            mlflow.create_experiment(self.experiment, self.artifacts_dir.as_uri())
        mlflow.set_experiment(self.experiment)
        return mlflow

    def log_search(
        self, outcomes: Mapping[str, model_trainer.SearchOutcome], *, profile: str
    ) -> None:
        """Log a parent run with one nested run per model: best parameters and scores by window."""
        mlflow = self._start()
        with mlflow.start_run(run_name="tune", tags={"profile": profile}):
            for name, outcome in outcomes.items():
                with mlflow.start_run(run_name=f"tune/{name}", nested=True):
                    mlflow.log_params(_short_params(outcome.params))
                    mlflow.log_metrics(_finite(_window_metrics(outcome.window_scores)))

    def log_training(
        self,
        selection: model_trainer.Selection,
        candidates: Mapping[str, model_trainer.Candidate],
        *,
        champion_dir: Path,
        manifest: Mapping[str, Any],
        sample: pd.DataFrame,
    ) -> None:
        """Log every candidate, register the champion and point `@champion` at it.

        The same package that gets evaluated (`model.joblib` and `manifest.json`) is attached to
        the run, and the registered version carries its hash.
        """
        mlflow = self._start()
        with mlflow.start_run(run_name="train"):
            mlflow.log_text(selection.table.to_csv(index=False), "selection.csv")
            mlflow.set_tag("best_model", selection.best)
            mlflow.set_tag("champion", selection.champion)
            for name, candidate in candidates.items():
                with mlflow.start_run(run_name=f"train/{name}", nested=True):
                    mlflow.log_params(_short_params(candidate.params))
                    mlflow.log_param("calibration", candidate.calibration.method)
                    mlflow.log_metrics(
                        _finite(
                            {f"{k}_2016Q4": v for k, v in candidate.calibration.metrics.items()}
                        )
                    )
                    if name == selection.champion:
                        self._register(mlflow, candidate, champion_dir, manifest, sample)

    def _register(
        self,
        mlflow: Any,
        candidate: model_trainer.Candidate,
        champion_dir: Path,
        manifest: Mapping[str, Any],
        sample: pd.DataFrame,
    ) -> None:
        mlflow.log_artifacts(str(champion_dir), artifact_path="champion_package")
        probabilities = candidate.model.predict_proba(sample)
        serialization: dict[str, Any] = (
            {"serialization_format": "cloudpickle"}
            if candidate.name in _CLOUDPICKLE_MODELS
            else {"skops_trusted_types": _SKOPS_TRUSTED_TYPES}
        )
        info = mlflow.sklearn.log_model(
            candidate.model,
            name="model",
            signature=mlflow.models.infer_signature(sample, probabilities),
            input_example=sample.head(2),
            pyfunc_predict_fn="predict_proba",
            registered_model_name=REGISTERED_MODEL,
            **serialization,
        )
        client = mlflow.MlflowClient()
        version = str(info.registered_model_version)
        client.set_model_version_tag(
            REGISTERED_MODEL, version, "sha256", manifest["model"]["sha256"]
        )
        client.set_registered_model_alias(REGISTERED_MODEL, CHAMPION_ALIAS, version)


def _short_params(params: Mapping[str, Any]) -> dict[str, str]:
    return {key.removeprefix("model__"): str(value) for key, value in params.items()}


def _window_metrics(scores: pd.DataFrame) -> dict[str, float]:
    per_window = {
        f"{metric}_{window}": float(scores.loc[window, metric])
        for window in scores.index
        for metric in scores.columns
    }
    means = {f"{metric}_mean": float(scores[metric].mean()) for metric in scores.columns}
    deviations = {f"{metric}_std": float(scores[metric].std(ddof=1)) for metric in scores.columns}
    return {**per_window, **means, **deviations}


def _finite(metrics: Mapping[str, float]) -> dict[str, float]:
    return {name: value for name, value in metrics.items() if value == value}
