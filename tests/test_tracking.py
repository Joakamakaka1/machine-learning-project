from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hotel_cancellation import model_trainer, models, tracking
from hotel_cancellation.data import data_loader

pytestmark = pytest.mark.slow

ONE_RUN = model_trainer.CalibrationChoice(
    method="none", briers={"none": 0.2}, metrics={"roc_auc": 0.8}, threshold_contrast=[]
)


def _small_mlp(seed: int) -> models.KerasMLPClassifier:
    return models.KerasMLPClassifier(hidden_layers=(8,), epochs=1, random_state=seed)


def _spec(name: str) -> models.ModelSpec:
    if name == "mlp":
        catalog = {spec.name: spec for spec in models.model_catalog("full")}
        return replace(catalog["mlp"], estimator=_small_mlp)
    return next(spec for spec in models.model_catalog("fast") if spec.name == name)


@pytest.mark.parametrize(
    ("name", "calibration"),
    [
        ("logistic_regression", "none"),
        ("logistic_regression", "sigmoid"),
        ("logistic_regression", "isotonic"),
        ("decision_tree", "isotonic"),
        ("random_forest", "sigmoid"),
        ("xgboost", "none"),
        ("xgboost", "sigmoid"),
        ("xgboost", "isotonic"),
        ("mlp", "sigmoid"),
    ],
)
def test_every_family_can_be_registered_and_restored(
    name: str, calibration: str, tmp_path: Path, synthetic_development: pd.DataFrame
) -> None:
    mlflow = pytest.importorskip("mlflow")
    if name == "mlp":
        pytest.importorskip("keras")
    spec = _spec(name)
    trainer = model_trainer.ModelTrainer(synthetic_development, catalog=(spec,), seed=0, n_jobs=1)
    served = trainer.fit_served(spec, {}, calibration)
    package = tmp_path / "package"
    package.mkdir()
    (package / "manifest.json").write_text("{}", "utf-8")
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    tracker = tracking.MlflowTracker(uri, tmp_path / "artifacts", experiment="families")
    selection = model_trainer.Selection(
        best=name, champion=name, table=pd.DataFrame({"model": [name]})
    )
    candidate = model_trainer.Candidate(name, {}, ONE_RUN, served)
    sample = data_loader.coerce_features(synthetic_development.head(30))

    tracker.log_training(
        selection,
        {name: candidate},
        champion_dir=package,
        manifest={"model": {"sha256": "abc"}},
        sample=sample,
    )
    try:
        mlflow.set_tracking_uri(uri)
        restored = mlflow.sklearn.load_model(f"models:/{tracking.REGISTERED_MODEL}@champion")
    finally:
        mlflow.set_tracking_uri("")
    np.testing.assert_allclose(
        restored.predict_proba(sample), served.predict_proba(sample), atol=1e-6
    )
