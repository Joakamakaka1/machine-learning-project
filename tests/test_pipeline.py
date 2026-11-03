import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hotel_cancellation import champion, cli, config, data_loader, evaluator, models, pipeline
from tests.conftest import synthetic_raw


@pytest.fixture
def settings(tmp_path: Path) -> config.Settings:
    return config.Settings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        outputs_dir=tmp_path / "outputs",
        mlflow_tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        mlflow_artifacts_dir=tmp_path / "mlartifacts",
    )


@pytest.fixture(scope="module")
def full_run(tmp_path_factory: pytest.TempPathFactory) -> config.Settings:
    """One complete run of the fast profile, shared by the tests that only read its results."""
    root = tmp_path_factory.mktemp("run")
    settings = config.Settings(
        data_dir=root / "data", models_dir=root / "models", outputs_dir=root / "outputs"
    )
    development = data_loader.split_temporal(data_loader.clean(synthetic_raw())).development
    pipeline.run_all(development, settings=settings, profile="fast", seed=0)
    return settings


class TestRunAll:
    def test_best_params_keep_every_model_and_its_windows(self, full_run: config.Settings) -> None:
        content = json.loads(full_run.best_params_path.read_text("utf-8"))
        assert content["profile"] == "fast"
        names = {spec.name for spec in models.model_catalog("fast")} | {"dummy"}
        assert set(content["models"]) == names
        for model in content["models"].values():
            assert len(model["windows"]) == len(config.EXPANDING_WINDOWS)
            assert set(model["mean"]) == set(evaluator.VALIDATION_METRICS)

    def test_champion_beats_the_baseline_in_validation(self, full_run: config.Settings) -> None:
        loaded = champion.load_champion(full_run.champion_dir)
        selection = {row["model"]: row for row in loaded.manifest["selection"]}
        assert loaded.manifest["model"]["name"] in selection
        content = json.loads(full_run.best_params_path.read_text("utf-8"))
        assert selection[loaded.manifest["model"]["name"]]["mean_roc_auc"] > 0.6
        assert content["models"]["dummy"]["mean"]["roc_auc"] == pytest.approx(0.5)

    def test_baseline_is_never_a_candidate_for_champion(self, full_run: config.Settings) -> None:
        manifest = champion.load_champion(full_run.champion_dir).manifest
        assert "dummy" not in {row["model"] for row in manifest["selection"]}

    def test_manifest_holds_the_validation_of_the_champion(self, full_run: config.Settings) -> None:
        manifest = champion.load_champion(full_run.champion_dir).manifest
        assert set(manifest["validation"]["windows"]) == {
            window.name for window in config.EXPANDING_WINDOWS
        }
        assert 0.5 < manifest["validation"]["mean"]["roc_auc"] < 1

    def test_every_candidate_is_saved_for_the_later_comparison(
        self, full_run: config.Settings
    ) -> None:
        saved = {path.stem for path in full_run.candidates_dir.glob("*.joblib")}
        assert saved == {spec.name for spec in models.model_catalog("fast")} | {"dummy"}

    def test_validation_tables_are_written(self, full_run: config.Settings) -> None:
        by_window = pd.read_csv(full_run.validation_dir / "metrics_by_window.csv")
        assert set(by_window["window"]) == {window.name for window in config.EXPANDING_WINDOWS}
        summary = pd.read_csv(full_run.validation_dir / "summary.csv")
        assert {"model", "roc_auc_mean", "roc_auc_std", "f1_mean", "brier_mean"} <= set(summary)
        assert (full_run.validation_dir / "selection.csv").exists()
        assert (full_run.validation_dir / "calibration.json").exists()
        assert (full_run.validation_dir / "search_xgboost.csv").exists()


class TestTrainInput:
    def test_training_without_a_search_explains_what_is_missing(
        self, settings: config.Settings, synthetic_development: pd.DataFrame
    ) -> None:
        with pytest.raises(FileNotFoundError, match="hotel-ml tune"):
            pipeline.train(synthetic_development, settings=settings, profile="fast")

    def test_training_refuses_a_search_of_another_profile(
        self, settings: config.Settings, synthetic_development: pd.DataFrame
    ) -> None:
        settings.models_dir.mkdir(parents=True)
        settings.best_params_path.write_text(json.dumps({"profile": "full", "models": {}}), "utf-8")
        with pytest.raises(ValueError, match="profile"):
            pipeline.train(synthetic_development, settings=settings, profile="fast")

    def test_a_partial_search_keeps_the_previous_models(
        self, settings: config.Settings, synthetic_development: pd.DataFrame
    ) -> None:
        pipeline.tune(
            synthetic_development, settings=settings, profile="fast", only=["logistic_regression"]
        )
        pipeline.tune(synthetic_development, settings=settings, profile="fast", only=["dummy"])
        content = json.loads(settings.best_params_path.read_text("utf-8"))
        assert set(content["models"]) == {"logistic_regression", "dummy"}


def _gzip_csv(frame: pd.DataFrame, path: Path) -> str:
    csv = frame.to_csv(index=False, na_rep="NULL").encode("utf-8")
    path.write_bytes(gzip.compress(csv))
    return hashlib.sha256(csv).hexdigest()


def _poison_2017(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep the rows valid, but make the outcomes and inputs of 2017 entirely different."""
    is_2017 = raw["arrival_date_year"] == 2017
    poisoned = raw.copy()
    poisoned.loc[is_2017, "is_canceled"] = 1 - poisoned.loc[is_2017, "is_canceled"]
    poisoned.loc[is_2017, "lead_time"] = 999
    poisoned.loc[is_2017, "adr"] = 1.0
    poisoned.loc[is_2017, "deposit_type"] = "Refundable"
    poisoned.loc[is_2017, "agent"] = "777"
    return poisoned


def test_training_and_selection_do_not_read_rows_of_2017(tmp_path: Path) -> None:
    raw = synthetic_raw()
    probe = data_loader.coerce_features(data_loader.clean(raw).head(50))
    assert (raw["arrival_date_year"] == 2017).sum() > 0
    results = []
    for name, frame in (("clean", raw), ("poisoned", _poison_2017(raw))):
        folder = tmp_path / name
        folder.mkdir()
        path = folder / "bookings.csv.gz"
        digest = _gzip_csv(frame, path)
        development = pipeline.load_development(path, expected_sha256=digest)
        settings = config.Settings(
            data_dir=folder / "data", models_dir=folder / "models", outputs_dir=folder / "outputs"
        )
        pipeline.run_all(development, settings=settings, profile="fast", seed=0)
        results.append(
            (
                settings.best_params_path.read_text("utf-8"),
                champion.load_champion(settings.champion_dir),
                pd.read_csv(settings.validation_dir / "metrics_by_window.csv"),
            )
        )
    (params_a, champion_a, table_a), (params_b, champion_b, table_b) = results
    assert json.loads(params_a)["models"] == json.loads(params_b)["models"]
    pd.testing.assert_frame_equal(table_a, table_b)
    assert champion_a.manifest["model"]["name"] == champion_b.manifest["model"]["name"]
    np.testing.assert_array_equal(
        champion_a.model.predict_proba(probe), champion_b.model.predict_proba(probe)
    )


def test_load_development_excludes_the_test_period(tmp_path: Path) -> None:
    path = tmp_path / "bookings.csv.gz"
    digest = _gzip_csv(synthetic_raw(rows_per_month=10), path)
    development = pipeline.load_development(path, expected_sha256=digest)
    assert development["arrival_date"].max() < config.TEST_START
    assert development["arrival_date"].is_monotonic_increasing


class TestCli:
    @pytest.fixture
    def development(
        self, monkeypatch: pytest.MonkeyPatch, settings: config.Settings
    ) -> config.Settings:
        monkeypatch.setattr(config, "get_settings", lambda: settings)
        frame = data_loader.split_temporal(data_loader.clean(synthetic_raw())).development
        monkeypatch.setattr(pipeline, "load_development", lambda: frame)
        return settings

    def test_run_all_trains_a_champion(self, development: config.Settings) -> None:
        arguments = ["run-all", "--profile", "fast", "--no-tracking", "--seed", "0"]
        assert cli.main(arguments) == 0
        assert (development.champion_dir / champion.MANIFEST_FILE).exists()

    def test_tune_and_train_run_separately(self, development: config.Settings) -> None:
        base = ["--profile", "fast", "--no-tracking", "--seed", "0"]
        assert cli.main(["tune", *base]) == 0
        assert not development.champion_dir.exists()
        assert cli.main(["train", *base]) == 0
        assert development.champion_dir.exists()

    def test_run_all_has_no_way_to_reach_the_test_by_default(self) -> None:
        arguments = cli.build_parser().parse_args(["run-all"])
        assert arguments.profile == "full"
        assert not getattr(arguments, "evaluate", False)

    def test_an_unknown_command_is_rejected(self) -> None:
        with pytest.raises(SystemExit):
            cli.main(["delete-everything"])


@pytest.mark.slow
def test_mlflow_tracks_the_run_and_points_the_alias_at_the_evaluated_artifact(
    settings: config.Settings, synthetic_development: pd.DataFrame
) -> None:
    mlflow = pytest.importorskip("mlflow")
    tracking = pytest.importorskip("hotel_cancellation.tracking")
    tracker = tracking.MlflowTracker(
        settings.mlflow_tracking_uri, settings.mlflow_artifacts_dir, experiment="tests"
    )
    pipeline.run_all(
        synthetic_development, settings=settings, profile="fast", seed=0, tracker=tracker
    )
    loaded = champion.load_champion(settings.champion_dir)
    probe = data_loader.coerce_features(synthetic_development.head(20))

    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        restored = mlflow.sklearn.load_model(f"models:/{tracking.REGISTERED_MODEL}@champion")
        version = mlflow.MlflowClient().get_model_version_by_alias(
            tracking.REGISTERED_MODEL, "champion"
        )
        runs = mlflow.search_runs(experiment_names=["tests"])
        pyfunc = mlflow.pyfunc.load_model(f"models:/{tracking.REGISTERED_MODEL}@champion")
    finally:
        mlflow.set_tracking_uri("")

    np.testing.assert_allclose(
        restored.predict_proba(probe), loaded.model.predict_proba(probe), atol=1e-9
    )
    assert version.tags["sha256"] == loaded.manifest["model"]["sha256"]
    assert pyfunc.predict(probe).shape == (20, 2)
    nested = runs[runs["tags.mlflow.parentRunId"].notna()]
    assert set(runs["tags.mlflow.runName"]) >= {"tune", "train"}
    assert {f"tune/{spec.name}" for spec in models.model_catalog("fast")} <= set(
        nested["tags.mlflow.runName"]
    )
    assert {f"train/{spec.name}" for spec in models.model_catalog("fast")} <= set(
        nested["tags.mlflow.runName"]
    )
