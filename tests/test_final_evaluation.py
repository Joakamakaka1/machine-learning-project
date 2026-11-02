"""The single execution on the test, always on synthetic rows: these tests never read 2017."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hotel_cancellation import champion, cli, config, data_loader, evaluator, pipeline, schema
from tests.conftest import synthetic_raw


def test_psi_of_identical_distributions_is_zero() -> None:
    values = pd.Series(np.arange(1000.0))
    assert evaluator.population_stability_index(values, values) == pytest.approx(0, abs=1e-9)


def test_psi_of_categories_matches_a_worked_example() -> None:
    reference = pd.Series(["a"] * 50 + ["b"] * 50, dtype="str")
    current = pd.Series(["a"] * 25 + ["b"] * 75, dtype="str")
    expected = (0.25 - 0.5) * np.log(0.25 / 0.5) + (0.75 - 0.5) * np.log(0.75 / 0.5)
    assert evaluator.population_stability_index(reference, current) == pytest.approx(expected)


def test_psi_grows_with_the_shift_and_tolerates_new_categories() -> None:
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(0, 1, 5000))
    small = evaluator.population_stability_index(reference, pd.Series(rng.normal(0.1, 1, 5000)))
    large = evaluator.population_stability_index(reference, pd.Series(rng.normal(1.0, 1, 5000)))
    assert small < 0.1 < large
    new_category = evaluator.population_stability_index(
        pd.Series(["a", "b"], dtype="str"), pd.Series(["a", "c"], dtype="str")
    )
    assert np.isfinite(new_category)


class TestBootstrap:
    @pytest.fixture
    def rows(self) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        size = 4000
        probabilities = rng.uniform(0, 1, size)
        return pd.DataFrame(
            {
                "is_canceled": rng.binomial(1, probabilities),
                "good": probabilities,
                "noise": rng.uniform(0, 1, size),
                "block": rng.integers(0, 100, size),
            }
        )

    def test_interval_contains_the_estimate_and_is_reproducible(self, rows: pd.DataFrame) -> None:
        first: Any = evaluator.block_bootstrap_auc(
            rows["is_canceled"], rows[["good", "noise"]], rows["block"], replicates=50, seed=0
        ).set_index("model")
        second = evaluator.block_bootstrap_auc(
            rows["is_canceled"], rows[["good", "noise"]], rows["block"], replicates=50, seed=0
        ).set_index("model")
        pd.testing.assert_frame_equal(first, second)
        assert first.loc["good", "ci_low"] <= first.loc["good", "roc_auc"]
        assert first.loc["good", "roc_auc"] <= first.loc["good", "ci_high"]

    def test_differences_are_paired_and_separate_a_real_gap(self, rows: pd.DataFrame) -> None:
        table: Any = evaluator.block_bootstrap_auc(
            rows["is_canceled"],
            rows[["good", "noise"]],
            rows["block"],
            replicates=50,
            seed=0,
            against="good",
        ).set_index("model")
        assert table.loc["good", "gap_ci_low"] == 0
        assert table.loc["noise", "gap_ci_high"] < 0

    def test_a_block_is_resampled_as_a_whole(self) -> None:
        # Two blocks, each with a single class: a replicate holding only one block has no ranking.
        rows = pd.DataFrame(
            {"is_canceled": [0, 0, 1, 1], "model": [0.1, 0.2, 0.8, 0.9], "block": [0, 0, 1, 1]}
        )
        table: Any = evaluator.block_bootstrap_auc(
            rows["is_canceled"], rows[["model"]], rows["block"], replicates=200, seed=0
        )
        assert table.loc[0, "ci_low"] == 1
        assert table.loc[0, "replicates_used"] < 200


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> config.Settings:
    root = tmp_path_factory.mktemp("trained")
    settings = config.Settings(
        data_dir=root / "data", models_dir=root / "models", outputs_dir=root / "outputs"
    )
    development = data_loader.split_temporal(data_loader.clean(synthetic_raw())).development
    pipeline.run_all(development, settings=settings, profile="fast", seed=0)
    return settings


@pytest.fixture
def test_rows() -> pd.DataFrame:
    return data_loader.split_temporal(data_loader.clean(synthetic_raw())).test


@pytest.fixture
def evaluated(
    trained: config.Settings,
    test_rows: pd.DataFrame,
    synthetic_development: pd.DataFrame,
    tmp_path: Path,
) -> config.Settings:
    """Each test gets its own outputs folder, over the same trained models."""
    settings = trained.model_copy(update={"outputs_dir": tmp_path / "outputs"})
    pipeline.evaluate(test_rows, synthetic_development, settings=settings, replicates=20)
    return settings


class TestFinalEvaluation:
    def test_it_leaves_a_record_with_date_commit_and_champion(
        self, evaluated: config.Settings, trained: config.Settings
    ) -> None:
        record = json.loads(evaluated.final_evaluation_record.read_text("utf-8"))
        manifest = champion.load_champion(trained.champion_dir).manifest
        assert len(record["executions"]) == 1
        assert record["executions"][0]["executed_at"]
        assert len(record["executions"][0]["code"]["commit"]) == 40
        assert record["executions"][0]["champion_sha256"] == manifest["model"]["sha256"]
        assert record["executions"][0]["forced"] is False

    def test_it_refuses_to_run_twice(
        self,
        evaluated: config.Settings,
        test_rows: pd.DataFrame,
        synthetic_development: pd.DataFrame,
    ) -> None:
        with pytest.raises(pipeline.TestAlreadyEvaluatedError, match="--force"):
            pipeline.evaluate(test_rows, synthetic_development, settings=evaluated, replicates=20)

    def test_forcing_it_declares_the_repetition(
        self,
        evaluated: config.Settings,
        test_rows: pd.DataFrame,
        synthetic_development: pd.DataFrame,
    ) -> None:
        pipeline.evaluate(
            test_rows, synthetic_development, settings=evaluated, replicates=20, force=True
        )
        record = json.loads(evaluated.final_evaluation_record.read_text("utf-8"))
        assert [execution["forced"] for execution in record["executions"]] == [False, True]

    def test_metrics_cover_every_model_threshold_and_mandatory_subset(
        self, evaluated: config.Settings, trained: config.Settings
    ) -> None:
        metrics = pd.read_csv(evaluated.final_evaluation_dir / "metrics.csv")
        saved = {path.stem for path in trained.candidates_dir.glob("*.joblib")}
        assert set(metrics["model"]) == saved
        assert set(metrics["subset"]) == {"all", "unique", "no_deposit", "non_refund"}
        assert set(metrics["threshold_name"]) == {"decision", "fixed_0.5"}
        assert {
            "roc_auc",
            "pr_auc",
            "brier",
            "calibration_slope",
            "f1",
            "tp",
            "fp",
            "fn",
            "tn",
        } <= set(metrics)

    def test_it_uses_the_threshold_of_the_manifest(self, evaluated: config.Settings) -> None:
        metrics = pd.read_csv(evaluated.final_evaluation_dir / "metrics.csv")
        decision = metrics[metrics["threshold_name"] == "decision"]
        assert decision["threshold"].unique().tolist() == [0.25]

    def test_it_evaluates_the_champion_as_it_is(
        self, evaluated: config.Settings, trained: config.Settings, test_rows: pd.DataFrame
    ) -> None:
        loaded = champion.load_champion(trained.champion_dir)
        predictions = pd.read_csv(evaluated.final_evaluation_dir / "predictions.csv.gz")
        expected = loaded.model.predict_proba(data_loader.coerce_features(test_rows))[:, 1]
        np.testing.assert_allclose(predictions["champion"], expected, atol=1e-9)
        assert len(predictions) == len(test_rows)

    def test_every_table_of_the_protocol_is_written(self, evaluated: config.Settings) -> None:
        folder = evaluated.final_evaluation_dir
        bootstrap = pd.read_csv(folder / "bootstrap.csv")
        assert set(bootstrap["blocks"]) == {"hotel_date", "week"}
        monthly = pd.read_csv(folder / "monthly.csv")
        assert monthly["month"].tolist()[0] == "2017-01"
        assert {"rows", "prevalence", "roc_auc"} <= set(monthly)
        psi = pd.read_csv(folder / "psi.csv")
        assert set(psi["feature"]) == set(schema.INPUT_FEATURES)
        segments = pd.read_csv(folder / "segments.csv")
        assert {"hotel", "market_segment", "customer_type"} <= set(segments["segment"])

    def test_it_refuses_rows_that_are_not_of_the_test_period(
        self, trained: config.Settings, tmp_path: Path, synthetic_development: pd.DataFrame
    ) -> None:
        settings = trained.model_copy(update={"outputs_dir": tmp_path / "outputs"})
        with pytest.raises(ValueError, match="test period"):
            pipeline.evaluate(
                synthetic_development, synthetic_development, settings=settings, replicates=20
            )


class TestCommands:
    def test_run_all_never_evaluates(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        settings = config.Settings(
            data_dir=tmp_path / "data", models_dir=tmp_path / "models", outputs_dir=tmp_path / "out"
        )
        monkeypatch.setattr(config, "get_settings", lambda: settings)
        frame = data_loader.split_temporal(data_loader.clean(synthetic_raw())).development
        monkeypatch.setattr(pipeline, "load_development", lambda: frame)

        def fail() -> None:
            raise AssertionError("run-all read the test rows")

        monkeypatch.setattr(pipeline, "load_test", fail)
        assert cli.main(["run-all", "--profile", "fast", "--no-tracking", "--seed", "0"]) == 0
        assert not settings.final_evaluation_dir.exists()

    def test_tune_and_train_do_not_load_the_test(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings = config.Settings(
            data_dir=tmp_path / "data", models_dir=tmp_path / "models", outputs_dir=tmp_path / "out"
        )
        monkeypatch.setattr(config, "get_settings", lambda: settings)
        frame = data_loader.split_temporal(data_loader.clean(synthetic_raw())).development
        monkeypatch.setattr(pipeline, "load_development", lambda: frame)
        calls: list[int] = []
        monkeypatch.setattr(pipeline, "load_test", lambda: calls.append(1))
        base = ["--profile", "fast", "--no-tracking", "--seed", "0"]
        assert cli.main(["tune", *base]) == 0
        assert cli.main(["train", *base]) == 0
        assert calls == []

    def test_evaluate_command_refuses_to_repeat(
        self,
        evaluated: config.Settings,
        test_rows: pd.DataFrame,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(config, "get_settings", lambda: evaluated)
        monkeypatch.setattr(pipeline, "load_test", lambda: test_rows)
        development = data_loader.split_temporal(data_loader.clean(synthetic_raw())).development
        monkeypatch.setattr(pipeline, "load_development", lambda: development)
        assert cli.main(["evaluate", "--replicates", "20"]) == 1
        assert cli.main(["evaluate", "--replicates", "20", "--force"]) == 0
