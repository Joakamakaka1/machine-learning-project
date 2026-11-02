"""The single execution on the test, always on synthetic rows: these tests never read 2017."""

import json
import shutil
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


def test_psi_of_a_binary_variable_matches_a_worked_example() -> None:
    reference = pd.Series(np.r_[np.zeros(970), np.ones(30)])
    current = pd.Series(np.r_[np.zeros(700), np.ones(300)])
    assert evaluator.population_stability_index(reference, current) == pytest.approx(
        0.7098, abs=1e-3
    )


def test_psi_sees_a_shift_in_a_variable_that_is_mostly_zero() -> None:
    rng = np.random.default_rng(0)
    reference = pd.Series(np.where(rng.random(10_000) < 0.97, 0, rng.integers(1, 300, 10_000)))
    current = pd.Series(np.where(rng.random(10_000) < 0.80, 0, rng.integers(1, 300, 10_000)))
    assert reference.nunique() > 100
    assert evaluator.population_stability_index(reference, current) > 0.25
    assert evaluator.population_stability_index(reference, reference) == pytest.approx(0, abs=1e-9)


def test_psi_splits_the_values_around_a_massive_tie_into_quantiles() -> None:
    rng = np.random.default_rng(0)
    rest = rng.uniform(1, 100, 5000)
    reference = pd.Series(np.r_[np.zeros(5000), rest])
    moved = pd.Series(np.r_[np.zeros(5000), rest + 50])
    assert evaluator.population_stability_index(reference, moved) > 0.25


def test_psi_gives_a_new_value_of_a_discrete_variable_its_own_bin() -> None:
    reference = pd.Series([0.0, 1.0] * 500)
    current = pd.Series([0.0, 1.0, 2.0, 2.0] * 250)
    assert evaluator.population_stability_index(reference, current) > 1


def test_psi_keeps_missing_values_apart() -> None:
    reference = pd.Series([0.0, 1.0] * 500)
    current = pd.Series([0.0, 1.0, np.nan, np.nan] * 250)
    assert evaluator.population_stability_index(reference, current) > 1


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

    def test_a_single_class_gives_nan_instead_of_failing(self) -> None:
        rows = pd.DataFrame({"is_canceled": [0, 0], "model": [0.1, 0.2], "block": [0, 1]})
        table = evaluator.block_bootstrap_auc(
            rows["is_canceled"], rows[["model"]], rows["block"], replicates=5, seed=0
        )
        assert table.loc[0, "replicates_used"] == 0
        assert table["ci_low"].isna().all()

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
    pipeline.evaluate(
        test_rows, synthetic_development, settings=settings, replicates=20, strict=False
    )
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
            pipeline.evaluate(
                test_rows, synthetic_development, settings=evaluated, replicates=20, strict=False
            )

    def test_forcing_it_declares_the_repetition(
        self,
        evaluated: config.Settings,
        test_rows: pd.DataFrame,
        synthetic_development: pd.DataFrame,
    ) -> None:
        pipeline.evaluate(
            test_rows,
            synthetic_development,
            settings=evaluated,
            replicates=20,
            force=True,
            strict=False,
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
                synthetic_development,
                synthetic_development,
                settings=settings,
                replicates=20,
                strict=False,
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
        monkeypatch.setattr(pipeline, "_assert_real_execution", lambda *_: None)
        development = data_loader.split_temporal(data_loader.clean(synthetic_raw())).development
        monkeypatch.setattr(pipeline, "load_development", lambda: development)
        assert cli.main(["evaluate", "--replicates", "20"]) == 1
        assert cli.main(["evaluate", "--replicates", "20", "--force"]) == 0


def _copy_models(trained: config.Settings, tmp_path: Path) -> config.Settings:
    shutil.copytree(trained.models_dir, tmp_path / "models")
    return trained.model_copy(
        update={"models_dir": tmp_path / "models", "outputs_dir": tmp_path / "outputs"}
    )


class TestMetricsTables:
    def test_each_cost_uses_its_own_threshold_and_is_also_priced_at_the_served_one(
        self, evaluated: config.Settings, test_rows: pd.DataFrame
    ) -> None:
        metrics = pd.read_csv(evaluated.final_evaluation_dir / "metrics.csv")
        predictions = pd.read_csv(evaluated.final_evaluation_dir / "predictions.csv.gz")
        name = "logistic_regression"
        row: Any = metrics[
            (metrics["model"] == name)
            & (metrics["subset"] == "all")
            & (metrics["threshold_name"] == "decision")
        ].iloc[0]
        y, p = predictions["is_canceled"], predictions[name]
        high_miss = evaluator.CostScenario(1, 5)
        assert row["cost_1_5"] == pytest.approx(
            evaluator.cost_per_booking(y, p, 1 / 6, high_miss), abs=1e-5
        )
        assert row["cost_1_5_at_threshold"] == pytest.approx(
            evaluator.cost_per_booking(y, p, 0.25, high_miss), abs=1e-5
        )

    def test_it_reports_the_slope_with_its_sensitivity(self, evaluated: config.Settings) -> None:
        metrics = pd.read_csv(evaluated.final_evaluation_dir / "metrics.csv")
        assert {
            "calibration_slope",
            "calibration_slope_clip_1e6",
            "calibration_slope_clip_1e2",
            "share_probability_zero",
            "share_probability_one",
        } <= set(metrics)

    def test_a_segment_with_a_single_class_keeps_its_row_count_and_gets_nan(
        self,
        trained: config.Settings,
        test_rows: pd.DataFrame,
        synthetic_development: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        settings = trained.model_copy(update={"outputs_dir": tmp_path / "outputs"})
        rows = test_rows.copy()
        rows.loc[rows["deposit_type"] == "Refundable", "is_canceled"] = 0
        rows.loc[rows.index[:8], "deposit_type"] = "Refundable"
        rows.loc[rows.index[:8], "is_canceled"] = 0
        pipeline.evaluate(
            rows, synthetic_development, settings=settings, replicates=5, strict=False
        )
        segments = pd.read_csv(settings.final_evaluation_dir / "segments.csv")
        refundable: Any = segments[segments["value"] == "Refundable"].iloc[0]
        assert refundable["rows"] >= 8
        assert np.isnan(refundable["roc_auc"])
        assert np.isnan(refundable["calibration_slope"])
        assert refundable["prevalence"] == 0

    def test_the_scenario_with_special_requests_is_measured_as_informative(
        self, evaluated: config.Settings
    ) -> None:
        scenario = pd.read_csv(evaluated.final_evaluation_dir / "scenario_special_requests.csv")
        assert scenario["informative_only"].all()
        assert set(scenario["subset"]) == {"all", "unique", "no_deposit", "non_refund"}
        # On synthetic rows the variable copies the outcome.
        assert scenario[scenario["subset"] == "all"]["roc_auc"].min() > 0.99
        record = json.loads(evaluated.final_evaluation_record.read_text("utf-8"))
        assert record["executions"][0]["scenario_sha256"]
        metrics = pd.read_csv(evaluated.final_evaluation_dir / "metrics.csv")
        assert pipeline.SCENARIO_NAME not in set(metrics["model"])
        assert metrics["roc_auc"].max() < 0.99


class TestAtomicity:
    def test_a_failure_while_computing_leaves_nothing_and_does_not_spend_the_test(
        self,
        trained: config.Settings,
        test_rows: pd.DataFrame,
        synthetic_development: pd.DataFrame,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        settings = trained.model_copy(update={"outputs_dir": tmp_path / "outputs"})

        def boom(*_: Any, **__: Any) -> None:
            raise RuntimeError("failure halfway")

        with monkeypatch.context() as patch:
            patch.setattr(evaluator, "block_bootstrap_auc", boom)
            with pytest.raises(RuntimeError, match="halfway"):
                pipeline.evaluate(
                    test_rows, synthetic_development, settings=settings, replicates=5, strict=False
                )
        assert not settings.final_evaluation_dir.exists()
        pipeline.evaluate(
            test_rows, synthetic_development, settings=settings, replicates=5, strict=False
        )
        record = json.loads(settings.final_evaluation_record.read_text("utf-8"))
        assert [execution["forced"] for execution in record["executions"]] == [False]

    def test_a_failure_while_writing_leaves_nothing_visible(
        self,
        trained: config.Settings,
        test_rows: pd.DataFrame,
        synthetic_development: pd.DataFrame,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        settings = trained.model_copy(update={"outputs_dir": tmp_path / "outputs"})
        calls: list[Path] = []
        real = pipeline._write_table

        def fail_on_third(table: pd.DataFrame, path: Path) -> None:
            calls.append(path)
            if len(calls) == 3:
                raise OSError("disk full")
            real(table, path)

        monkeypatch.setattr(pipeline, "_write_table", fail_on_third)
        with pytest.raises(OSError, match="disk full"):
            pipeline.evaluate(
                test_rows, synthetic_development, settings=settings, replicates=5, strict=False
            )
        assert not settings.final_evaluation_dir.exists()
        assert list(settings.outputs_dir.glob("*")) == []

    def test_forcing_keeps_the_previous_execution_apart(
        self,
        evaluated: config.Settings,
        test_rows: pd.DataFrame,
        synthetic_development: pd.DataFrame,
    ) -> None:
        pipeline.evaluate(
            test_rows,
            synthetic_development,
            settings=evaluated,
            replicates=5,
            force=True,
            strict=False,
        )
        previous = evaluated.outputs_dir / "final_evaluation.superseded-1"
        assert (previous / "metrics.csv").exists()
        assert (evaluated.final_evaluation_dir / "metrics.csv").exists()


class TestPreflight:
    """Every check runs before reading a single row of the test."""

    def _run_command(
        self, settings: config.Settings, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[int, list[str]]:
        reads: list[str] = []

        def read_rows() -> pd.DataFrame:
            reads.append("read")
            raise AssertionError("the test rows were read")

        monkeypatch.setattr(config, "get_settings", lambda: settings)
        monkeypatch.setattr(pipeline, "load_test", read_rows)
        monkeypatch.setattr(pipeline, "_assert_real_execution", lambda *_: None)
        return cli.main(["evaluate", "--replicates", "5"]), reads

    @pytest.fixture
    def copied(self, trained: config.Settings, tmp_path: Path) -> config.Settings:
        return _copy_models(trained, tmp_path)

    def test_a_good_package_reaches_the_rows(
        self, copied: config.Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with pytest.raises(AssertionError, match="test rows were read"):
            self._run_command(copied, monkeypatch)

    def test_a_missing_candidate_stops_it(
        self, copied: config.Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (copied.candidates_dir / "xgboost.joblib").unlink()
        assert self._run_command(copied, monkeypatch) == (1, [])

    def test_a_modified_candidate_stops_it(
        self, copied: config.Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (copied.candidates_dir / "decision_tree.joblib").write_bytes(b"modified")
        assert self._run_command(copied, monkeypatch) == (1, [])

    def test_a_missing_registry_stops_it(
        self, copied: config.Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        copied.candidates_registry_path.unlink()
        assert self._run_command(copied, monkeypatch) == (1, [])

    def test_a_registry_without_a_model_of_the_catalog_stops_it(
        self, copied: config.Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        registry = json.loads(copied.candidates_registry_path.read_text("utf-8"))
        del registry["candidates"]["dummy"]
        copied.candidates_registry_path.write_text(json.dumps(registry), "utf-8")
        assert self._run_command(copied, monkeypatch) == (1, [])

    def test_a_champion_that_is_not_its_candidate_stops_it(
        self, copied: config.Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        registry = json.loads(copied.candidates_registry_path.read_text("utf-8"))
        name = registry["champion"]["name"]
        other = next(key for key in registry["candidates"] if key != name)
        swapped = copied.candidates_dir / f"{other}.joblib"
        shutil.copy(swapped, copied.candidates_dir / f"{name}.joblib")
        registry["candidates"][name] = {
            **registry["candidates"][other],
            "file": f"candidates/{name}.joblib",
        }
        copied.candidates_registry_path.write_text(json.dumps(registry), "utf-8")
        assert self._run_command(copied, monkeypatch) == (1, [])

    def test_a_modified_scenario_stops_it(
        self, copied: config.Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        copied.scenario_model_path.write_bytes(b"modified")
        assert self._run_command(copied, monkeypatch) == (1, [])

    def test_a_modified_champion_stops_it(
        self, copied: config.Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (copied.champion_dir / champion.MODEL_FILE).write_bytes(b"modified")
        assert self._run_command(copied, monkeypatch) == (1, [])


class TestRealExecutionConditions:
    @pytest.fixture
    def default_full(
        self, trained: config.Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> config.Settings:
        """A package of the full profile in the default folders, with a clean tree."""
        settings = _copy_models(trained, tmp_path)
        manifest = json.loads((settings.champion_dir / champion.MANIFEST_FILE).read_text("utf-8"))
        manifest["profile"] = "full"
        (settings.champion_dir / champion.MANIFEST_FILE).write_text(json.dumps(manifest), "utf-8")
        monkeypatch.setattr(config, "DEFAULT_MODELS_DIR", settings.models_dir)
        monkeypatch.setattr(config, "DEFAULT_OUTPUTS_DIR", settings.outputs_dir)
        monkeypatch.setattr(champion, "tree_is_clean", lambda: True)
        return settings

    def test_it_passes_when_every_condition_holds(self, default_full: config.Settings) -> None:
        pipeline._assert_real_execution(default_full, {"profile": "full"})

    def test_it_demands_the_full_profile(self, default_full: config.Settings) -> None:
        with pytest.raises(pipeline.EvaluationNotReadyError, match="full"):
            pipeline._assert_real_execution(default_full, {"profile": "fast"})

    def test_it_demands_a_clean_tree(
        self, default_full: config.Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(champion, "tree_is_clean", lambda: False)
        with pytest.raises(pipeline.EvaluationNotReadyError, match="clean"):
            pipeline._assert_real_execution(default_full, {"profile": "full"})

    def test_it_demands_the_default_outputs(
        self, default_full: config.Settings, tmp_path: Path
    ) -> None:
        moved = default_full.model_copy(update={"outputs_dir": tmp_path / "elsewhere"})
        with pytest.raises(pipeline.EvaluationNotReadyError, match="HOTEL_ML_OUTPUTS_DIR"):
            pipeline._assert_real_execution(moved, {"profile": "full"})

    def test_it_demands_the_default_models(
        self, default_full: config.Settings, tmp_path: Path
    ) -> None:
        moved = default_full.model_copy(update={"models_dir": tmp_path / "elsewhere"})
        with pytest.raises(pipeline.EvaluationNotReadyError, match="HOTEL_ML_MODELS_DIR"):
            pipeline._assert_real_execution(moved, {"profile": "full"})

    def test_the_real_path_is_the_default(
        self,
        trained: config.Settings,
        test_rows: pd.DataFrame,
        synthetic_development: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        settings = trained.model_copy(update={"outputs_dir": tmp_path / "outputs"})
        with pytest.raises(pipeline.EvaluationNotReadyError):
            pipeline.evaluate(test_rows, synthetic_development, settings=settings, replicates=5)
        assert not settings.final_evaluation_dir.exists()

    def test_a_real_command_on_the_fast_profile_never_reads_the_rows(
        self, trained: config.Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = trained.model_copy(update={"outputs_dir": tmp_path / "outputs"})

        def fail() -> None:
            raise AssertionError("the test rows were read")

        monkeypatch.setattr(config, "get_settings", lambda: settings)
        monkeypatch.setattr(pipeline, "load_test", fail)
        assert cli.main(["evaluate", "--replicates", "5"]) == 1


class TestChampionIsFrozenAfterTheTest:
    @pytest.fixture
    def after_test(
        self,
        trained: config.Settings,
        test_rows: pd.DataFrame,
        synthetic_development: pd.DataFrame,
        tmp_path: Path,
    ) -> config.Settings:
        settings = _copy_models(trained, tmp_path)
        pipeline.evaluate(
            test_rows, synthetic_development, settings=settings, replicates=5, strict=False
        )
        return settings

    def test_train_refuses_without_force(
        self, after_test: config.Settings, synthetic_development: pd.DataFrame
    ) -> None:
        before = (after_test.champion_dir / champion.MODEL_FILE).read_bytes()
        with pytest.raises(pipeline.TestAlreadyEvaluatedError, match="--force"):
            pipeline.train(synthetic_development, settings=after_test, profile="fast", seed=0)
        assert (after_test.champion_dir / champion.MODEL_FILE).read_bytes() == before

    def test_train_replaces_it_with_force(
        self, after_test: config.Settings, synthetic_development: pd.DataFrame
    ) -> None:
        report = pipeline.train(
            synthetic_development, settings=after_test, profile="fast", seed=0, force=True
        )
        assert report.manifest["model"]["name"]

    def test_run_all_refuses_before_searching(
        self, after_test: config.Settings, synthetic_development: pd.DataFrame
    ) -> None:
        before = after_test.best_params_path.read_bytes()
        with pytest.raises(pipeline.TestAlreadyEvaluatedError):
            pipeline.run_all(synthetic_development, settings=after_test, profile="fast", seed=0)
        assert after_test.best_params_path.read_bytes() == before

    def test_the_scenario_cannot_be_trained_after_the_test(
        self, after_test: config.Settings, synthetic_development: pd.DataFrame
    ) -> None:
        with pytest.raises(pipeline.TestAlreadyEvaluatedError):
            pipeline.train_scenario(synthetic_development, settings=after_test)

    def test_moving_the_outputs_does_not_escape_the_record(
        self,
        after_test: config.Settings,
        synthetic_development: pd.DataFrame,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(config, "DEFAULT_MODELS_DIR", after_test.models_dir)
        monkeypatch.setattr(config, "DEFAULT_OUTPUTS_DIR", after_test.outputs_dir)
        moved = after_test.model_copy(update={"outputs_dir": tmp_path / "elsewhere"})
        with pytest.raises(pipeline.TestAlreadyEvaluatedError):
            pipeline.train(synthetic_development, settings=moved, profile="fast", seed=0)

    def test_the_commands_answer_with_an_error_code(
        self,
        after_test: config.Settings,
        synthetic_development: pd.DataFrame,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(config, "get_settings", lambda: after_test)
        monkeypatch.setattr(pipeline, "load_development", lambda: synthetic_development)
        base = ["--profile", "fast", "--no-tracking", "--seed", "0"]
        assert cli.main(["train", *base]) == 1
        assert cli.main(["run-all", *base]) == 1
        assert cli.main(["train", *base, "--force"]) == 0


class TestScenarioRegistration:
    def test_training_it_does_not_touch_the_champion_nor_the_registered_candidates(
        self, trained: config.Settings, synthetic_development: pd.DataFrame, tmp_path: Path
    ) -> None:
        settings = _copy_models(trained, tmp_path)
        before = (settings.champion_dir / champion.MODEL_FILE).read_bytes()
        registry_before = json.loads(settings.candidates_registry_path.read_text("utf-8"))
        settings.scenario_model_path.unlink()
        pipeline.train_scenario(synthetic_development, settings=settings)
        registry = json.loads(settings.candidates_registry_path.read_text("utf-8"))
        assert (settings.champion_dir / champion.MODEL_FILE).read_bytes() == before
        assert registry["candidates"] == registry_before["candidates"]
        assert registry["champion"] == registry_before["champion"]
        assert settings.scenario_model_path.exists()
        pipeline.prepare_evaluation(settings, strict=False)

    def test_a_fresh_registry_is_built_from_the_files(
        self, trained: config.Settings, synthetic_development: pd.DataFrame, tmp_path: Path
    ) -> None:
        settings = _copy_models(trained, tmp_path)
        settings.candidates_registry_path.unlink()
        pipeline.train_scenario(synthetic_development, settings=settings)
        pipeline.prepare_evaluation(settings, strict=False)

    def test_a_registry_cannot_be_rebuilt_over_another_champion(
        self, trained: config.Settings, synthetic_development: pd.DataFrame, tmp_path: Path
    ) -> None:
        settings = _copy_models(trained, tmp_path)
        settings.candidates_registry_path.unlink()
        manifest = json.loads((settings.champion_dir / champion.MANIFEST_FILE).read_text("utf-8"))
        path = settings.candidates_dir / f"{manifest['model']['name']}.joblib"
        path.write_bytes(b"not the champion")
        with pytest.raises(pipeline.EvaluationNotReadyError, match="not the champion"):
            pipeline.train_scenario(synthetic_development, settings=settings)
