"""Business value, interpretation, audit and figures, always on synthetic results."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hotel_cancellation import audit, champion, cli, config, evaluator, explain, pipeline
from hotel_cancellation.data import data_loader, features, schema
from tests.conftest import synthetic_raw

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory: pytest.TempPathFactory) -> config.Settings:
    root = tmp_path_factory.mktemp("evaluated")
    settings = config.Settings(
        data_dir=root / "data", models_dir=root / "models", outputs_dir=root / "outputs"
    )
    split = data_loader.split_temporal(data_loader.clean(synthetic_raw()))
    pipeline.run_all(split.development, settings=settings, profile="fast", seed=0)
    pipeline.evaluate(split.test, split.development, settings=settings, replicates=5, strict=False)
    return settings


@pytest.fixture(scope="module")
def test_rows() -> pd.DataFrame:
    return data_loader.split_temporal(data_loader.clean(synthetic_raw())).test


class TestBusinessValue:
    Y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    P = np.array([0.9, 0.2, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1])

    def test_it_gives_the_decisions_of_each_scenario(self) -> None:
        table: Any = evaluator.business_value_table(self.Y, self.P).set_index("threshold")
        assert list(table.index) == pytest.approx([1 / 3, 1 / 4, 1 / 6])
        central: Any = table.iloc[1]
        assert central["true_alarms"] == 1
        assert central["false_alarms"] == 1
        assert central["missed_cancellations"] == 2
        assert central["recall"] == pytest.approx(1 / 3)
        assert central["precision"] == pytest.approx(1 / 2)
        assert central["false_alarm_rate"] == pytest.approx(1 / 5)
        assert central["flagged_share"] == pytest.approx(2 / 8)
        assert central["cost_per_booking"] == pytest.approx((1 + 2 * 3) / 8)

    def test_a_lower_threshold_catches_more(self) -> None:
        table: Any = evaluator.business_value_table(self.Y, self.P)
        assert table["recall"].iloc[2] == pytest.approx(2 / 3)

    def test_it_compares_with_the_best_decision_without_a_model(self) -> None:
        table: Any = evaluator.business_value_table(self.Y, self.P).iloc[1]
        # Without a model: flag nobody costs 3 * 3 / 8; flag everybody costs 5 * 1 / 8.
        assert table["cost_without_model"] == pytest.approx(5 / 8)
        assert table["saving_per_booking"] == pytest.approx(5 / 8 - 7 / 8)

    def test_it_declares_the_costs_as_assumptions(self) -> None:
        assert evaluator.business_value_table(self.Y, self.P)["costs_are_assumptions"].all()


class TestOriginalFeature:
    @pytest.mark.parametrize(
        ("encoded", "expected"),
        [
            ("numeric__lead_time", "lead_time"),
            ("categorical__agent_9", "agent"),
            ("categorical__agent_infrequent_sklearn", "agent"),
            ("categorical__agent_nan", "agent"),
            ("categorical__market_segment_Online TA", "market_segment"),
            ("categorical__distribution_channel_TA/TO", "distribution_channel"),
        ],
    )
    def test_it_goes_back_to_the_variable(self, encoded: str, expected: str) -> None:
        assert features.original_feature(encoded) == expected

    def test_it_refuses_a_name_it_does_not_know(self) -> None:
        with pytest.raises(ValueError, match="country"):
            features.original_feature("categorical__country_PRT")


class TestExplain:
    @pytest.fixture(scope="class")
    def table(self, evaluated: config.Settings, test_rows: pd.DataFrame) -> pd.DataFrame:
        pytest.importorskip("shap")
        return explain.explain(
            test_rows, settings=evaluated, n_repeats=2, shap_rows=100, n_jobs=1, seed=0
        )

    def test_it_aggregates_every_method_by_original_variable(self, table: pd.DataFrame) -> None:
        assert set(table["variable"]) == set(schema.INPUT_FEATURES)
        for column in ("forest_importance", "permutation_mean", "permutation_std", "shap_mean_abs"):
            assert table[column].notna().all(), column
        assert table["forest_importance"].sum() == pytest.approx(1)

    def test_it_finds_the_variables_that_carry_the_signal(self, table: pd.DataFrame) -> None:
        for column in ("forest_importance", "permutation_mean", "shap_mean_abs"):
            top = set(table.nlargest(3, column)["variable"])
            assert {"deposit_type", "lead_time"} & top, column
        low = table.set_index("variable")["permutation_mean"]
        assert low["deposit_type"] > low["babies"]

    def test_it_writes_its_table(self, evaluated: config.Settings, table: pd.DataFrame) -> None:
        saved = pd.read_csv(evaluated.interpretation_dir / "importance.csv")
        assert set(saved["variable"]) == set(table["variable"])

    def test_it_refuses_to_read_the_test_before_the_execution(
        self, test_rows: pd.DataFrame, tmp_path: Path
    ) -> None:
        settings = config.Settings(outputs_dir=tmp_path / "outputs", models_dir=tmp_path / "models")
        with pytest.raises(champion.TestNotExecutedError):
            explain.explain(test_rows, settings=settings)


class TestAudit:
    @pytest.fixture(scope="class")
    def table(self, evaluated: config.Settings) -> pd.DataFrame:
        cleaned = data_loader.clean(synthetic_raw())
        return audit.audit_inflation(
            cleaned, settings=evaluated, params={"n_estimators": 20, "max_depth": 3}
        )

    def test_it_has_the_four_steps_in_order(self, table: pd.DataFrame) -> None:
        assert list(table["step"]) == [
            "random_split_starting_variables",
            "temporal_split_starting_variables",
            "temporal_split_without_documented_leaks",
            "official_model",
        ]
        assert table["label"].str.len().min() > 0

    def test_leaking_variables_inflate_the_first_steps(self, table: pd.DataFrame) -> None:
        # On synthetic rows, the accumulating variables copy the outcome.
        aucs = table.set_index("step")["roc_auc"]
        assert aucs["random_split_starting_variables"] > 0.99
        assert aucs["temporal_split_starting_variables"] > 0.99
        assert aucs["temporal_split_without_documented_leaks"] > 0.99
        assert aucs["official_model"] < 0.95

    def test_the_official_step_is_the_published_figure(
        self, table: pd.DataFrame, evaluated: config.Settings
    ) -> None:
        metrics = pd.read_csv(evaluated.final_evaluation_dir / "metrics.csv")
        manifest = champion.load_champion(evaluated.champion_dir).manifest
        official = metrics[
            (metrics["model"] == manifest["model"]["name"])
            & (metrics["subset"] == "all")
            & (metrics["threshold_name"] == "decision")
        ]["roc_auc"].iloc[0]
        row = table.set_index("step").loc["official_model"]
        assert row["roc_auc"] == official
        assert row["test_rows"] > 0

    def test_it_measures_how_much_of_the_random_test_is_already_in_train(
        self, table: pd.DataFrame
    ) -> None:
        shares = table.set_index("step")["test_rows_already_in_train"]
        assert 0 <= shares["random_split_starting_variables"] <= 1
        assert shares["temporal_split_starting_variables"] <= 1

    def test_it_writes_its_table(self, evaluated: config.Settings, table: pd.DataFrame) -> None:
        saved = pd.read_csv(evaluated.audit_dir / "inflation.csv")
        assert list(saved["step"]) == list(table["step"])

    def test_it_refuses_to_read_the_test_before_the_execution(self, tmp_path: Path) -> None:
        settings = config.Settings(outputs_dir=tmp_path / "outputs", models_dir=tmp_path / "models")
        with pytest.raises(champion.TestNotExecutedError):
            audit.audit_inflation(data_loader.clean(synthetic_raw()), settings=settings)


@pytest.mark.slow
class TestFigures:
    @pytest.fixture(scope="class")
    def saved(self, evaluated: config.Settings) -> list[Path]:
        pytest.importorskip("matplotlib")
        from hotel_cancellation import plots

        pytest.importorskip("shap")
        explain.explain(
            data_loader.split_temporal(data_loader.clean(synthetic_raw())).test,
            settings=evaluated,
            n_repeats=2,
            shap_rows=50,
            n_jobs=1,
            seed=0,
        )
        audit.audit_inflation(
            data_loader.clean(synthetic_raw()),
            settings=evaluated,
            params={"n_estimators": 5, "max_depth": 2},
        )
        return plots.save_all(evaluated)

    def test_every_figure_of_the_report_is_written(self, saved: list[Path]) -> None:
        assert {path.stem for path in saved} == {
            "roc",
            "precision_recall",
            "intervalos_roc_auc",
            "matrices_confusion_campeon",
            "matrices_confusion_modelos",
            "calibracion",
            "importancia_variables",
            "roc_auc_por_mes",
            "prevalencia_por_mes",
            "deriva_psi",
            "coste_umbral",
            "inflacion",
        }
        assert all(path.stat().st_size > 5000 for path in saved)

    def test_confusion_matrix_gives_counts_and_percentages_in_spanish(self) -> None:
        pytest.importorskip("matplotlib")
        from hotel_cancellation import plots

        figure = plots.confusion_figure({"Campeón": (10, 30, 20, 40)})
        texts = " ".join(text.get_text() for text in figure.axes[0].texts)
        assert "10" in texts
        assert "10.0 %" in texts
        assert "40.0 %" in texts
        labels = [label.get_text() for label in figure.axes[0].get_xticklabels()]
        assert labels == ["No cancela", "Cancela"]

    def test_figures_can_be_regenerated_without_the_optional_tables(
        self, evaluated: config.Settings, tmp_path: Path
    ) -> None:
        pytest.importorskip("matplotlib")
        import shutil

        from hotel_cancellation import plots

        outputs = tmp_path / "outputs"
        shutil.copytree(evaluated.final_evaluation_dir, outputs / "final_evaluation")
        settings = evaluated.model_copy(update={"outputs_dir": outputs})
        saved = {path.stem for path in plots.save_all(settings)}
        assert "roc" in saved
        assert "importancia_variables" not in saved
        assert "inflacion" not in saved


class TestCommands:
    def _settings(self, evaluated: config.Settings, tmp_path: Path) -> config.Settings:
        return evaluated.model_copy(
            update={"outputs_dir": tmp_path / "outputs", "models_dir": evaluated.models_dir}
        )

    def test_explain_and_audit_refuse_before_the_execution(
        self, evaluated: config.Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = self._settings(evaluated, tmp_path)

        def fail() -> None:
            raise AssertionError("the test rows were read")

        monkeypatch.setattr(config, "get_settings", lambda: settings)
        monkeypatch.setattr(pipeline, "load_test", fail)
        monkeypatch.setattr(pipeline, "load_cleaned", fail)
        assert cli.main(["explain"]) == 1
        assert cli.main(["audit"]) == 1

    def test_figures_refuses_before_the_execution(
        self, evaluated: config.Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("matplotlib")
        monkeypatch.setattr(config, "get_settings", lambda: self._settings(evaluated, tmp_path))
        assert cli.main(["figures"]) == 1
