from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, roc_auc_score

from hotel_cancellation import config, data_loader, features, model_trainer, models, schema
from tests.conftest import synthetic_raw


@pytest.fixture
def trainer(synthetic_development: pd.DataFrame) -> model_trainer.ModelTrainer:
    return model_trainer.ModelTrainer(
        synthetic_development, catalog=models.model_catalog("fast"), seed=0, n_jobs=1
    )


def _catalog_spec(name: str) -> models.ModelSpec:
    return next(spec for spec in models.model_catalog("fast") if spec.name == name)


def _rows(development: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp) -> pd.DataFrame:
    dates = development[schema.ARRIVAL_DATE]
    return development[(dates < end) & (dates >= start if start is not None else True)]


def _x_y(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return data_loader.coerce_features(rows), rows[schema.TARGET]


class TestOneStandardError:
    SCORES: ClassVar[dict[str, list[float]]] = {
        "xgboost": [0.80, 0.82, 0.84, 0.86],
        # Always just below the best: the shift is real, although windows vary much more than it.
        "logistic_regression": [0.79, 0.815, 0.83, 0.855],
        # Mean gap of 0.005 with a paired standard error of 0.0065.
        "decision_tree": [0.80, 0.80, 0.85, 0.85],
        "random_forest": [0.70, 0.70, 0.70, 0.70],
    }

    def test_the_simplest_model_within_the_rule_wins(self) -> None:
        selection = model_trainer.select_by_one_standard_error(self.SCORES)
        assert selection.best == "xgboost"
        assert selection.champion == "decision_tree"

    def test_the_rule_uses_paired_differences_per_window(self) -> None:
        table = model_trainer.select_by_one_standard_error(self.SCORES).table.set_index("model")
        assert table.loc["logistic_regression", "mean_gap"] == pytest.approx(0.0075)
        assert table.loc["logistic_regression", "paired_se"] == pytest.approx(0.0014434, abs=1e-6)
        assert not table.loc["logistic_regression", "within_rule"]
        assert table.loc["decision_tree", "mean_gap"] == pytest.approx(0.005)
        assert table.loc["decision_tree", "paired_se"] == pytest.approx(0.0064550, abs=1e-6)
        assert table.loc["decision_tree", "within_rule"]
        assert not table.loc["random_forest", "within_rule"]
        assert table.loc["xgboost", "within_rule"]

    def test_logistic_regression_is_accepted_when_it_stays_within(self) -> None:
        scores = {
            "logistic_regression": [0.80, 0.80, 0.85, 0.85],
            "xgboost": [0.80, 0.82, 0.84, 0.86],
        }
        assert model_trainer.select_by_one_standard_error(scores).champion == "logistic_regression"

    def test_the_best_model_wins_when_nothing_else_stays_within(self) -> None:
        scores = {"logistic_regression": [0.70, 0.71, 0.72, 0.73], "mlp": [0.80, 0.81, 0.82, 0.83]}
        assert model_trainer.select_by_one_standard_error(scores).champion == "mlp"

    def test_it_rejects_windows_of_different_lengths(self) -> None:
        with pytest.raises(ValueError, match="windows"):
            model_trainer.select_by_one_standard_error({"xgboost": [0.8, 0.8], "mlp": [0.8]})

    def test_it_rejects_a_missing_score(self) -> None:
        with pytest.raises(ValueError, match="xgboost"):
            model_trainer.select_by_one_standard_error({"xgboost": [0.8, float("nan")]})


class TestCalibrationRule:
    def test_a_small_gain_does_not_justify_calibrating(self) -> None:
        briers = {"none": 0.2000, "sigmoid": 0.1995, "isotonic": 0.1992}
        assert model_trainer.choose_calibration_method(briers) == "none"

    def test_the_lowest_brier_wins_when_the_gain_is_enough(self) -> None:
        briers = {"none": 0.2000, "sigmoid": 0.1980, "isotonic": 0.1985}
        assert model_trainer.choose_calibration_method(briers) == "sigmoid"

    def test_calibration_that_hurts_is_rejected(self) -> None:
        briers = {"none": 0.2000, "sigmoid": 0.2100, "isotonic": 0.2050}
        assert model_trainer.choose_calibration_method(briers) == "none"


class TestDevelopmentOnly:
    def test_trainer_refuses_rows_of_the_test_period(self) -> None:
        cleaned = data_loader.clean(synthetic_raw(rows_per_month=10))
        with pytest.raises(model_trainer.TestPeriodError, match="2017-01-01"):
            model_trainer.ModelTrainer(cleaned, catalog=models.model_catalog("fast"))


class TestSearch:
    def test_scores_are_those_of_the_expanding_windows(
        self, trainer: model_trainer.ModelTrainer, synthetic_development: pd.DataFrame
    ) -> None:
        spec = _catalog_spec("logistic_regression")
        outcome = trainer.search(spec)
        assert list(outcome.window_scores.index) == [w.name for w in config.EXPANDING_WINDOWS]
        for window in config.EXPANDING_WINDOWS:
            x_train, y_train = _x_y(_rows(synthetic_development, None, window.start))
            x_valid, y_valid = _x_y(_rows(synthetic_development, window.start, window.end))
            fitted = spec.pipeline(seed=0).set_params(**outcome.params).fit(x_train, y_train)
            expected = roc_auc_score(y_valid, fitted.predict_proba(x_valid)[:, 1])
            assert outcome.window_scores.loc[window.name, "roc_auc"] == pytest.approx(expected)

    def test_best_params_have_the_highest_mean_score_of_the_search(
        self, trainer: model_trainer.ModelTrainer
    ) -> None:
        outcome = trainer.search(_catalog_spec("decision_tree"))
        assert outcome.all_configurations["mean_roc_auc"].max() == pytest.approx(
            outcome.window_scores["roc_auc"].mean()
        )
        assert len(outcome.all_configurations) == 2
        assert outcome.params["model__max_depth"] in (3, 5)

    def test_arrival_order_does_not_depend_on_the_input_order(
        self, trainer: model_trainer.ModelTrainer, synthetic_development: pd.DataFrame
    ) -> None:
        shuffled = synthetic_development.sample(frac=1, random_state=0)
        other = model_trainer.ModelTrainer(
            shuffled, catalog=models.model_catalog("fast"), seed=0, n_jobs=1
        )
        spec = _catalog_spec("logistic_regression")
        pd.testing.assert_frame_equal(
            trainer.search(spec).window_scores, other.search(spec).window_scores
        )

    def test_baseline_only_learns_the_prevalence(self, trainer: model_trainer.ModelTrainer) -> None:
        outcome = trainer.search(models.BASELINE)
        assert (outcome.window_scores["roc_auc"] == 0.5).all()


class TestCalibrationComparison:
    def test_it_fits_on_the_third_quarter_and_measures_on_the_fourth(
        self, trainer: model_trainer.ModelTrainer, synthetic_development: pd.DataFrame
    ) -> None:
        spec = _catalog_spec("xgboost")
        params = {"model__learning_rate": 0.1}
        cut = config.CALIBRATION_SELECTION
        choice = trainer.compare_calibration(spec, params)

        x_base, y_base = _x_y(_rows(synthetic_development, None, cut.base_end))
        x_fit, y_fit = _x_y(_rows(synthetic_development, cut.base_end, cut.calibration_end))
        x_eval, y_eval = _x_y(_rows(synthetic_development, cut.calibration_end, config.TEST_START))
        base = spec.pipeline(seed=0).set_params(**params).fit(x_base, y_base)
        sigmoid = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid").fit(x_fit, y_fit)

        assert choice.briers["none"] == pytest.approx(
            brier_score_loss(y_eval, base.predict_proba(x_eval)[:, 1])
        )
        assert choice.briers["sigmoid"] == pytest.approx(
            brier_score_loss(y_eval, sigmoid.predict_proba(x_eval)[:, 1])
        )
        assert set(choice.briers) == {"none", "sigmoid", "isotonic"}
        assert choice.method == model_trainer.choose_calibration_method(choice.briers)

    def test_it_contrasts_each_cost_scenario_on_the_fourth_quarter(
        self, trainer: model_trainer.ModelTrainer
    ) -> None:
        choice = trainer.compare_calibration(_catalog_spec("logistic_regression"), {})
        contrast = choice.threshold_contrast
        assert [row["threshold"] for row in contrast] == pytest.approx([1 / 3, 1 / 4, 1 / 6])
        for row in contrast:
            assert row["cost_at_empirical_best"] <= row["cost_at_threshold"]


class TestServedModel:
    def test_base_uses_arrivals_before_q4_and_the_calibrator_uses_q4(
        self, trainer: model_trainer.ModelTrainer, synthetic_development: pd.DataFrame
    ) -> None:
        spec = _catalog_spec("logistic_regression")
        cut = config.CHAMPION_CUT
        served = trainer.fit_served(spec, {"model__C": 1.0}, "sigmoid")

        x_base, y_base = _x_y(_rows(synthetic_development, None, cut.base_end))
        x_fit, y_fit = _x_y(_rows(synthetic_development, cut.base_end, cut.calibration_end))
        base = spec.pipeline(seed=0).set_params(model__C=1.0).fit(x_base, y_base)
        expected = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid").fit(x_fit, y_fit)
        probe = data_loader.coerce_features(synthetic_development.head(50))
        np.testing.assert_allclose(served.predict_proba(probe), expected.predict_proba(probe))

    def test_without_calibration_the_fourth_quarter_is_not_used(
        self, trainer: model_trainer.ModelTrainer, synthetic_development: pd.DataFrame
    ) -> None:
        spec = _catalog_spec("logistic_regression")
        served = trainer.fit_served(spec, {"model__C": 1.0}, "none")
        x_base, y_base = _x_y(_rows(synthetic_development, None, config.CHAMPION_CUT.base_end))
        base = spec.pipeline(seed=0).set_params(model__C=1.0).fit(x_base, y_base)
        probe = data_loader.coerce_features(synthetic_development.head(50))
        np.testing.assert_allclose(served.predict_proba(probe), base.predict_proba(probe))

    def test_training_all_candidates_includes_the_baseline(
        self, trainer: model_trainer.ModelTrainer
    ) -> None:
        outcomes = trainer.search_all()
        candidates = trainer.train_candidates(outcomes)
        assert set(candidates) == {spec.name for spec in models.model_catalog("fast")} | {"dummy"}
        assert candidates["dummy"].calibration.method == "none"
        probe = data_loader.coerce_features(trainer.development.head(5))
        for candidate in candidates.values():
            assert candidate.model.predict_proba(probe).shape == (5, 2)
            features.assert_no_forbidden_inputs(candidate.model)


class TestLeakageGuard:
    def test_searching_with_a_forbidden_column_fails_before_fitting(
        self, trainer: model_trainer.ModelTrainer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        leaking = features.build_preprocessor(
            features.Family.SCALED, numeric=("lead_time", "total_of_special_requests")
        )
        monkeypatch.setattr(features, "build_preprocessor", lambda *_, **__: leaking)
        with pytest.raises(features.LeakageError, match="total_of_special_requests"):
            trainer.search(_catalog_spec("logistic_regression"))

    def test_final_training_with_a_forbidden_column_fails_before_fitting(
        self, trainer: model_trainer.ModelTrainer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        leaking = features.build_preprocessor(features.Family.SCALED, numeric=("country",))
        monkeypatch.setattr(features, "build_preprocessor", lambda *_, **__: leaking)
        with pytest.raises(features.LeakageError, match="country"):
            trainer.fit_served(_catalog_spec("logistic_regression"), {}, "none")
