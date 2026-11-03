"""Hyperparameter search, model selection, calibration choice and final training.

Each step uses the windows of `docs/protocolo.md`. This module only sees development rows: it
refuses any arrival of the test period.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import RandomizedSearchCV

from hotel_cancellation import config, models
from hotel_cancellation.data import data_loader, features, schema
from hotel_cancellation.evaluation import evaluator

logger = logging.getLogger(__name__)

DEFAULT_SEED = 42
NO_CALIBRATION = "none"
CALIBRATION_METHODS: tuple[str, ...] = ("sigmoid", "isotonic")
MIN_BRIER_GAIN = 0.001


class TestPeriodError(ValueError):
    """Rows of the test period reached the training code."""

    __test__ = False


@dataclass(frozen=True)
class SearchOutcome:
    """Result of searching one model: its best configuration and how it scored in each window."""

    name: str
    params: dict[str, Any]
    window_scores: pd.DataFrame
    all_configurations: pd.DataFrame


@dataclass(frozen=True)
class Selection:
    """Result of the one standard error rule."""

    best: str
    champion: str
    table: pd.DataFrame


@dataclass(frozen=True)
class CalibrationChoice:
    """Calibration comparison of one model, measured on the fourth quarter of 2016."""

    method: str
    briers: dict[str, float]
    metrics: dict[str, float]
    threshold_contrast: list[dict[str, float]]


@dataclass(frozen=True)
class Candidate:
    """A model trained on the final windows, ready to be evaluated exactly as it is."""

    name: str
    params: dict[str, Any]
    calibration: CalibrationChoice
    model: Any


def scenario_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Give the inputs of the informative scenario: the 19 features and the special requests."""
    values = pd.to_numeric(rows[schema.SPECIAL_REQUESTS]).astype("float64")
    return data_loader.coerce_features(rows).assign(**{schema.SPECIAL_REQUESTS: values})


def select_by_one_standard_error(window_scores: Mapping[str, Sequence[float]]) -> Selection:
    """Apply the one standard error rule on paired differences per window.

    Windows differ much more than models do, so each model is compared with the best one window by
    window. Among the models whose mean gap does not exceed the standard error of those paired
    differences, the first one of `models.SIMPLICITY_ORDER` wins.

    Args:
        window_scores: ROC-AUC of each model per window, in the same window order.

    Returns:
        The best model, the chosen one and the table behind the decision.

    Raises:
        ValueError: If a model misses a score or models do not share the windows.
    """
    frame = pd.DataFrame({name: pd.Series(list(v)) for name, v in window_scores.items()})
    for name, scores in frame.items():
        if scores.isna().any():
            lengths = {model: len(values) for model, values in window_scores.items()}
            raise ValueError(f"{name} misses a score, or windows differ between models: {lengths}")

    best = str(frame.mean().idxmax())
    gaps = frame.rsub(frame[best], axis=0)
    mean_gap = gaps.mean()
    paired_se = gaps.std(ddof=1) / np.sqrt(len(frame))
    within_rule = mean_gap <= paired_se
    ordered = [name for name in models.SIMPLICITY_ORDER if name in frame.columns]
    champion = next(name for name in ordered if within_rule[name])
    table = pd.DataFrame(
        {
            "model": ordered,
            "mean_roc_auc": frame[ordered].mean().to_numpy(),
            "std_roc_auc": frame[ordered].std(ddof=1).to_numpy(),
            "mean_gap": mean_gap[ordered].to_numpy(),
            "paired_se": paired_se[ordered].to_numpy(),
            "within_rule": within_rule[ordered].to_numpy(),
        }
    )
    return Selection(best=best, champion=champion, table=table)


def choose_calibration_method(briers: Mapping[str, float]) -> str:
    """Pick the calibration with the lowest Brier, unless it improves by less than 0.001.

    Args:
        briers: Brier score of each method, `none` included.

    Returns:
        The chosen method.
    """
    best = min(CALIBRATION_METHODS, key=lambda method: briers[method])
    gain = briers[NO_CALIBRATION] - briers[best]
    return best if gain >= MIN_BRIER_GAIN else NO_CALIBRATION


class ModelTrainer:
    """Searches, selects and trains the candidates on development rows.

    Args:
        development: Clean rows with arrivals before the test period, in any order.
        catalog: Models to compare. The baseline is always trained on top of them.
        seed: Seed of the search and of each model.
        n_jobs: Parallel fits of each search. The specification can force a serial search.

    Raises:
        TestPeriodError: If any row arrives within the test period.
    """

    def __init__(
        self,
        development: pd.DataFrame,
        *,
        catalog: Sequence[models.ModelSpec],
        seed: int = DEFAULT_SEED,
        n_jobs: int = -1,
    ) -> None:
        """Keep the rows in arrival order: Keras validates on the last ones."""
        last_arrival = development[schema.ARRIVAL_DATE].max()
        if last_arrival >= config.TEST_START:
            raise TestPeriodError(
                f"Arrival on {last_arrival.date()} is not before {config.TEST_START.date()}: "
                "training only accepts development rows."
            )
        self.development = development.sort_values(schema.ARRIVAL_DATE, kind="stable").reset_index(
            drop=True
        )
        self.catalog = tuple(catalog)
        self.seed = seed
        self.n_jobs = n_jobs
        self._features = data_loader.coerce_features(self.development)
        self._target = self.development[schema.TARGET]
        self._dates = self.development[schema.ARRIVAL_DATE]

    def search(self, spec: models.ModelSpec) -> SearchOutcome:
        """Search one model over the expanding windows, keeping every score per window.

        Args:
            spec: The model and its search space.

        Returns:
            The configuration with the highest mean ROC-AUC, with the scores of all of them.
        """
        pipeline = spec.pipeline(self.seed)
        features.assert_no_forbidden_inputs(pipeline)
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=dict(spec.space),
            n_iter=spec.n_iter,
            cv=data_loader.expanding_windows(self.development),
            scoring=evaluator.validation_scorer,
            refit=False,
            random_state=self.seed,
            n_jobs=self.n_jobs if spec.parallel_search else 1,
            error_score="raise",
        )
        search.fit(self._features, self._target)
        return _search_outcome(spec.name, search.cv_results_)

    def search_all(self) -> dict[str, SearchOutcome]:
        """Search the baseline and every model of the catalog."""
        return {spec.name: self.search(spec) for spec in (models.BASELINE, *self.catalog)}

    def compare_calibration(
        self, spec: models.ModelSpec, params: Mapping[str, Any]
    ) -> CalibrationChoice:
        """Fit a base before 2016T3, calibrate on 2016T3 and measure on 2016T4.

        Args:
            spec: The model.
            params: Its chosen hyperparameters.

        Returns:
            The Brier of each method, the chosen one and its measurements on 2016T4.
        """
        cut = config.CALIBRATION_SELECTION
        base = self._fit_base(spec, params, cut.base_end)
        calibration_rows = self._between(cut.base_end, cut.calibration_end)
        measured_rows = self._between(cut.calibration_end, cut.evaluation_end or config.TEST_START)
        measured_features = self._features[measured_rows]
        measured_target = self._target[measured_rows]

        fitted = {NO_CALIBRATION: base}
        for method in CALIBRATION_METHODS:
            fitted[method] = _calibrate(
                base, method, self._features[calibration_rows], self._target[calibration_rows]
            )
        probabilities = {
            method: model.predict_proba(measured_features)[:, 1] for method, model in fitted.items()
        }
        briers = {
            method: float(brier_score_loss(measured_target, scores))
            for method, scores in probabilities.items()
        }
        chosen = choose_calibration_method(briers)
        return CalibrationChoice(
            method=chosen,
            briers=briers,
            metrics=evaluator.validation_metrics(
                measured_target, probabilities[chosen], measured_features
            ),
            threshold_contrast=_threshold_contrast(measured_target, probabilities[chosen]),
        )

    def fit_served(self, spec: models.ModelSpec, params: Mapping[str, Any], method: str) -> Any:
        """Train what gets evaluated and served: base before 2016T4, calibrator on 2016T4.

        Args:
            spec: The model.
            params: Its chosen hyperparameters.
            method: Calibration method, or `none`.

        Returns:
            The fitted model.
        """
        cut = config.CHAMPION_CUT
        base = self._fit_base(spec, params, cut.base_end)
        if method == NO_CALIBRATION:
            return base
        rows = self._between(cut.base_end, cut.calibration_end)
        return _calibrate(base, method, self._features[rows], self._target[rows])

    def fit_special_requests_scenario(
        self, spec: models.ModelSpec, params: Mapping[str, Any], method: str
    ) -> Any:
        """Train the informative scenario on the windows of the champion, plus the forbidden input.

        This is the only model that skips the leakage guard, on purpose, and the guard rejects it
        when saving a champion. The protocol publishes it next to the official figure.

        Args:
            spec: The model of the champion.
            params: Its hyperparameters.
            method: Its calibration method, or `none`.

        Returns:
            The fitted model. It expects the columns of `scenario_frame`.
        """
        cut = config.CHAMPION_CUT
        frame = scenario_frame(self.development)
        preprocessor = features.build_preprocessor(
            spec.family, numeric=(*schema.NUMERIC_FEATURES, schema.SPECIAL_REQUESTS)
        )
        pipeline = features.build_pipeline(preprocessor, spec.estimator(self.seed))
        rows = self._dates < cut.base_end
        base = pipeline.set_params(**params).fit(frame[rows], self._target[rows])
        if method == NO_CALIBRATION:
            return base
        calibration_rows = self._between(cut.base_end, cut.calibration_end)
        return _calibrate(base, method, frame[calibration_rows], self._target[calibration_rows])

    def train_candidates(self, outcomes: Mapping[str, SearchOutcome]) -> dict[str, Candidate]:
        """Choose the calibration of each model and train its final version.

        The baseline is trained too, without calibration, so that the test can compare against it.

        Args:
            outcomes: Result of the search of each model to train.

        Returns:
            The trained candidates by name.
        """
        candidates = {}
        for spec in (models.BASELINE, *self.catalog):
            if spec.name not in outcomes:
                continue
            params = outcomes[spec.name].params
            choice = self.compare_calibration(spec, params)
            if spec.name == models.BASELINE.name:
                choice = _without_calibration(choice)
            logger.info("%s: calibration %s, %s", spec.name, choice.method, choice.briers)
            model = self.fit_served(spec, params, choice.method)
            candidates[spec.name] = Candidate(spec.name, dict(params), choice, model)
        return candidates

    def _fit_base(
        self, spec: models.ModelSpec, params: Mapping[str, Any], end: pd.Timestamp
    ) -> Any:
        pipeline = spec.pipeline(self.seed).set_params(**params)
        features.assert_no_forbidden_inputs(pipeline)
        rows = self._dates < end
        return pipeline.fit(self._features[rows], self._target[rows])

    def _between(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        return (self._dates >= start) & (self._dates < end)


def _calibrate(base: Any, method: str, rows: pd.DataFrame, target: pd.Series) -> Any:
    return CalibratedClassifierCV(FrozenEstimator(base), method=method).fit(rows, target)


def _without_calibration(choice: CalibrationChoice) -> CalibrationChoice:
    """A prior has no ranking to calibrate, and it stays out of the comparison."""
    return CalibrationChoice(
        method=NO_CALIBRATION,
        briers=choice.briers,
        metrics=choice.metrics,
        threshold_contrast=choice.threshold_contrast,
    )


def _threshold_contrast(y_true: pd.Series, probabilities: Any) -> list[dict[str, float]]:
    rows = []
    for scenario in evaluator.COST_SCENARIOS:
        threshold = evaluator.optimal_threshold(scenario)
        best_threshold, best_cost = evaluator.empirical_best_threshold(
            y_true, probabilities, scenario
        )
        rows.append(
            {
                "false_positive_cost": scenario.false_positive,
                "false_negative_cost": scenario.false_negative,
                "threshold": threshold,
                "cost_at_threshold": evaluator.cost_per_booking(
                    y_true, probabilities, threshold, scenario
                ),
                "empirical_best_threshold": best_threshold,
                "cost_at_empirical_best": best_cost,
            }
        )
    return rows


def _search_outcome(name: str, results: Mapping[str, Any]) -> SearchOutcome:
    means = np.asarray(results["mean_test_roc_auc"], dtype=float)
    best = int(np.nanargmax(means))
    windows = {
        window.name: {
            metric: float(results[f"split{position}_test_{metric}"][best])
            for metric in evaluator.VALIDATION_METRICS
        }
        for position, window in enumerate(config.EXPANDING_WINDOWS)
    }
    configurations = pd.DataFrame(
        {
            "params": [repr(params) for params in results["params"]],
            **{
                f"mean_{metric}": results[f"mean_test_{metric}"]
                for metric in evaluator.VALIDATION_METRICS
            },
            "std_roc_auc": results["std_test_roc_auc"],
            "mean_fit_seconds": results["mean_fit_time"],
        }
    )
    return SearchOutcome(
        name=name,
        params=dict(results["params"][best]),
        window_scores=pd.DataFrame.from_dict(windows, orient="index"),
        all_configurations=configurations,
    )
