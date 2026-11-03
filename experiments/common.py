"""Shared machinery of the experiments: variants compared window by window on development rows.

A variant receives the training and the validation rows of a window and returns the probabilities
for the validation rows. Both frames hold every raw column, so a variant decides what it uses, but
the metrics always come from `evaluator.validation_metrics`, exactly as in the model search.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from hotel_cancellation import champion, config, data_loader, evaluator, features, models, schema

SEED = 42

# Shared by every experiment so that their figures are comparable.
REFERENCE_XGBOOST: dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 2,
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": SEED,
}

SUMMARY_METRICS: tuple[str, ...] = (
    "roc_auc",
    "roc_auc_unique",
    "roc_auc_no_deposit",
    "roc_auc_non_refund",
    "pr_auc",
    "brier",
    "calibration_slope",
    "f1",
    "cost_1_3",
)

FitPredict = Callable[[pd.DataFrame, pd.DataFrame], ArrayLike]


@dataclass(frozen=True)
class Variant:
    """One of the alternatives an experiment compares."""

    name: str
    fit_predict: FitPredict


@dataclass(frozen=True)
class ExperimentResult:
    """Scores of each variant per window, and their summary against the first variant."""

    name: str
    by_window: pd.DataFrame
    summary: pd.DataFrame


def reference_xgboost(**overrides: Any) -> XGBClassifier:
    """Build the reference model of the experiments."""
    return XGBClassifier(**{**REFERENCE_XGBOOST, **overrides})


def load_development() -> pd.DataFrame:
    """Load the development rows of the real dataset."""
    raw = data_loader.load_raw()
    data_loader.validate_raw(raw)
    return data_loader.split_temporal(data_loader.clean(raw)).development


def model_frame(
    rows: pd.DataFrame, *, extra_numeric: Sequence[str] = (), extra_categorical: Sequence[str] = ()
) -> pd.DataFrame:
    """Coerce the input features and add the extra columns an experiment wants to try."""
    frame = data_loader.coerce_features(rows)
    for column in extra_numeric:
        frame[column] = pd.to_numeric(rows[column]).astype("float64")
    for column in extra_categorical:
        frame[column] = rows[column].astype("str")
    return frame


def tabular_predictor(
    make_estimator: Callable[[], Any],
    *,
    family: features.Family = features.Family.UNSCALED,
    extra_numeric: Sequence[str] = (),
    extra_categorical: Sequence[str] = (),
) -> FitPredict:
    """Build a variant from an estimator, with the standard preprocessing of its family."""
    numeric = (*schema.NUMERIC_FEATURES, *extra_numeric)
    categorical = (*schema.CATEGORICAL_FEATURES, *extra_categorical)

    def fit_predict(train: pd.DataFrame, valid: pd.DataFrame) -> NDArray[np.float64]:
        pipeline = features.build_pipeline(
            features.build_preprocessor(family, numeric=numeric, categorical=categorical),
            make_estimator(),
        )
        frames = {
            "extra_numeric": extra_numeric,
            "extra_categorical": extra_categorical,
        }
        pipeline.fit(model_frame(train, **frames), train[schema.TARGET])
        return np.asarray(pipeline.predict_proba(model_frame(valid, **frames))[:, 1])

    return fit_predict


def tuned_pipeline(name: str) -> Pipeline:
    """Build a model of the catalog with the hyperparameters found by `hotel-ml tune`.

    Args:
        name: Model name in the catalog.

    Returns:
        An unfitted pipeline.
    """
    spec = next(spec for spec in models.model_catalog("full") if spec.name == name)
    content = json.loads(config.get_settings().best_params_path.read_text("utf-8"))
    return spec.pipeline(SEED).set_params(**content["models"][name]["params"])


def champion_name() -> str:
    """Name of the model that the selection chose."""
    manifest = json.loads(
        (config.get_settings().champion_dir / champion.MANIFEST_FILE).read_text("utf-8")
    )
    name: str = manifest["model"]["name"]
    return name


def compare(name: str, development: pd.DataFrame, variants: Sequence[Variant]) -> ExperimentResult:
    """Run every variant on the four expanding windows.

    Args:
        name: Name of the experiment, which also names its files.
        development: Development rows.
        variants: Alternatives to compare. The first one is the reference of the paired gaps.

    Returns:
        The scores per window and the summary.

    Raises:
        ValueError: If a row belongs to the test period.
    """
    dates = development[schema.ARRIVAL_DATE]
    if (dates >= config.TEST_START).any():
        raise ValueError(f"Experiments only accept arrivals before {config.TEST_START.date()}.")

    rows = []
    for window in config.EXPANDING_WINDOWS:
        train = development[dates < window.start]
        valid = development[(dates >= window.start) & (dates < window.end)]
        valid_features = data_loader.coerce_features(valid)
        for variant in variants:
            probabilities = variant.fit_predict(train, valid)
            scores = evaluator.validation_metrics(
                valid[schema.TARGET], probabilities, valid_features
            )
            rows.append({"variant": variant.name, "window": window.name, **scores})
            print(f"{name} | {window.name} | {variant.name}: {scores['roc_auc']:.4f}", flush=True)
    by_window = pd.DataFrame(rows)
    return ExperimentResult(name, by_window, summarize(by_window, variants[0].name))


def summarize(by_window: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Average each metric across windows, and compare each variant with the reference.

    `gap` is the mean of the paired ROC-AUC differences against the reference, positive when the
    variant is better, and `paired_se` is its standard error.
    """
    by_variant = by_window.groupby("variant", sort=False)
    summary = by_variant[list(SUMMARY_METRICS)].mean()
    summary.insert(1, "roc_auc_std", by_variant["roc_auc"].std(ddof=1))
    aucs = by_window.pivot_table(index="window", columns="variant", values="roc_auc")
    gaps = aucs.sub(aucs[reference], axis=0)
    summary.insert(2, "gap", gaps.mean())
    summary.insert(3, "paired_se", gaps.std(ddof=1) / np.sqrt(len(gaps)))
    return summary.reset_index()


def save(result: ExperimentResult) -> None:
    """Store the result in `outputs/experiments/` and print its summary."""
    folder = config.get_settings().experiments_dir
    folder.mkdir(parents=True, exist_ok=True)
    result.by_window.round(5).to_csv(
        folder / f"{result.name}_by_window.csv", index=False, lineterminator="\n"
    )
    result.summary.round(5).to_csv(folder / f"{result.name}.csv", index=False, lineterminator="\n")
    print(result.summary.round(4).to_string(index=False), flush=True)
