"""Baselines and models, from the simplest to the most complex, on the same windows.

Run it with `uv run python -m experiments.review.ladder`.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from experiments import common
from experiments.review import harness
from hotel_cancellation import data_loader, features, schema


def prior(train: pd.DataFrame, valid: pd.DataFrame) -> NDArray[np.float64]:
    """Predict the training prevalence for everybody."""
    return np.full(len(valid), float(train[schema.TARGET].mean()))


def rule_deposit(train: pd.DataFrame, valid: pd.DataFrame) -> NDArray[np.float64]:
    """Predict the training rate of each deposit type: the simplest rule with any signal."""
    rates = train.groupby("deposit_type")[schema.TARGET].mean()
    return np.asarray(
        valid["deposit_type"].map(rates).fillna(train[schema.TARGET].mean()), dtype=float
    )


def rule_deposit_lead(train: pd.DataFrame, valid: pd.DataFrame) -> NDArray[np.float64]:
    """Predict the training rate per deposit type and lead time band."""
    bands = [-1, 7, 30, 90, 180, 365, 10_000]
    train_key = train["deposit_type"] + pd.cut(train["lead_time"], bands).astype("str")
    valid_key = valid["deposit_type"] + pd.cut(valid["lead_time"], bands).astype("str")
    rates = train[schema.TARGET].groupby(train_key).mean()
    return np.asarray(valid_key.map(rates).fillna(train[schema.TARGET].mean()), dtype=float)


def _logistic(c: float = 1.0) -> LogisticRegression:
    return LogisticRegression(C=c, max_iter=3000, random_state=common.SEED)


def small_logistic(columns: Sequence[str], categorical: Sequence[str]) -> harness.FitPredict:
    """Build a logistic regression on a few columns, with the protocol's preprocessing."""

    def fit_predict(train: pd.DataFrame, valid: pd.DataFrame) -> NDArray[np.float64]:
        numeric = [column for column in columns if column not in categorical]
        pipeline = features.build_pipeline(
            features.build_preprocessor(
                features.Family.SCALED, numeric=numeric, categorical=list(categorical)
            ),
            _logistic(),
        )
        x_train = data_loader.coerce_features(train)
        x_valid = data_loader.coerce_features(valid)
        pipeline.fit(x_train, train[schema.TARGET])
        return np.asarray(pipeline.predict_proba(x_valid)[:, 1])

    return fit_predict


def tuned_logistic(train: pd.DataFrame, valid: pd.DataFrame) -> NDArray[np.float64]:
    """Fit the published logistic regression with its tuned hyperparameter."""
    pipeline = common.tuned_pipeline("logistic_regression")
    pipeline.fit(data_loader.coerce_features(train), train[schema.TARGET])
    return np.asarray(pipeline.predict_proba(data_loader.coerce_features(valid))[:, 1])


def spline_logistic(train: pd.DataFrame, valid: pd.DataFrame) -> NDArray[np.float64]:
    """Give the logistic regression non-linear numbers: splines of lead time and rate."""
    smooth = ["lead_time", "adr"]
    other_numeric = [column for column in schema.NUMERIC_FEATURES if column not in smooth]
    preprocessor = ColumnTransformer(
        [
            (
                "splines",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        (
                            "spline",
                            SplineTransformer(degree=3, n_knots=6, extrapolation="constant"),
                        ),
                    ]
                ),
                smooth,
            ),
            (
                "numeric",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                other_numeric,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=features.MIN_CATEGORY_FREQUENCY,
                    sparse_output=False,
                ),
                list(schema.CATEGORICAL_FEATURES),
            ),
        ]
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", _logistic(0.1))])
    pipeline.fit(data_loader.coerce_features(train), train[schema.TARGET])
    return np.asarray(pipeline.predict_proba(data_loader.coerce_features(valid))[:, 1])


def tuned_model(name: str) -> harness.FitPredict:
    """Fit a published model with its tuned hyperparameters, using every core."""

    def fit_predict(train: pd.DataFrame, valid: pd.DataFrame) -> NDArray[np.float64]:
        pipeline = common.tuned_pipeline(name)
        if name in ("random_forest", "xgboost"):
            pipeline.set_params(model__n_jobs=-1)
        pipeline.fit(data_loader.coerce_features(train), train[schema.TARGET])
        return np.asarray(pipeline.predict_proba(data_loader.coerce_features(valid))[:, 1])

    return fit_predict


def hist_gradient(train: pd.DataFrame, valid: pd.DataFrame) -> NDArray[np.float64]:
    """Fit scikit-learn's native boosting, with defaults, as a contrast without one-hot encoding."""
    frame_train = data_loader.coerce_features(train)
    frame_valid = data_loader.coerce_features(valid)
    categorical = list(schema.CATEGORICAL_FEATURES)
    for column in categorical:
        counts = frame_train[column].value_counts()
        frequent = counts[counts >= features.MIN_CATEGORY_FREQUENCY].index
        codes = {value: position for position, value in enumerate(frequent)}
        frame_train[column] = frame_train[column].map(codes).astype(float)
        frame_valid[column] = frame_valid[column].map(codes).astype(float)
    model = HistGradientBoostingClassifier(
        categorical_features=categorical,
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        random_state=common.SEED,
    )
    model.fit(frame_train, train[schema.TARGET])
    return np.asarray(model.predict_proba(frame_valid)[:, 1])


VARIANTS: tuple[tuple[str, harness.FitPredict], ...] = (
    ("prior", prior),
    ("rule_deposit", rule_deposit),
    ("rule_deposit_lead", rule_deposit_lead),
    (
        "logistic_deposit_lead",
        small_logistic(("lead_time", "deposit_type"), ("deposit_type",)),
    ),
    (
        "logistic_5_variables",
        small_logistic(
            ("lead_time", "deposit_type", "market_segment", "hotel", "customer_type"),
            ("deposit_type", "market_segment", "hotel", "customer_type"),
        ),
    ),
    ("logistic_19_tuned", tuned_logistic),
    ("logistic_19_splines", spline_logistic),
    ("decision_tree_tuned", tuned_model("decision_tree")),
    ("random_forest_champion", tuned_model("random_forest")),
    ("xgboost_tuned", tuned_model("xgboost")),
    ("hist_gradient_native", hist_gradient),
)


def main() -> None:
    """Run the ladder on both schemes and compare every step with the champion's configuration."""
    rows = harness.load_rows()
    pd.set_option("display.width", 240, "display.max_columns", 40)
    columns = [
        "variant",
        "roc_auc",
        "roc_auc_std",
        "roc_auc_no_deposit",
        "roc_auc_distinct",
        "log_loss",
    ]
    for scheme, windows in (("q4", harness.Q4), ("m12", harness.M12)):
        by_window = harness.run_variants(f"ladder {scheme}", rows, VARIANTS, windows)
        table = harness.compare_to_reference(by_window, "random_forest_champion")
        harness.save(f"ladder_{scheme}_by_window", by_window)
        harness.save(f"ladder_{scheme}", table)
        extended = [*columns, "roc_auc_gap", "roc_auc_low", "roc_auc_high"]
        print(table[extended].round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
