"""Runtime smoke tests of the pandas 3 stack on synthetic data.

The dependency resolver only proves that versions can coexist. These tests prove that they work
together on the shapes this project feeds them: `str` columns with missing values.
"""

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

NUMERIC = ["lead_time", "adr"]
CATEGORICAL = ["hotel", "agent"]
ROWS = 400
# MLflow 3 stores sklearn models with skops, which refuses types it does not know yet.
XGBOOST_SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
]


def _encoder() -> OneHotEncoder:
    return OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=5,
        sparse_output=False,
    )


def _preprocessor(*, scale: bool = False) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), NUMERIC),
            ("categorical", _encoder(), CATEGORICAL),
        ]
    )


def _xgboost() -> XGBClassifier:
    return XGBClassifier(n_estimators=20, max_depth=3, n_jobs=1, random_state=0)


@pytest.fixture
def features() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    adr = rng.normal(100, 30, ROWS)
    adr[::37] = np.nan
    agent: list[str | None] = [str(value) for value in rng.choice(["1", "9", "240"], ROWS)]
    agent[::11] = [None] * len(agent[::11])
    return pd.DataFrame(
        {
            "lead_time": rng.integers(0, 300, ROWS).astype(float),
            "adr": adr,
            "hotel": rng.choice(["City Hotel", "Resort Hotel"], ROWS),
            "agent": agent,
        }
    )


@pytest.fixture
def target(features: pd.DataFrame) -> pd.Series:
    rng = np.random.default_rng(1)
    score = (
        features["lead_time"] / 300
        + (features["hotel"] == "City Hotel") * 0.5
        + rng.normal(0, 0.3, ROWS)
    )
    return (score > score.median()).astype(int).rename("is_canceled")


@pytest.fixture
def pipeline(features: pd.DataFrame, target: pd.Series) -> Pipeline:
    fitted = Pipeline([("preprocess", _preprocessor()), ("model", _xgboost())])
    return fitted.fit(features, target)


def test_pandas_infers_str_columns_that_select_dtypes_can_find(features: pd.DataFrame) -> None:
    assert pd.api.types.is_string_dtype(features["agent"])
    assert features["agent"].isna().sum() > 0
    selected = features.select_dtypes(include=["object", "string"]).columns
    assert list(selected) == CATEGORICAL


def test_encoder_accepts_str_columns_with_missing_values(features: pd.DataFrame) -> None:
    encoded = _encoder().fit_transform(features[CATEGORICAL])
    assert encoded.shape[0] == ROWS
    assert not np.isnan(encoded).any()


def test_imputer_and_encoder_chain_accepts_str_columns(features: pd.DataFrame) -> None:
    chain = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encode", _encoder()),
        ]
    )
    encoded = chain.fit_transform(features[CATEGORICAL])
    assert encoded.shape[0] == ROWS
    assert "agent_missing" in chain.named_steps["encode"].get_feature_names_out(CATEGORICAL)


def test_xgboost_pipeline_predicts_unseen_categories_and_survives_joblib(
    pipeline: Pipeline, features: pd.DataFrame, target: pd.Series, tmp_path: Path
) -> None:
    probabilities = pipeline.predict_proba(features)[:, 1]
    assert roc_auc_score(target, probabilities) > 0.7

    path = tmp_path / "model.joblib"
    joblib.dump(pipeline, path)
    restored = joblib.load(path)
    np.testing.assert_array_equal(restored.predict_proba(features)[:, 1], probabilities)

    unseen = features.head(3).assign(
        agent=pd.Series(["never-seen", "never-seen", None], dtype="str")
    )
    assert restored.predict_proba(unseen).shape == (3, 2)


def test_random_search_accepts_index_folds_and_several_metrics(
    features: pd.DataFrame, target: pd.Series
) -> None:
    folds = [
        (np.arange(0, 200), np.arange(200, 300)),
        (np.arange(0, 300), np.arange(300, 400)),
    ]
    search = RandomizedSearchCV(
        Pipeline([("preprocess", _preprocessor()), ("model", _xgboost())]),
        param_distributions={"model__max_depth": [2, 3, 4]},
        n_iter=2,
        cv=folds,
        scoring=["roc_auc", "average_precision"],
        refit=False,
        random_state=0,
        n_jobs=1,
    )
    search.fit(features, target)
    assert len(search.cv_results_["split1_test_roc_auc"]) == 2


@pytest.mark.slow
def test_keras_trains_on_encoded_str_columns(features: pd.DataFrame, target: pd.Series) -> None:
    keras = pytest.importorskip("keras")
    encoded = _preprocessor(scale=True).fit_transform(features, target).astype("float32")
    model = keras.Sequential(
        [
            keras.Input(shape=(encoded.shape[1],)),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy")
    history = model.fit(
        encoded, target.to_numpy(), epochs=2, validation_split=0.2, batch_size=32, verbose=0
    )
    assert math.isfinite(history.history["val_loss"][-1])
    probabilities = model.predict(encoded, verbose=0)
    assert probabilities.shape == (ROWS, 1)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


@pytest.mark.slow
def test_mlflow_logs_and_reloads_pipeline_with_sqlite(
    pipeline: Pipeline, features: pd.DataFrame, tmp_path: Path
) -> None:
    mlflow = pytest.importorskip("mlflow")
    signature = mlflow.models.infer_signature(features, pipeline.predict_proba(features))
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    try:
        experiment_id = mlflow.create_experiment(
            "smoke", artifact_location=(tmp_path / "artifacts").as_uri()
        )
        with mlflow.start_run(experiment_id=experiment_id):
            mlflow.log_metric("roc_auc", 0.5)
            info = mlflow.sklearn.log_model(
                pipeline,
                name="model",
                signature=signature,
                skops_trusted_types=XGBOOST_SKOPS_TRUSTED_TYPES,
            )
        restored = mlflow.sklearn.load_model(info.model_uri)
        pyfunc = mlflow.pyfunc.load_model(info.model_uri)
    finally:
        mlflow.set_tracking_uri("")

    expected = pipeline.predict_proba(features)
    np.testing.assert_allclose(restored.predict_proba(features), expected)
    assert len(pyfunc.predict(features)) == ROWS


@pytest.mark.slow
def test_shap_and_xgboost_contributions_add_up_to_the_margin(
    pipeline: Pipeline, features: pd.DataFrame
) -> None:
    shap = pytest.importorskip("shap")
    xgboost = pytest.importorskip("xgboost")
    encoded = pipeline.named_steps["preprocess"].transform(features)
    model = pipeline.named_steps["model"]
    margin = model.predict(encoded, output_margin=True)

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(encoded)
    np.testing.assert_allclose(values.sum(axis=1) + explainer.expected_value, margin, atol=1e-3)

    contributions = model.get_booster().predict(xgboost.DMatrix(encoded), pred_contribs=True)
    np.testing.assert_allclose(contributions.sum(axis=1), margin, atol=1e-3)


@pytest.mark.slow
def test_imbalanced_learn_resamples_encoded_str_columns(
    features: pd.DataFrame, target: pd.Series
) -> None:
    over_sampling = pytest.importorskip("imblearn.over_sampling")
    under_sampling = pytest.importorskip("imblearn.under_sampling")
    encoded = _preprocessor().fit_transform(features, target)
    imbalanced = target.to_numpy().copy()
    imbalanced[imbalanced.nonzero()[0][::2]] = 0

    _, oversampled = over_sampling.SMOTE(random_state=0).fit_resample(encoded, imbalanced)
    _, undersampled = under_sampling.RandomUnderSampler(random_state=0).fit_resample(
        encoded, imbalanced
    )
    assert oversampled.mean() == pytest.approx(0.5)
    assert undersampled.mean() == pytest.approx(0.5)
