"""Preprocessing per model family and the guard against forbidden columns.

Only scikit-learn classes appear here, so a saved model does not need this package to load.
"""

from collections.abc import Sequence
from enum import StrEnum
from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hotel_cancellation import schema

MIN_CATEGORY_FREQUENCY = 50
PREPROCESS_STEP = "preprocess"
MODEL_STEP = "model"


class Family(StrEnum):
    """Model families by what they need from the numeric inputs."""

    SCALED = "scaled"
    UNSCALED = "unscaled"


class LeakageError(ValueError):
    """A column that must not reach a model would reach it."""


def build_preprocessor(
    family: Family,
    *,
    numeric: Sequence[str] = schema.NUMERIC_FEATURES,
    categorical: Sequence[str] = schema.CATEGORICAL_FEATURES,
    min_frequency: int = MIN_CATEGORY_FREQUENCY,
) -> ColumnTransformer:
    """Build the preprocessing of a model family.

    A missing category is a category of its own: a booking without `agent` is a direct booking.

    Args:
        family: Whether the numbers must be scaled.
        numeric: Numeric columns. The experiments are the only callers that change them.
        categorical: Categorical columns.
        min_frequency: Categories below this count are grouped together.

    Returns:
        An unfitted transformer that drops every other column.
    """
    numeric_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if family is Family.SCALED:
        numeric_steps.append(("scale", StandardScaler()))
    encoder = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=min_frequency,
        sparse_output=False,
    )
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), list(numeric)),
            ("categorical", encoder, list(categorical)),
        ]
    )


def build_pipeline(preprocessor: ColumnTransformer, estimator: Any) -> Pipeline:
    """Chain a preprocessor and an estimator under the names the rest of the package expects."""
    return Pipeline([(PREPROCESS_STEP, preprocessor), (MODEL_STEP, estimator)])


def base_pipeline(model: Any) -> Any:
    """Get to the pipeline inside the calibration wrapper."""
    if isinstance(model, CalibratedClassifierCV):
        model = model.estimator
    if isinstance(model, FrozenEstimator):
        model = model.estimator
    return model


def original_feature(encoded_name: str) -> str:
    """Go from a column of the preprocessor output back to its input variable.

    Args:
        encoded_name: Such as `numeric__lead_time` or `categorical__agent_9`.

    Returns:
        The name of the input variable.

    Raises:
        ValueError: If the column does not come from an input variable.
    """
    _, _, name = encoded_name.partition("__")
    if name in schema.NUMERIC_FEATURES:
        return name
    for feature in schema.CATEGORICAL_FEATURES:
        if name.startswith(f"{feature}_"):
            return feature
    raise ValueError(f"{encoded_name} does not come from an input feature.")


def assert_no_forbidden_inputs(model: Any) -> None:
    """Fail if a column outside `schema.INPUT_FEATURES` reaches the model.

    Every model is checked before training and before being saved. It reads what the preprocessor
    really uses, not what the frame holds: extra columns in a frame are dropped on the way in.

    Args:
        model: A pipeline, fitted or not, optionally inside a calibration wrapper.

    Raises:
        LeakageError: Naming each offending column and why it is out.
    """
    pipeline = base_pipeline(model)
    if not isinstance(pipeline, Pipeline) or PREPROCESS_STEP not in pipeline.named_steps:
        raise LeakageError(f"Cannot inspect the inputs: there is no {PREPROCESS_STEP} step.")
    preprocessor = pipeline.named_steps[PREPROCESS_STEP]
    used = {column for _, _, columns in preprocessor.transformers for column in columns}
    offending = sorted(used.difference(schema.INPUT_FEATURES))
    if offending:
        reasons = (
            f"{column}: {schema.FORBIDDEN_FEATURES.get(column, 'not an input feature.')}"
            for column in offending
        )
        raise LeakageError("Columns that must not reach a model:\n- " + "\n- ".join(reasons))
