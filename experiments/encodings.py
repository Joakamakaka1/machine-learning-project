"""How should categories be encoded? One-hot against target and ordinal encoding.

`TargetEncoder` cross-fits inside `fit_transform`. Its default splitter shuffles the rows, and many
of them are exact duplicates, which would send identical bookings to both sides of its inner folds.
So the splitter is explicit: contiguous folds of rows in arrival order, where identical bookings
share a date and mostly a fold. The shuffled version stays as a contrast.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder

from experiments import common
from hotel_cancellation.data import data_loader, features, schema

HIGH_CARDINALITY = ("agent",)
_INNER_FOLDS = 5
_UNKNOWN_CATEGORY = -1
_MISSING_CATEGORY = -2


def _predictor(categorical_steps: Sequence[tuple[str, Any, list[str]]]) -> common.FitPredict:
    def fit_predict(train: pd.DataFrame, valid: pd.DataFrame) -> NDArray[np.float64]:
        preprocessor = ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline([("impute", SimpleImputer(strategy="median"))]),
                    list(schema.NUMERIC_FEATURES),
                ),
                *categorical_steps,
            ]
        )
        pipeline = features.build_pipeline(preprocessor, common.reference_xgboost())
        pipeline.fit(data_loader.coerce_features(train), train[schema.TARGET])
        probabilities = pipeline.predict_proba(data_loader.coerce_features(valid))[:, 1]
        return np.asarray(probabilities)

    return fit_predict


def _low_cardinality() -> list[str]:
    return [c for c in schema.CATEGORICAL_FEATURES if c not in HIGH_CARDINALITY]


def _one_hot() -> OneHotEncoder:
    return OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=features.MIN_CATEGORY_FREQUENCY,
        sparse_output=False,
    )


def _target_encoder_contiguous() -> TargetEncoder:
    return TargetEncoder(target_type="binary", cv=KFold(n_splits=_INNER_FOLDS, shuffle=False))


def _target_encoder_shuffled() -> TargetEncoder:
    splitter = StratifiedKFold(n_splits=_INNER_FOLDS, shuffle=True, random_state=common.SEED)
    return TargetEncoder(target_type="binary", cv=splitter)


def encodings(development: pd.DataFrame) -> common.ExperimentResult:
    """Compare the encodings on the reference model.

    Args:
        development: Development rows.

    Returns:
        One-hot as the reference, target encoding of `agent` and of every category, and ordinal.
    """
    all_categories = list(schema.CATEGORICAL_FEATURES)
    ordinal = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=_UNKNOWN_CATEGORY,
        encoded_missing_value=_MISSING_CATEGORY,
    )
    return common.compare(
        "encodings",
        development,
        [
            common.Variant(
                "one_hot_min_frequency", _predictor([("categories", _one_hot(), all_categories)])
            ),
            common.Variant(
                "target_agent_contiguous_folds",
                _predictor(
                    [
                        ("low", _one_hot(), _low_cardinality()),
                        ("high", _target_encoder_contiguous(), list(HIGH_CARDINALITY)),
                    ]
                ),
            ),
            common.Variant(
                "target_all_contiguous_folds",
                _predictor([("categories", _target_encoder_contiguous(), all_categories)]),
            ),
            common.Variant(
                "target_agent_shuffled_folds",
                _predictor(
                    [
                        ("low", _one_hot(), _low_cardinality()),
                        ("high", _target_encoder_shuffled(), list(HIGH_CARDINALITY)),
                    ]
                ),
            ),
            common.Variant("ordinal", _predictor([("categories", ordinal, all_categories)])),
        ],
    )
