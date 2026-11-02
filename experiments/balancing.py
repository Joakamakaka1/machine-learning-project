"""Does balancing the classes help? SMOTE, undersampling and class weights against nothing.

Balancing changes the prevalence that the model learns, which should distort its probabilities. The
Brier score, the calibration slope and the cost measure that, even where the ROC-AUC does not move.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier

from experiments import common
from hotel_cancellation import data_loader, features, schema

_NEIGHBORS = 5


def _logistic(**overrides: Any) -> LogisticRegression:
    return LogisticRegression(max_iter=2000, random_state=common.SEED, **overrides)


def _smote() -> SMOTE:
    # On encoded rows, distances only mean something when the numbers are scaled.
    neighbors = NearestNeighbors(n_neighbors=_NEIGHBORS + 1, n_jobs=-1)
    return SMOTE(k_neighbors=neighbors, random_state=common.SEED)


def _undersampler() -> RandomUnderSampler:
    return RandomUnderSampler(random_state=common.SEED)


def _resampled(
    make_estimator: Callable[[], Any], make_sampler: Callable[[], Any]
) -> common.FitPredict:
    def fit_predict(train: pd.DataFrame, valid: pd.DataFrame) -> NDArray[np.float64]:
        preprocessor = features.build_preprocessor(features.Family.SCALED)
        encoded = preprocessor.fit_transform(
            data_loader.coerce_features(train), train[schema.TARGET]
        )
        resampled, outcomes = make_sampler().fit_resample(encoded, train[schema.TARGET])
        model = make_estimator().fit(resampled, outcomes)
        probabilities = model.predict_proba(
            preprocessor.transform(data_loader.coerce_features(valid))
        )
        return np.asarray(probabilities[:, 1])

    return fit_predict


def _weighted_xgboost(train: pd.DataFrame, valid: pd.DataFrame) -> NDArray[np.float64]:
    outcomes = train[schema.TARGET]
    ratio = float((outcomes == 0).sum() / (outcomes == 1).sum())

    def weighted() -> XGBClassifier:
        return common.reference_xgboost(scale_pos_weight=ratio)

    return np.asarray(common.tabular_predictor(weighted)(train, valid))


def balancing(development: pd.DataFrame) -> common.ExperimentResult:
    """Compare the balancing methods on the reference model and on the logistic regression.

    Args:
        development: Development rows.

    Returns:
        No balancing on the reference model as the reference of the gaps.
    """
    scaled = features.Family.SCALED
    return common.compare(
        "balancing",
        development,
        [
            common.Variant("xgboost_none", common.tabular_predictor(common.reference_xgboost)),
            common.Variant("xgboost_scale_pos_weight", _weighted_xgboost),
            common.Variant("xgboost_smote", _resampled(common.reference_xgboost, _smote)),
            common.Variant(
                "xgboost_undersampling", _resampled(common.reference_xgboost, _undersampler)
            ),
            common.Variant("logistic_none", common.tabular_predictor(_logistic, family=scaled)),
            common.Variant(
                "logistic_class_weight",
                common.tabular_predictor(lambda: _logistic(class_weight="balanced"), family=scaled),
            ),
            common.Variant("logistic_smote", _resampled(_logistic, _smote)),
            common.Variant("logistic_undersampling", _resampled(_logistic, _undersampler)),
        ],
    )
