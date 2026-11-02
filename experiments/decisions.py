"""Experiments behind the variable and model decisions of the plan.

Run them with `uv run python -m experiments.run_all`, which only reads arrivals before 2017.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from xgboost import XGBClassifier

from experiments import common
from hotel_cancellation import data_loader, features, models, schema

CALENDAR_NUMERIC = ("arrival_date_week_number", "arrival_date_day_of_month")
CALENDAR_MONTH = "arrival_date_month"
CALENDAR_YEAR = "arrival_date_year"
SPECIAL_REQUESTS = "total_of_special_requests"
_NOT_BOOKING_CONTENT = {"reservation_status", "reservation_status_date", schema.ARRIVAL_DATE}


def _reference() -> XGBClassifier:
    return common.reference_xgboost()


def calendar(development: pd.DataFrame) -> common.ExperimentResult:
    """Does the calendar help? Without it, the model must generalize to dates it has not seen."""
    return common.compare(
        "calendar",
        development,
        [
            common.Variant("19_variables", common.tabular_predictor(_reference)),
            common.Variant(
                "plus_month",
                common.tabular_predictor(_reference, extra_categorical=(CALENDAR_MONTH,)),
            ),
            common.Variant(
                "plus_month_week_day",
                common.tabular_predictor(
                    _reference,
                    extra_numeric=CALENDAR_NUMERIC,
                    extra_categorical=(CALENDAR_MONTH,),
                ),
            ),
            common.Variant(
                "plus_year_month_week_day",
                common.tabular_predictor(
                    _reference,
                    extra_numeric=(CALENDAR_YEAR, *CALENDAR_NUMERIC),
                    extra_categorical=(CALENDAR_MONTH,),
                ),
            ),
        ],
    )


def company(development: pd.DataFrame) -> common.ExperimentResult:
    """Does `company` help? Without an effect, the simpler model keeps it out."""
    return common.compare(
        "company",
        development,
        [
            common.Variant("19_variables", common.tabular_predictor(_reference)),
            common.Variant(
                "plus_company", common.tabular_predictor(_reference, extra_categorical=("company",))
            ),
        ],
    )


def _deduplicated(fit_predict: common.FitPredict) -> common.FitPredict:
    def run(train: pd.DataFrame, valid: pd.DataFrame) -> NDArray[np.float64]:
        content = [column for column in train.columns if column not in _NOT_BOOKING_CONTENT]
        return np.asarray(fit_predict(train.drop_duplicates(subset=content), valid))

    return run


def duplicates(development: pd.DataFrame) -> common.ExperimentResult:
    """Does removing identical bookings from training help? Validation always keeps all rows."""
    all_rows = common.tabular_predictor(_reference)
    return common.compare(
        "duplicates",
        development,
        [
            common.Variant("train_all_rows", all_rows),
            common.Variant("train_unique_rows", _deduplicated(all_rows)),
        ],
    )


def _per_hotel(train: pd.DataFrame, valid: pd.DataFrame) -> NDArray[np.float64]:
    probabilities = np.zeros(len(valid))
    hotels_in_valid = valid["hotel"].to_numpy()
    for hotel in sorted(schema.ALLOWED_VALUES["hotel"]):
        is_hotel = hotels_in_valid == hotel
        if not is_hotel.any():
            continue
        hotel_train = train[train["hotel"] == hotel]
        pipeline = features.build_pipeline(
            features.build_preprocessor(
                features.Family.UNSCALED,
                categorical=[c for c in schema.CATEGORICAL_FEATURES if c != "hotel"],
            ),
            _reference(),
        )
        pipeline.fit(data_loader.coerce_features(hotel_train), hotel_train[schema.TARGET])
        hotel_valid = data_loader.coerce_features(valid[is_hotel])
        probabilities[is_hotel] = pipeline.predict_proba(hotel_valid)[:, 1]
    return probabilities


def per_hotel(development: pd.DataFrame) -> common.ExperimentResult:
    """Is one model per hotel better than a global one?"""
    return common.compare(
        "per_hotel",
        development,
        [
            common.Variant("global_model", common.tabular_predictor(_reference)),
            common.Variant("one_model_per_hotel", _per_hotel),
        ],
    )


def tree_depth(development: pd.DataFrame) -> common.ExperimentResult:
    """Do very deep trees generalize better than regularized ones?"""

    def deep() -> XGBClassifier:
        return common.reference_xgboost(
            n_estimators=600,
            max_depth=16,
            learning_rate=0.03,
            colsample_bytree=1.0,
            min_child_weight=1,
            reg_lambda=1,
        )

    def shallow() -> XGBClassifier:
        return common.reference_xgboost(
            n_estimators=300, max_depth=3, min_child_weight=10, reg_lambda=5
        )

    return common.compare(
        "tree_depth",
        development,
        [
            common.Variant("depth_6_regularized", common.tabular_predictor(_reference)),
            common.Variant("depth_16", common.tabular_predictor(deep)),
            common.Variant("depth_3", common.tabular_predictor(shallow)),
        ],
    )


def _family(name: str) -> features.Family:
    return next(spec.family for spec in models.model_catalog("full") if spec.name == name)


def special_requests(development: pd.DataFrame, *, tuned: str | None) -> common.ExperimentResult:
    """Publish both scenarios: `total_of_special_requests` as a leak, and as if it were available.

    Args:
        development: Development rows.
        tuned: Model of the catalog to repeat the comparison with, usually the champion.
    """
    variants = [
        common.Variant("xgboost_19_variables", common.tabular_predictor(_reference)),
        common.Variant(
            "xgboost_plus_special_requests",
            common.tabular_predictor(_reference, extra_numeric=(SPECIAL_REQUESTS,)),
        ),
    ]
    if tuned is not None:

        def tuned_estimator() -> object:
            return common.tuned_pipeline(tuned).named_steps[features.MODEL_STEP]

        family = _family(tuned)
        variants += [
            common.Variant(
                f"{tuned}_19_variables", common.tabular_predictor(tuned_estimator, family=family)
            ),
            common.Variant(
                f"{tuned}_plus_special_requests",
                common.tabular_predictor(
                    tuned_estimator, family=family, extra_numeric=(SPECIAL_REQUESTS,)
                ),
            ),
        ]
    return common.compare("special_requests", development, variants)
