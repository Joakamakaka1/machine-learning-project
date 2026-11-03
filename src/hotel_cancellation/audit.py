"""Reproducible audit of how much the first figures were inflated, in four steps.

Each step removes one reason for the inflation. It reads rows of the test period, so it only runs
after the single execution. Nothing that comes out of here can change the champion.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from hotel_cancellation import champion, config, schema

# Hyperparameters of the starting point: very deep trees, without much regularization.
STARTING_XGBOOST: Mapping[str, Any] = {
    "n_estimators": 600,
    "max_depth": 16,
    "learning_rate": 0.03,
    "subsample": 0.9,
    "colsample_bytree": 1.0,
}
_RANDOM_TEST_SIZE = 0.2
_SEED = 42
_MAX_CATEGORIES_TO_ONE_HOT = 8
_MISSING_CATEGORY = "none"

# The start already left out these columns, which restate the outcome or come from the check-in.
_STARTING_DROPS = (
    "reservation_status",
    "reservation_status_date",
    "arrival_date_year",
    "required_car_parking_spaces",
    "assigned_room_type",
)
_DOCUMENTED_LEAKS = ("country", "booking_changes")
_NOT_INPUTS = (schema.TARGET, schema.ARRIVAL_DATE)


def _starting_frames(
    train: pd.DataFrame, test: pd.DataFrame, drops: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare the variables as at the start: native categories, with `none` for missing."""
    removed = {*drops, *_NOT_INPUTS}
    columns = [column for column in schema.RAW_COLUMNS if column not in removed]
    x_train, x_test = train[columns].copy(), test[columns].copy()
    for column in columns:
        if pd.api.types.is_numeric_dtype(x_train[column]):
            continue
        filled_train = x_train[column].fillna(_MISSING_CATEGORY)
        categories = pd.CategoricalDtype(sorted(filled_train.unique()))
        x_train[column] = filled_train.astype(categories)
        x_test[column] = x_test[column].fillna(_MISSING_CATEGORY).astype(categories)
    return x_train, x_test


def _share_already_in_train(train: pd.DataFrame, test: pd.DataFrame) -> float:
    """Measure the rows of the test that are identical to some row of training."""
    columns = [c for c in schema.RAW_COLUMNS if c not in ("reservation_status_date",)]
    seen = pd.util.hash_pandas_object(train[columns], index=False)
    return float(pd.util.hash_pandas_object(test[columns], index=False).isin(seen).mean())


def _measure_step(
    step: str,
    label: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    drops: Sequence[str],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    x_train, x_test = _starting_frames(train, test, drops)
    model = XGBClassifier(
        **params,
        enable_categorical=True,
        tree_method="hist",
        max_cat_to_onehot=_MAX_CATEGORIES_TO_ONE_HOT,
        n_jobs=-1,
        random_state=_SEED,
    ).fit(x_train, train[schema.TARGET])
    probabilities = model.predict_proba(x_test)[:, 1]
    actual = test[schema.TARGET]
    return {
        "step": step,
        "label": label,
        "model": "xgboost_starting",
        "train_rows": len(train),
        "test_rows": len(test),
        "test_rows_already_in_train": _share_already_in_train(train, test),
        "roc_auc": float(roc_auc_score(actual, probabilities)),
        "pr_auc": float(average_precision_score(actual, probabilities)),
        "brier": float(brier_score_loss(actual, probabilities)),
    }


def _official_step(settings: config.Settings) -> dict[str, Any]:
    """Copy the published figure: the official model is never measured a second time."""
    manifest = champion.load_champion(settings.champion_dir).manifest
    name = manifest["model"]["name"]
    metrics = pd.read_csv(settings.final_evaluation_dir / "metrics.csv")
    row = metrics[
        (metrics["model"] == name)
        & (metrics["subset"] == "all")
        & (metrics["threshold_name"] == "decision")
    ].iloc[0]
    return {
        "step": "official_model",
        "label": "Modelo oficial: 19 variables, sin calendario ni variables acumuladas",
        "model": name,
        "train_rows": manifest["development_rows"],
        "test_rows": int(row["rows"]),
        "test_rows_already_in_train": float("nan"),
        "roc_auc": float(row["roc_auc"]),
        "pr_auc": float(row["pr_auc"]),
        "brier": float(row["brier"]),
    }


def audit_inflation(
    cleaned: pd.DataFrame,
    *,
    settings: config.Settings,
    params: Mapping[str, Any] = STARTING_XGBOOST,
) -> pd.DataFrame:
    """Measure the four steps from the inflated figure to the official one.

    Args:
        cleaned: Every clean row, the test period included.
        settings: Where the published results are and where to write.
        params: Hyperparameters of XGBoost in the first three steps.

    Returns:
        A row per step.

    Raises:
        TestNotExecutedError: If the single execution has not happened yet.
    """
    champion.assert_test_executed(settings)
    is_test = cleaned[schema.ARRIVAL_DATE] >= config.TEST_START
    temporal_train, temporal_test = cleaned[~is_test], cleaned[is_test]
    random_train, random_test = train_test_split(
        cleaned,
        test_size=_RANDOM_TEST_SIZE,
        random_state=_SEED,
        stratify=cleaned[schema.TARGET],
    )
    table = pd.DataFrame(
        [
            _measure_step(
                "random_split_starting_variables",
                "Partición aleatoria, variables de partida",
                random_train,
                random_test,
                _STARTING_DROPS,
                params,
            ),
            _measure_step(
                "temporal_split_starting_variables",
                "Partición temporal, variables de partida",
                temporal_train,
                temporal_test,
                _STARTING_DROPS,
                params,
            ),
            _measure_step(
                "temporal_split_without_documented_leaks",
                "Partición temporal, sin country ni booking_changes",
                temporal_train,
                temporal_test,
                (*_STARTING_DROPS, *_DOCUMENTED_LEAKS),
                params,
            ),
            _official_step(settings),
        ]
    )
    settings.audit_dir.mkdir(parents=True, exist_ok=True)
    rounded = table.assign(
        **{
            column: table[column].astype(float).round(6).replace({np.inf: np.nan})
            for column in ("test_rows_already_in_train", "pr_auc", "brier")
        }
    )
    rounded.to_csv(settings.audit_dir / "inflation.csv", index=False, lineterminator="\n")
    return table
