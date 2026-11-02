"""Global interpretation of the frozen champion, aggregated by original variable.

It reads rows of the test period, only for interpreting and only after the single execution, as the
protocol says. Nothing that comes out of here can change the model.
"""

from typing import Any

import joblib
import pandas as pd
from numpy.typing import NDArray
from sklearn.inspection import permutation_importance

from hotel_cancellation import champion, config, data_loader, features, schema

FOREST_NAME = "random_forest"
PERMUTATION_REPEATS = 10
SHAP_ROWS = 2000


def _by_variable(encoded_names: list[str], values: NDArray[Any]) -> pd.Series:
    """Add up what each encoded column contributes to its original variable."""
    series = pd.Series(values, index=[features.original_feature(name) for name in encoded_names])
    return series.groupby(level=0).sum().reindex(list(schema.INPUT_FEATURES), fill_value=0.0)


def _shap_by_variable(forest: Any, rows: pd.DataFrame, shap_rows: int, seed: int) -> pd.Series:
    """Average the absolute SHAP value of each original variable, in the base model's probability.

    The contributions of the columns of a variable are added per booking before taking the absolute
    value, because together they say what that variable did to that booking.
    """
    import shap

    sample = rows.sample(n=min(shap_rows, len(rows)), random_state=seed)
    preprocessor = forest.named_steps[features.PREPROCESS_STEP]
    encoded = preprocessor.transform(sample)
    names = list(preprocessor.get_feature_names_out())
    explainer = shap.TreeExplainer(forest.named_steps[features.MODEL_STEP])
    values = explainer.shap_values(encoded, check_additivity=False)
    positive = values[1] if isinstance(values, list) else values
    if positive.ndim == 3:
        positive = positive[:, :, 1]
    contributions = pd.DataFrame(positive, columns=[features.original_feature(n) for n in names])
    per_variable = contributions.T.groupby(level=0).sum().T
    return per_variable.abs().mean().reindex(list(schema.INPUT_FEATURES), fill_value=0.0)


def explain(
    test: pd.DataFrame,
    *,
    settings: config.Settings,
    n_repeats: int = PERMUTATION_REPEATS,
    shap_rows: int = SHAP_ROWS,
    n_jobs: int = -1,
    seed: int = 42,
) -> pd.DataFrame:
    """Measure the importance of each original variable with the three methods of the protocol.

    - `forest_importance`: `feature_importances_` of the Random Forest base model, learned on
      arrivals before 2016T4. It is biased towards variables with many categories.
    - `permutation_mean` and `permutation_std`: drop in ROC-AUC of the served champion on the test
      when a variable is shuffled.
    - `shap_mean_abs`: mean absolute SHAP value of the Random Forest base model on a sample of the
      test, in probability before calibrating.

    Args:
        test: Rows of the test period.
        settings: Where the models are and where to write.
        n_repeats: Shuffles per variable.
        shap_rows: Rows of the SHAP sample.
        n_jobs: Parallel shuffles.
        seed: Seed of the shuffles and of the sample.

    Returns:
        A row per input feature, by descending permutation importance.

    Raises:
        TestNotExecutedError: If the single execution has not happened yet.
    """
    champion.assert_test_executed(settings)
    served = champion.load_champion(settings.champion_dir).model
    calibrated_forest = joblib.load(settings.candidates_dir / f"{FOREST_NAME}.joblib")
    forest = features.base_pipeline(calibrated_forest)

    rows = data_loader.coerce_features(test)
    target = test[schema.TARGET]
    shuffled = permutation_importance(
        served,
        rows,
        target,
        scoring="roc_auc",
        n_repeats=n_repeats,
        n_jobs=n_jobs,
        random_state=seed,
    )
    preprocessor = forest.named_steps[features.PREPROCESS_STEP]
    forest_importance = _by_variable(
        list(preprocessor.get_feature_names_out()),
        forest.named_steps[features.MODEL_STEP].feature_importances_,
    )
    table = pd.DataFrame(
        {
            "variable": list(schema.INPUT_FEATURES),
            "forest_importance": forest_importance.to_numpy(),
            "permutation_mean": shuffled.importances_mean,
            "permutation_std": shuffled.importances_std,
            "shap_mean_abs": _shap_by_variable(forest, rows, shap_rows, seed).to_numpy(),
        }
    ).sort_values("permutation_mean", ascending=False, ignore_index=True)

    settings.interpretation_dir.mkdir(parents=True, exist_ok=True)
    table.round(6).to_csv(
        settings.interpretation_dir / "importance.csv", index=False, lineterminator="\n"
    )
    return table
