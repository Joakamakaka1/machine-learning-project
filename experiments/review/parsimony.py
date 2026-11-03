"""Confirm with the champion's configuration what the reference XGBoost suggested.

The ablation says that most inputs add nothing on their own, and that recency weights might help.
Both are tried here with the Random Forest, because the experiments used XGBoost. A smaller model
would also depend on fewer values of the snapshot.

Run it with `uv run python -m experiments.review.parsimony`.
"""

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from experiments import common
from experiments.review import data_treatments, feature_tiers, harness
from hotel_cancellation import data_loader, features, schema

CORE_5: tuple[str, ...] = (
    "lead_time",
    "agent",
    "deposit_type",
    "customer_type",
    "previous_bookings_not_canceled",
)
CORE_8: tuple[str, ...] = (*CORE_5, "previous_cancellations", "adr", "market_segment")


def _forest() -> Any:
    pipeline = common.tuned_pipeline("random_forest").set_params(model__n_jobs=-1)
    return pipeline.named_steps[features.MODEL_STEP]


def _weighted_forest(half_life_days: float) -> harness.FitPredict:
    def fit_predict(train: pd.DataFrame, valid: pd.DataFrame) -> NDArray[np.float64]:
        pipeline = common.tuned_pipeline("random_forest").set_params(model__n_jobs=-1)
        _, weights = data_treatments.recency_weights(half_life_days)(train)
        pipeline.fit(
            data_loader.coerce_features(train),
            train[schema.TARGET],
            **{f"{features.MODEL_STEP}__sample_weight": weights},
        )
        return np.asarray(pipeline.predict_proba(data_loader.coerce_features(valid))[:, 1])

    return fit_predict


VARIANTS: tuple[tuple[str, harness.FitPredict], ...] = (
    ("forest_19", feature_tiers.predictor(schema.INPUT_FEATURES, make_model=_forest)),
    ("forest_core_8", feature_tiers.predictor(CORE_8, make_model=_forest)),
    ("forest_core_5", feature_tiers.predictor(CORE_5, make_model=_forest)),
    ("forest_19_recency_365", _weighted_forest(365.0)),
    ("forest_19_recency_180", _weighted_forest(180.0)),
)


def main() -> None:
    """Run the variants on both schemes."""
    rows = harness.load_rows()
    pd.set_option("display.width", 240, "display.max_columns", 40)
    for scheme, windows in (("m12", harness.M12), ("q4", harness.Q4)):
        by_window = harness.run_variants(f"parsimony {scheme}", rows, VARIANTS, windows)
        table = harness.add_holm(harness.compare_to_reference(by_window, "forest_19"))
        harness.save(f"parsimony_{scheme}_by_window", by_window)
        harness.save(f"parsimony_{scheme}", table)
        print(table.round(4).T.to_string(), flush=True)


if __name__ == "__main__":
    main()
