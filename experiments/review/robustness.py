"""Stability across seeds, sensitivity to the data, learning curve and ceiling of the dataset.

Run it with `uv run python -m experiments.review.robustness`.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from experiments import common
from experiments.review import harness
from hotel_cancellation import data_loader, features, schema

_FRACTIONS = (0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
_RF_SEED_PARAM = "model__random_state"


def _random_forest(seed: int) -> Any:
    return common.tuned_pipeline("random_forest").set_params(
        model__n_jobs=-1, **{_RF_SEED_PARAM: seed}
    )


def _xgboost(seed: int) -> Any:
    return common.tuned_pipeline("xgboost").set_params(model__n_jobs=-1, **{_RF_SEED_PARAM: seed})


def _fit_predict(model: Any, train: pd.DataFrame, valid: pd.DataFrame) -> Any:
    model.fit(data_loader.coerce_features(train), train[schema.TARGET])
    return model.predict_proba(data_loader.coerce_features(valid))[:, 1]


def seed_stability(rows: pd.DataFrame) -> pd.DataFrame:
    """Refit with several seeds on the windows of the protocol."""
    results = []
    dates = rows[schema.ARRIVAL_DATE]
    makers: dict[str, Callable[[int], Any]] = {"random_forest": _random_forest, "xgboost": _xgboost}
    for name, make in makers.items():
        for seed in harness.SEEDS:
            for window in harness.Q4:
                train = rows[dates < window.train_end]
                valid = rows[(dates >= window.start) & (dates < window.end)]
                scores = harness.score_window(valid, _fit_predict(make(seed), train, valid))
                results.append({"model": name, "seed": seed, "window": window.name, **scores})
            print(f"seed | {name} | {seed}", flush=True)
    return pd.DataFrame(results)


def learning_curve(rows: pd.DataFrame) -> pd.DataFrame:
    """Train on growing random shares of whole batches and score fixed validation rows."""
    results = []
    dates = rows[schema.ARRIVAL_DATE]
    rng = np.random.default_rng(common.SEED)
    for window in harness.Q4[1:]:
        train = rows[dates < window.train_end]
        valid = rows[(dates >= window.start) & (dates < window.end)]
        batches = train["batch_id"].unique()
        for fraction in _FRACTIONS:
            for repetition in range(1 if fraction == 1.0 else 3):
                kept = rng.choice(batches, int(fraction * batches.size), replace=False)
                subset = train[train["batch_id"].isin(kept)]
                pipeline = features.build_pipeline(
                    features.build_preprocessor(features.Family.UNSCALED),
                    common.reference_xgboost(),
                )
                scores = harness.score_window(valid, _fit_predict(pipeline, subset, valid))
                results.append(
                    {
                        "window": window.name,
                        "fraction": fraction,
                        "repetition": repetition,
                        "train_rows": len(subset),
                        **scores,
                    }
                )
        print(f"curve | {window.name}", flush=True)
    return pd.DataFrame(results)


def perturbation(rows: pd.DataFrame) -> pd.DataFrame:
    """Disturb validation inputs slightly: a fragile model would lose much more than the noise."""
    results = []
    dates = rows[schema.ARRIVAL_DATE]
    rng = np.random.default_rng(common.SEED)
    for window in harness.Q4:
        train = rows[dates < window.train_end]
        valid = rows[(dates >= window.start) & (dates < window.end)]
        pipeline = _random_forest(common.SEED).fit(
            data_loader.coerce_features(train), train[schema.TARGET]
        )
        disturbed = {
            "none": valid,
            "adr_times_noise_10_percent": valid.assign(
                adr=valid["adr"] * rng.normal(1.0, 0.1, len(valid)).clip(0.5, 1.5)
            ),
            "lead_time_plus_minus_7_days": valid.assign(
                lead_time=(valid["lead_time"] + rng.integers(-7, 8, len(valid))).clip(lower=0)
            ),
            "agent_unknown_for_all": valid.assign(agent="unseen"),
            "previous_cancellations_zero": valid.assign(previous_cancellations=0),
        }
        for name, variant in disturbed.items():
            scores = pipeline.predict_proba(data_loader.coerce_features(variant))[:, 1]
            results.append(
                {"perturbation": name, "window": window.name, **harness.score_window(valid, scores)}
            )
    return pd.DataFrame(results)


def _boosting(seed: int = common.SEED) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=30, random_state=seed
    )


def ceiling(rows: pd.DataFrame) -> pd.DataFrame:
    """Estimate what these 19 variables could reach if the future were not a different period.

    Inside each window, a grouped cross-validation with whole batches adds rows of the same quarter
    to training. It uses information that is not available in production, so it is an upper bound,
    and its distance to the temporal figure is the cost of drift. The ensemble of the two tuned
    models is the other reference: if averaging adds nothing, the models have converged.
    """
    results = []
    dates = rows[schema.ARRIVAL_DATE]
    for window in harness.Q4:
        train = rows[dates < window.train_end]
        valid = rows[(dates >= window.start) & (dates < window.end)]
        forest = _fit_predict(_random_forest(common.SEED), train, valid)
        boost = _fit_predict(_xgboost(common.SEED), train, valid)
        within = np.zeros(len(valid))
        splitter = StratifiedGroupKFold(5, shuffle=True, random_state=common.SEED)
        for fit_index, score_index in splitter.split(
            valid, valid[schema.TARGET], valid["batch_id"]
        ):
            extended = pd.concat([train, valid.iloc[fit_index]])
            pipeline = features.build_pipeline(
                features.build_preprocessor(features.Family.UNSCALED), common.reference_xgboost()
            )
            within[score_index] = _fit_predict(pipeline, extended, valid.iloc[score_index])
        actual = valid[schema.TARGET].to_numpy()
        no_deposit = (valid["deposit_type"] == "No Deposit").to_numpy()
        for name, scores in (
            ("random_forest_temporal", forest),
            ("xgboost_temporal", boost),
            ("ensemble_temporal", (forest + boost) / 2),
            ("xgboost_same_quarter_grouped_upper_bound", within),
        ):
            results.append(
                {
                    "variant": name,
                    "window": window.name,
                    "roc_auc": float(roc_auc_score(actual, scores)),
                    "roc_auc_no_deposit": float(
                        roc_auc_score(actual[no_deposit], scores[no_deposit])
                    ),
                }
            )
        print(f"ceiling | {window.name}", flush=True)
    return pd.DataFrame(results)


def main() -> None:
    """Write the robustness tables."""
    rows = harness.load_rows()
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 100)
    seeds = seed_stability(rows)
    harness.save("robustness_seeds", seeds)
    per_seed = seeds.groupby(["model", "seed"])[["roc_auc", "roc_auc_no_deposit"]].mean()
    print(per_seed.round(4))
    print(per_seed.groupby("model").agg(["mean", "std"]).round(5))
    curve = learning_curve(rows)
    harness.save("robustness_learning_curve", curve)
    print(
        curve.groupby(["window", "fraction"])[["train_rows", "roc_auc", "roc_auc_no_deposit"]]
        .mean()
        .round(4)
    )
    disturbed = perturbation(rows)
    harness.save("robustness_perturbation", disturbed)
    print(disturbed.groupby("perturbation")[["roc_auc", "roc_auc_no_deposit"]].mean().round(4))
    limits = ceiling(rows)
    harness.save("robustness_ceiling", limits)
    print(limits.groupby("variant")[["roc_auc", "roc_auc_no_deposit"]].mean().round(4))
    print(limits.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
