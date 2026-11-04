"""Measure the optimism of tuning and selecting on the same windows.

The published search picks the configuration with the best mean over the four windows, and reports
that same mean. Here each configuration is scored per window. A nested estimate then picks the
configuration only with earlier windows and scores it on the next one, which is an honest estimate
of the procedure.

Run it with `uv run python -m experiments.review.nested_search`.
"""

import json
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler

from experiments import common
from experiments.review import harness
from hotel_cancellation import config, models
from hotel_cancellation.data import data_loader, schema

_MODELS = ("random_forest", "xgboost")
_DRAWS = 20


def _score_one(
    name: str,
    params: dict[str, Any],
    features: pd.DataFrame,
    target: pd.Series,
    train: Any,
    valid: Any,
) -> float:
    spec = next(spec for spec in models.model_catalog("full") if spec.name == name)
    pipeline = spec.pipeline(common.SEED).set_params(**params)
    pipeline.fit(features.iloc[train], target.iloc[train])
    return float(
        roc_auc_score(target.iloc[valid], pipeline.predict_proba(features.iloc[valid])[:, 1])
    )


def score_grid(name: str, rows: pd.DataFrame) -> pd.DataFrame:
    """Score the published search's configurations on each window, with its sampler and seed."""
    spec = next(spec for spec in models.model_catalog("full") if spec.name == name)
    draws = list(ParameterSampler(dict(spec.space), _DRAWS, random_state=common.SEED))
    features = data_loader.coerce_features(rows)
    target = rows[schema.TARGET]
    folds = data_loader.expanding_windows(rows)
    tasks = [
        (position, window, params, train, valid)
        for position, params in enumerate(draws)
        for window, (train, valid) in enumerate(folds)
    ]
    scores = joblib.Parallel(n_jobs=-1, verbose=1)(
        joblib.delayed(_score_one)(name, params, features, target, train, valid)
        for _, _, params, train, valid in tasks
    )
    return pd.DataFrame(
        {
            "model": name,
            "configuration": [task[0] for task in tasks],
            "params": [json.dumps(task[2], sort_keys=True) for task in tasks],
            "window": [config.EXPANDING_WINDOWS[task[1]].name for task in tasks],
            "roc_auc": scores,
        }
    )


def nested_estimate(grid: pd.DataFrame) -> pd.DataFrame:
    """For each window, pick the best configuration on earlier windows and score it on this one."""
    table = grid.pivot_table(index="configuration", columns="window", values="roc_auc")
    windows = list(table.columns)
    overall_best: Any = table.mean(axis=1).idxmax()
    results = []
    for position in range(1, len(windows)):
        chosen: Any = table[windows[:position]].mean(axis=1).idxmax()
        current = windows[position]
        results.append(
            {
                "window": current,
                "chosen_configuration": int(chosen),
                "nested_roc_auc": float(table.loc[chosen, current]),
                "best_possible_in_this_window": float(table[current].max()),
                "median_configuration": float(table[current].median()),
                "non_nested_best_overall": float(table.loc[overall_best, current]),
            }
        )
    return pd.DataFrame(results)


def main() -> None:
    """Score the grids and write both estimates."""
    rows = harness.load_rows()
    pd.set_option("display.width", 220, "display.max_columns", 30)
    grids, nested = [], []
    for name in _MODELS:
        grid = score_grid(name, rows)
        grids.append(grid)
        estimate = nested_estimate(grid).assign(model=name)
        nested.append(estimate)
        means = grid.groupby("configuration")["roc_auc"].mean()
        print(f"{name}: from {means.min():.4f} to {means.max():.4f}, std {means.std():.4f}")
        print(estimate.round(4).to_string(index=False), flush=True)
        print(
            "gap, nested minus non-nested:",
            (estimate["nested_roc_auc"] - estimate["non_nested_best_overall"]).mean(),
        )
    harness.save("nested_grid", pd.concat(grids))
    harness.save("nested_estimate", pd.concat(nested))


if __name__ == "__main__":
    main()
