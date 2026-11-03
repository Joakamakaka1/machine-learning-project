"""Reproduce the published baseline without touching what is frozen.

It checks the hashes, recomputes the test figures from the stored predictions, so that 2017 is not
scored again, and trains the champion's configuration again on the four windows of 2016. It also
trains the served model again in memory and compares its probabilities on development rows.

Run it with `uv run python -m experiments.review.baseline`.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from experiments import common
from experiments.review import harness
from hotel_cancellation import config, evaluator, model_trainer, models
from hotel_cancellation.data import data_loader, schema

_TOLERANCE = 1e-5


def stored_test_figures(folder: Path) -> pd.DataFrame:
    """Recompute the champion's figures from stored predictions and compare them with the table."""
    predictions = pd.read_csv(folder / "predictions.csv.gz")
    published = pd.read_csv(folder / "metrics.csv")
    rows = []
    for model in ("champion", "random_forest", "xgboost", "logistic_regression", "decision_tree"):
        name = "random_forest" if model == "champion" else model
        reference = published[
            (published["model"] == name)
            & (published["subset"] == "all")
            & (published["threshold_name"] == "decision")
        ].iloc[0]
        actual, scores = predictions[schema.TARGET], predictions[model]
        recomputed = {
            "roc_auc": roc_auc_score(actual, scores),
            "pr_auc": average_precision_score(actual, scores),
            "brier": brier_score_loss(actual, scores),
        }
        for metric, value in recomputed.items():
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "published": float(reference[metric]),
                    "recomputed": float(value),
                    "matches": abs(float(reference[metric]) - float(value)) < _TOLERANCE,
                }
            )
    return pd.DataFrame(rows)


def validation_windows(development: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Train the champion's configuration on each window and compare it with the stored table."""
    spec = next(spec for spec in models.model_catalog("full") if spec.name == "random_forest")
    published = pd.read_csv(config.get_settings().validation_dir / "metrics_by_window.csv")
    published = published[published["model"] == "random_forest"].set_index("window")
    features = data_loader.coerce_features(development)
    rows = []
    for position, (train, valid) in enumerate(data_loader.expanding_windows(development)):
        window = config.EXPANDING_WINDOWS[position].name
        pipeline = spec.pipeline(common.SEED).set_params(**params)
        pipeline.set_params(model__n_jobs=-1)
        pipeline.fit(features.iloc[train], development[schema.TARGET].iloc[train])
        scores = pipeline.predict_proba(features.iloc[valid])[:, 1]
        metrics = evaluator.validation_metrics(
            development[schema.TARGET].iloc[valid], scores, features.iloc[valid]
        )
        rows.append(
            {
                "window": window,
                "published_roc_auc": published.loc[window, "roc_auc"],
                "reproduced_roc_auc": metrics["roc_auc"],
                "published_unique": published.loc[window, "roc_auc_unique"],
                "reproduced_unique": metrics["roc_auc_unique"],
                "published_no_deposit": published.loc[window, "roc_auc_no_deposit"],
                "reproduced_no_deposit": metrics["roc_auc_no_deposit"],
            }
        )
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def served_model(development: pd.DataFrame, manifest: dict[str, Any]) -> dict[str, Any]:
    """Train the served model again and compare it with the frozen one on development rows."""
    trainer = model_trainer.ModelTrainer(
        development, catalog=models.model_catalog("full"), n_jobs=1
    )
    spec = next(spec for spec in trainer.catalog if spec.name == manifest["model"]["name"])
    rebuilt = trainer.fit_served(
        spec, manifest["model"]["params"], manifest["model"]["calibration"]
    )
    frozen = joblib.load(config.get_settings().champion_dir / manifest["model"]["file"])
    rows = data_loader.coerce_features(development)
    first, second = rebuilt.predict_proba(rows)[:, 1], frozen.predict_proba(rows)[:, 1]
    return {
        "rows_compared": len(rows),
        "max_abs_difference": float(np.abs(first - second).max()),
        "identical": bool(np.array_equal(first, second)),
        "calibrators_in_frozen_model": len(frozen.calibrated_classifiers_),
        "isotonic_steps": len(frozen.calibrated_classifiers_[0].calibrators[0].X_thresholds_),
    }


def main() -> None:
    """Write the reproduction of the baseline to `outputs/review/`."""
    settings = config.get_settings()
    harness.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((settings.champion_dir / "manifest.json").read_text("utf-8"))
    model_path = settings.champion_dir / manifest["model"]["file"]
    actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    development = common.load_development()

    test_figures = stored_test_figures(settings.final_evaluation_dir)
    test_figures.to_csv(
        harness.OUTPUT_DIR / "baseline_test_figures.csv", index=False, lineterminator="\n"
    )
    windows = validation_windows(development, manifest["model"]["params"])
    windows.round(6).to_csv(
        harness.OUTPUT_DIR / "baseline_windows.csv", index=False, lineterminator="\n"
    )
    served = served_model(development, manifest)

    report = {
        "champion_hash_matches_manifest": actual_hash == manifest["model"]["sha256"],
        "development_rows": len(development),
        "stored_figures_match": bool(test_figures["matches"].all()),
        "windows_max_abs_gap": float(
            (windows["published_roc_auc"] - windows["reproduced_roc_auc"]).abs().max()
        ),
        "served_model": served,
    }
    (harness.OUTPUT_DIR / "baseline.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(test_figures.round(6).to_string(index=False))
    print(windows.round(4).to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
