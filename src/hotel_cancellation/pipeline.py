"""Training workflow: search, select, train the candidates and package the champion.

Nothing here reads the test rows: `load_development` drops them as soon as the contract has
checked the file, and `ModelTrainer` refuses them if they arrive some other way.
"""

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from hotel_cancellation import champion, config, data_loader, evaluator, model_trainer, models
from hotel_cancellation.tracking import NullTracker, Tracker

logger = logging.getLogger(__name__)

_COMPRESSION_LEVEL = 3


@dataclass(frozen=True)
class TrainingReport:
    """What `train` produced."""

    selection: model_trainer.Selection
    candidates: dict[str, model_trainer.Candidate]
    manifest: dict[str, Any]


def load_development(
    path: Path | None = None, *, expected_sha256: str = config.RAW_DATASET_SHA256
) -> pd.DataFrame:
    """Load the file, check its contract and keep the development rows only.

    Args:
        path: Compressed CSV. Defaults to the location in the settings.
        expected_sha256: Hash of the uncompressed CSV.

    Returns:
        Clean rows with arrivals before the test period, in arrival order.
    """
    raw = data_loader.load_raw(path, expected_sha256=expected_sha256)
    data_loader.validate_raw(raw)
    return data_loader.split_temporal(data_loader.clean(raw)).development


def _trainer(
    development: pd.DataFrame, profile: models.ProfileName, seed: int
) -> model_trainer.ModelTrainer:
    return model_trainer.ModelTrainer(
        development,
        catalog=models.model_catalog(profile),
        seed=seed,
        # Parallel searches cost more than they save on the tiny frames of the fast profile.
        n_jobs=-1 if profile == "full" else 1,
    )


def _specs(profile: models.ProfileName) -> dict[str, models.ModelSpec]:
    return {spec.name: spec for spec in (models.BASELINE, *models.model_catalog(profile))}


def _model_record(outcome: model_trainer.SearchOutcome) -> dict[str, Any]:
    scores = outcome.window_scores
    return {
        "params": outcome.params,
        "configurations_tried": len(outcome.all_configurations),
        "windows": scores.to_dict(orient="index"),
        "mean": scores.mean().to_dict(),
        # Sample dispersion between windows, the figure the protocol asks to publish.
        "std": scores.std(ddof=1).to_dict(),
    }


def _read_best_params(path: Path) -> dict[str, Any]:
    content: dict[str, Any] = json.loads(path.read_text("utf-8"))
    return content


def tune(
    development: pd.DataFrame,
    *,
    settings: config.Settings,
    profile: models.ProfileName = "full",
    seed: int = model_trainer.DEFAULT_SEED,
    tracker: Tracker | None = None,
    only: Sequence[str] | None = None,
) -> dict[str, model_trainer.SearchOutcome]:
    """Search each model and record the results after each one, so a failure loses little.

    Args:
        development: Development rows.
        settings: Where to write.
        profile: Models and budget to use.
        seed: Seed of the search and the models.
        tracker: Where to track the runs.
        only: Models to search. Previous results of the others are kept.

    Returns:
        The result of each searched model.
    """
    tracker = tracker or NullTracker()
    trainer = _trainer(development, profile, seed)
    specs = _specs(profile)
    names = list(specs) if only is None else list(only)

    path = settings.best_params_path
    records: dict[str, Any] = {}
    if only is not None and path.exists():
        records = _read_best_params(path).get("models", {})
    content = {
        "profile": profile,
        "seed": seed,
        "code": champion.code_state(),
        "dataset_sha256": config.RAW_DATASET_SHA256,
        "development_rows": len(development),
        "models": records,
    }

    settings.validation_dir.mkdir(parents=True, exist_ok=True)
    outcomes = {}
    for name in names:
        logger.info("Searching %s", name)
        outcome = trainer.search(specs[name])
        outcomes[name] = outcome
        records[name] = _model_record(outcome)
        champion.write_json(path, content)
        outcome.all_configurations.to_csv(
            settings.validation_dir / f"search_{name}.csv", index=False, lineterminator="\n"
        )
        logger.info("%s: mean ROC-AUC %.4f", name, outcome.window_scores["roc_auc"].mean())
    _write_validation_tables(records, settings.validation_dir)
    tracker.log_search(outcomes, profile=profile)
    return outcomes


def _write_validation_tables(records: Mapping[str, Any], folder: Path) -> None:
    by_window = pd.concat(
        {name: pd.DataFrame.from_dict(r["windows"], orient="index") for name, r in records.items()},
        names=["model", "window"],
    ).reset_index()
    by_window.to_csv(folder / "metrics_by_window.csv", index=False, lineterminator="\n")

    summary = pd.DataFrame(
        [
            {
                "model": name,
                **{f"{metric}_mean": value for metric, value in record["mean"].items()},
                **{f"{metric}_std": value for metric, value in record["std"].items()},
            }
            for name, record in records.items()
        ]
    )
    summary.to_csv(folder / "summary.csv", index=False, lineterminator="\n")


def _load_outcomes(
    settings: config.Settings, profile: models.ProfileName
) -> tuple[dict[str, Any], dict[str, model_trainer.SearchOutcome]]:
    path = settings.best_params_path
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist: run `hotel-ml tune` first.")
    content = _read_best_params(path)
    if content["profile"] != profile:
        raise ValueError(f"{path} comes from profile {content['profile']}, not from {profile}.")
    outcomes = {
        name: model_trainer.SearchOutcome(
            name=name,
            params=record["params"],
            window_scores=pd.DataFrame.from_dict(record["windows"], orient="index"),
            all_configurations=pd.DataFrame(),
        )
        for name, record in content["models"].items()
    }
    return content, outcomes


def train(
    development: pd.DataFrame,
    *,
    settings: config.Settings,
    profile: models.ProfileName = "full",
    seed: int = model_trainer.DEFAULT_SEED,
    tracker: Tracker | None = None,
) -> TrainingReport:
    """Select the champion, train every candidate on the final windows and package the champion.

    Args:
        development: Development rows.
        settings: Where to read the search from and where to write.
        profile: Must match the profile of the search.
        seed: Seed of the models.
        tracker: Where to track the runs.

    Returns:
        The selection, the candidates and the manifest of the champion.
    """
    tracker = tracker or NullTracker()
    content, outcomes = _load_outcomes(settings, profile)
    selection = model_trainer.select_by_one_standard_error(
        {
            name: outcome.window_scores["roc_auc"].tolist()
            for name, outcome in outcomes.items()
            if name != models.BASELINE.name
        }
    )
    logger.info("Best model %s, chosen %s", selection.best, selection.champion)

    candidates = _trainer(development, profile, seed).train_candidates(outcomes)
    settings.candidates_dir.mkdir(parents=True, exist_ok=True)
    for name, candidate in candidates.items():
        joblib.dump(
            candidate.model, settings.candidates_dir / f"{name}.joblib", compress=_COMPRESSION_LEVEL
        )

    chosen = candidates[selection.champion]
    manifest = champion.save_champion(
        chosen.model,
        settings.champion_dir,
        model_name=chosen.name,
        params=chosen.params,
        calibration_method=chosen.calibration.method,
        calibration_briers=chosen.calibration.briers,
        threshold_contrast=chosen.calibration.threshold_contrast,
        validation=content["models"][chosen.name],
        selection=selection.table.to_dict(orient="records"),
        seed=seed,
        profile=profile,
        development_rows=len(development),
    )

    settings.validation_dir.mkdir(parents=True, exist_ok=True)
    selection.table.to_csv(
        settings.validation_dir / "selection.csv", index=False, lineterminator="\n"
    )
    champion.write_json(
        settings.validation_dir / "calibration.json",
        {
            name: {
                "method": candidate.calibration.method,
                "briers_on_2016q4": candidate.calibration.briers,
                "metrics_on_2016q4": candidate.calibration.metrics,
                "threshold_contrast_on_2016q4": candidate.calibration.threshold_contrast,
            }
            for name, candidate in candidates.items()
        },
    )
    tracker.log_training(
        selection,
        candidates,
        champion_dir=settings.champion_dir,
        manifest=manifest,
        sample=data_loader.coerce_features(development.head(100)),
    )
    _log_validation(selection, manifest)
    return TrainingReport(selection=selection, candidates=candidates, manifest=manifest)


def _log_validation(selection: model_trainer.Selection, manifest: Mapping[str, Any]) -> None:
    logger.info("Selection:\n%s", selection.table.round(4).to_string(index=False))
    scenario = evaluator.DEFAULT_COST_SCENARIO
    logger.info(
        "Champion: %s, calibration %s, threshold %.3f at costs %g:%g",
        manifest["model"]["name"],
        manifest["model"]["calibration"],
        manifest["decision"]["threshold"],
        scenario.false_positive,
        scenario.false_negative,
    )


def run_all(
    development: pd.DataFrame,
    *,
    settings: config.Settings,
    profile: models.ProfileName = "full",
    seed: int = model_trainer.DEFAULT_SEED,
    tracker: Tracker | None = None,
) -> TrainingReport:
    """Search and train in one go. It never reads the test: it only receives development rows."""
    tune(development, settings=settings, profile=profile, seed=seed, tracker=tracker)
    return train(development, settings=settings, profile=profile, seed=seed, tracker=tracker)
