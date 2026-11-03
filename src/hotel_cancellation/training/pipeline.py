"""Training workflow: search, final training, packaging and the registry of the models.

Training never reads the test rows: `data_loader.load_development` drops them as soon as the
contract has checked the file, and `ModelTrainer` refuses them if they arrive some other way.
"""

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from hotel_cancellation import champion, config, models
from hotel_cancellation.data import data_loader
from hotel_cancellation.evaluation import evaluator
from hotel_cancellation.training import model_trainer
from hotel_cancellation.training.tracking import NullTracker, Tracker

logger = logging.getLogger(__name__)

_COMPRESSION_LEVEL = 3


@dataclass(frozen=True)
class TrainingReport:
    """What `train` produced."""

    selection: model_trainer.Selection
    candidates: dict[str, model_trainer.Candidate]
    manifest: dict[str, Any]


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
    force: bool = False,
) -> dict[str, model_trainer.SearchOutcome]:
    """Search each model and record the results after each one, so a failure loses little.

    Args:
        development: Development rows.
        settings: Where to write.
        profile: Models and budget to use.
        seed: Seed of the search and the models.
        tracker: Where to track the runs.
        only: Models to search. Previous results of the others are kept.
        force: Search although the test already ran, overwriting the versioned results.

    Returns:
        The result of each searched model.

    Raises:
        champion.TestAlreadyEvaluatedError: If the test already ran and `force` is false.
    """
    _assert_champion_can_change(settings, force=force)
    tracker = tracker or NullTracker()
    trainer = _trainer(development, profile, seed)
    specs = models.specs_by_name(profile)
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
    force: bool = False,
) -> TrainingReport:
    """Select the champion, train every candidate on the final windows and package the champion.

    Args:
        development: Development rows.
        settings: Where to read the search from and where to write.
        profile: Must match the profile of the search.
        seed: Seed of the models.
        tracker: Where to track the runs.
        force: Replace the champion although the test already ran.

    Returns:
        The selection, the candidates and the manifest of the champion.

    Raises:
        champion.TestAlreadyEvaluatedError: If the test already ran and `force` is false.
    """
    _assert_champion_can_change(settings, force=force)
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
    _fit_and_save_scenario(development, settings, manifest)
    _write_registry(settings, manifest, keep_candidates=False)

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


def _assert_champion_can_change(settings: config.Settings, *, force: bool) -> None:
    """Refuse to change what was evaluated, looking also at the record in its default place."""
    if force:
        return
    records = [settings.final_evaluation_record]
    if config.same_path(settings.models_dir, config.DEFAULT_MODELS_DIR):
        default = settings.model_copy(update={"outputs_dir": config.DEFAULT_OUTPUTS_DIR})
        records.append(default.final_evaluation_record)
    if any(record.exists() for record in records):
        raise champion.TestAlreadyEvaluatedError(
            "The test already ran with the current champion. Replace it with --force, "
            "and declare it in the report."
        )


def _artifact_entry(path: Path, settings: config.Settings) -> dict[str, Any]:
    return {
        "file": path.relative_to(settings.models_dir).as_posix(),
        "sha256": champion.sha256_of(path),
        "size_bytes": path.stat().st_size,
    }


def _write_registry(
    settings: config.Settings, manifest: Mapping[str, Any], *, keep_candidates: bool
) -> None:
    """Record the hash of every model that git does not keep.

    Args:
        settings: Where the models are.
        manifest: Manifest of the champion.
        keep_candidates: Trust the registered candidates instead of hashing them again.

    Raises:
        champion.EvaluationNotReadyError: If the candidate that won is not the saved champion.
    """
    path = settings.candidates_registry_path
    candidates = None
    if keep_candidates and path.exists():
        candidates = json.loads(path.read_text("utf-8"))["candidates"]
    if candidates is None:
        candidates = {
            model.stem: _artifact_entry(model, settings)
            for model in sorted(settings.candidates_dir.glob("*.joblib"))
        }
    name, digest = manifest["model"]["name"], manifest["model"]["sha256"]
    if candidates.get(name, {}).get("sha256") != digest:
        raise champion.EvaluationNotReadyError(
            f"Candidate {name} is not the champion: hashes differ."
        )
    champion.write_json(
        path,
        {
            "champion": {"name": name, "sha256": digest},
            "candidates": candidates,
            "scenario": _artifact_entry(settings.scenario_model_path, settings),
        },
    )


def _fit_and_save_scenario(
    development: pd.DataFrame, settings: config.Settings, manifest: Mapping[str, Any]
) -> None:
    profile = manifest["profile"]
    spec = models.specs_by_name(profile)[manifest["model"]["name"]]
    scenario = _trainer(development, profile, manifest["seed"]).fit_special_requests_scenario(
        spec, manifest["model"]["params"], manifest["model"]["calibration"]
    )
    settings.scenario_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scenario, settings.scenario_model_path, compress=_COMPRESSION_LEVEL)


def train_scenario(
    development: pd.DataFrame, *, settings: config.Settings, force: bool = False
) -> None:
    """Train the informative scenario of the saved champion and register its hash.

    The champion and the registered candidates stay as they are.

    Args:
        development: Development rows.
        settings: Where the models are.
        force: Train it although the test already ran.

    Raises:
        champion.TestAlreadyEvaluatedError: If the test already ran and `force` is false.
    """
    _assert_champion_can_change(settings, force=force)
    manifest = champion.load_champion(settings.champion_dir).manifest
    _fit_and_save_scenario(development, settings, manifest)
    _write_registry(settings, manifest, keep_candidates=True)


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
    force: bool = False,
) -> TrainingReport:
    """Search and train in one go. It never reads the test: it only receives development rows."""
    tune(development, settings=settings, profile=profile, seed=seed, tracker=tracker, force=force)
    return train(
        development, settings=settings, profile=profile, seed=seed, tracker=tracker, force=force
    )
