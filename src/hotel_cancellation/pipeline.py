"""Training workflow and the single execution on the test.

Training never reads the test rows: `load_development` drops them as soon as the contract has
checked the file, and `ModelTrainer` refuses them if they arrive some other way. Only `load_test`
and `load_cleaned` return them: `hotel-ml evaluate` after `prepare_evaluation`, and
`hotel-ml explain` and `hotel-ml audit` after the single execution.
"""

import json
import logging
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from hotel_cancellation import (
    champion,
    config,
    data_loader,
    evaluator,
    model_trainer,
    models,
    schema,
)
from hotel_cancellation.tracking import NullTracker, Tracker

logger = logging.getLogger(__name__)

_COMPRESSION_LEVEL = 3
_SEGMENTS = ("hotel", "market_segment", "customer_type", "distribution_channel", "deposit_type")


SCENARIO_NAME = "special_requests_scenario"


class TestAlreadyEvaluatedError(RuntimeError):
    """The test was already executed, and it is executed only once."""

    __test__ = False


class EvaluationNotReadyError(RuntimeError):
    """The models or the environment are not the ones the single execution demands."""


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


def load_test(
    path: Path | None = None, *, expected_sha256: str = config.RAW_DATASET_SHA256
) -> pd.DataFrame:
    """Load the rows of the test period.

    Only `hotel-ml evaluate` may call this, or `hotel-ml explain` once the test has been executed.

    Args:
        path: Compressed CSV. Defaults to the location in the settings.
        expected_sha256: Hash of the uncompressed CSV.

    Returns:
        Clean rows with arrivals in the test period, in arrival order.
    """
    raw = data_loader.load_raw(path, expected_sha256=expected_sha256)
    data_loader.validate_raw(raw)
    return data_loader.split_temporal(data_loader.clean(raw)).test


def load_cleaned(
    path: Path | None = None, *, expected_sha256: str = config.RAW_DATASET_SHA256
) -> pd.DataFrame:
    """Load every clean row. Only `hotel-ml audit` may call this, after the single execution.

    Args:
        path: Compressed CSV. Defaults to the location in the settings.
        expected_sha256: Hash of the uncompressed CSV.

    Returns:
        Clean rows of the whole period.
    """
    raw = data_loader.load_raw(path, expected_sha256=expected_sha256)
    data_loader.validate_raw(raw)
    return data_loader.clean(raw)


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
        TestAlreadyEvaluatedError: If the test already ran and `force` is false.
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


def _same_path(first: Path, second: Path) -> bool:
    return first.resolve() == second.resolve()


def _assert_champion_can_change(settings: config.Settings, *, force: bool) -> None:
    """Refuse to change what was evaluated, looking also at the record in its default place."""
    if force:
        return
    records = [settings.final_evaluation_record]
    if _same_path(settings.models_dir, config.DEFAULT_MODELS_DIR):
        default = settings.model_copy(update={"outputs_dir": config.DEFAULT_OUTPUTS_DIR})
        records.append(default.final_evaluation_record)
    if any(record.exists() for record in records):
        raise TestAlreadyEvaluatedError(
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
        EvaluationNotReadyError: If the candidate that won is not the saved champion.
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
        raise EvaluationNotReadyError(f"Candidate {name} is not the champion: hashes differ.")
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
    spec = _specs(profile)[manifest["model"]["name"]]
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
        TestAlreadyEvaluatedError: If the test already ran and `force` is false.
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


def _subset_rows(rows: pd.DataFrame, target: pd.Series) -> dict[str, pd.Series]:
    with_target = rows.assign(__target__=target.to_numpy())
    return {
        "all": pd.Series(data=True, index=rows.index),
        "unique": ~with_target.duplicated(),
        "no_deposit": rows["deposit_type"] == "No Deposit",
        "non_refund": rows["deposit_type"] == "Non Refund",
    }


def _measure(
    target: pd.Series, probabilities: pd.Series, rows: pd.DataFrame, threshold: float
) -> dict[str, float]:
    """Measure a subset at one threshold: ranking, probabilities, decisions and their costs."""
    predicted = probabilities >= threshold
    actual = target == 1
    measures = evaluator.validation_metrics(target, probabilities, rows)
    return {
        "rows": len(target),
        "prevalence": float(target.mean()) if len(target) else float("nan"),
        "threshold": threshold,
        **{metric: measures[metric] for metric in ("roc_auc", "pr_auc", "brier")},
        **evaluator.calibration_diagnostics(target, probabilities),
        **evaluator.classification_metrics(target, probabilities, threshold),
        **evaluator.costs_at_own_threshold(target, probabilities),
        **evaluator.costs_at_threshold(target, probabilities, threshold),
        "tp": int((predicted & actual).sum()),
        "fp": int((predicted & ~actual).sum()),
        "fn": int((~predicted & actual).sum()),
        "tn": int((~predicted & ~actual).sum()),
    }


def _metrics_table(
    target: pd.Series, predictions: pd.DataFrame, rows: pd.DataFrame, decision: float
) -> pd.DataFrame:
    subsets = _subset_rows(rows, target)
    records = []
    for model in predictions.columns:
        for threshold_name, threshold in (("decision", decision), ("fixed_0.5", 0.5)):
            for subset, mask in subsets.items():
                measured = _measure(
                    target[mask], predictions.loc[mask, model], rows[mask], threshold
                )
                records.append(
                    {
                        "model": model,
                        "threshold_name": threshold_name,
                        "subset": subset,
                        **measured,
                    }
                )
    return pd.DataFrame(records)


def _segments_table(
    target: pd.Series, probabilities: pd.Series, rows: pd.DataFrame, decision: float
) -> pd.DataFrame:
    records = []
    for segment in _SEGMENTS:
        for value, group in rows.groupby(segment, dropna=False):
            measured = _measure(target[group.index], probabilities[group.index], group, decision)
            records.append({"segment": segment, "value": value, **measured})
    return pd.DataFrame(records)


def _monthly_table(
    target: pd.Series, probabilities: pd.Series, dates: pd.Series, rows: pd.DataFrame
) -> pd.DataFrame:
    records = []
    for month, indices in dates.groupby(dates.dt.strftime("%Y-%m")).groups.items():
        measures = evaluator.validation_metrics(
            target[indices], probabilities[indices], rows.loc[indices]
        )
        records.append(
            {
                "month": month,
                "rows": len(indices),
                "prevalence": float(target[indices].mean()),
                "roc_auc": measures["roc_auc"],
                "brier": measures["brier"],
            }
        )
    return pd.DataFrame(records)


def _previous_executions(settings: config.Settings) -> list[dict[str, Any]]:
    path = settings.final_evaluation_record
    if not path.exists():
        return []
    executions: list[dict[str, Any]] = json.loads(path.read_text("utf-8"))["executions"]
    return executions


def _assert_real_execution(settings: config.Settings, manifest: Mapping[str, Any]) -> None:
    """Demand what makes this execution the official one, so nothing can escape the record."""
    if manifest["profile"] != "full":
        raise EvaluationNotReadyError(f"Profile must be full, not {manifest['profile']}.")
    if not champion.tree_is_clean():
        raise EvaluationNotReadyError("The git tree must be clean: commit everything first.")
    if not _same_path(settings.outputs_dir, config.DEFAULT_OUTPUTS_DIR):
        raise EvaluationNotReadyError("Unset HOTEL_ML_OUTPUTS_DIR: the record has one place.")
    if not _same_path(settings.models_dir, config.DEFAULT_MODELS_DIR):
        raise EvaluationNotReadyError("Unset HOTEL_ML_MODELS_DIR: the champion has one place.")


def _verify_entry(name: str, entry: Mapping[str, Any], settings: config.Settings) -> None:
    path = settings.models_dir / entry["file"]
    if not path.exists():
        raise EvaluationNotReadyError(f"{name}: {path} does not exist. Run `hotel-ml train`.")
    if champion.sha256_of(path) != entry["sha256"]:
        raise EvaluationNotReadyError(f"{name}: {path} does not match its registered hash.")


def _verify_registry(settings: config.Settings, manifest: Mapping[str, Any]) -> None:
    path = settings.candidates_registry_path
    if not path.exists():
        raise EvaluationNotReadyError(f"{path} does not exist. Run `hotel-ml train-scenario`.")
    registry = json.loads(path.read_text("utf-8"))
    candidates = registry["candidates"]
    missing = sorted(set(_specs(manifest["profile"])).difference(candidates))
    if missing:
        raise EvaluationNotReadyError(f"Models missing from {path}: {', '.join(missing)}.")
    for name, entry in candidates.items():
        _verify_entry(name, entry, settings)
    _verify_entry(SCENARIO_NAME, registry["scenario"], settings)
    name, digest = manifest["model"]["name"], manifest["model"]["sha256"]
    if candidates[name]["sha256"] != digest:
        raise EvaluationNotReadyError(f"Candidate {name} is not the champion: hashes differ.")


def prepare_evaluation(
    settings: config.Settings, *, force: bool = False, strict: bool = True
) -> None:
    """Check everything the single execution needs, before reading a row of the test.

    Args:
        settings: Where the models and the record are.
        force: Allow a repetition.
        strict: Demand the conditions of the official execution.

    Raises:
        TestAlreadyEvaluatedError: If it already ran and `force` is false.
        EvaluationNotReadyError: If a model or the environment is not the expected one.
        ChampionIntegrityError: If the champion is not what its manifest says.
    """
    executions = _previous_executions(settings)
    if executions and not force:
        raise TestAlreadyEvaluatedError(
            f"The test already ran on {executions[-1]['executed_at']}. "
            "Repeat it with --force, and declare it in the report."
        )
    if not (settings.champion_dir / champion.MANIFEST_FILE).exists():
        raise EvaluationNotReadyError(f"There is no champion in {settings.champion_dir}.")
    manifest = champion.load_champion(settings.champion_dir).manifest
    if strict:
        _assert_real_execution(settings, manifest)
    _verify_registry(settings, manifest)


def evaluate(
    test: pd.DataFrame,
    development: pd.DataFrame,
    *,
    settings: config.Settings,
    replicates: int = evaluator.BOOTSTRAP_REPLICATES,
    force: bool = False,
    seed: int = model_trainer.DEFAULT_SEED,
    strict: bool = True,
) -> None:
    """Execute the test once, on the models exactly as they were saved, and write every table.

    Nothing gets trained here. Everything is computed in memory, and the record appears together
    with the results, so a failure leaves no figures to read and does not spend the test.

    Args:
        test: Rows of the test period.
        development: Development rows: what the base model learned from is the reference of the PSI.
        settings: Where to read the models from and where to write.
        replicates: Bootstrap replicates.
        force: Repeat the execution. The record declares it.
        seed: Seed of the bootstrap.
        strict: Demand the conditions of the official execution. Only the tests turn it off.

    Raises:
        TestAlreadyEvaluatedError: If it already ran and `force` is false.
        EvaluationNotReadyError: If a model or the environment is not the expected one.
        ValueError: If a row is outside the test period.
    """
    prepare_evaluation(settings, force=force, strict=strict)
    dates = test[schema.ARRIVAL_DATE]
    if dates.min() < config.TEST_START or dates.max() >= config.TEST_END:
        raise ValueError("Only rows of the test period can be evaluated.")
    _execute(test, development, settings=settings, replicates=replicates, seed=seed)


def rehearse(
    *,
    settings: config.Settings,
    window: str = "2016Q4",
    output_dir: Path,
    replicates: int = evaluator.BOOTSTRAP_REPLICATES,
    seed: int = model_trainer.DEFAULT_SEED,
) -> Path:
    """Run the whole evaluation with a window of 2016 as a pseudo test, in another folder.

    It proves that the code works on real rows without reading the test. Its figures do not count:
    the calibrator of the champion was fitted on 2016T4.

    Args:
        settings: Where the models are.
        window: Name of an expanding window.
        output_dir: Where to write instead of the real outputs.
        replicates: Bootstrap replicates.
        seed: Seed of the bootstrap.

    Returns:
        The folder with the results.

    Raises:
        ValueError: If the window does not exist.
    """
    chosen = next((w for w in config.EXPANDING_WINDOWS if w.name == window), None)
    if chosen is None:
        raise ValueError(f"{window} is not a window of the protocol.")
    target = settings.model_copy(update={"outputs_dir": output_dir})
    prepare_evaluation(target, strict=False)
    development = load_development()
    dates = development[schema.ARRIVAL_DATE]
    rows = development[(dates >= chosen.start) & (dates < chosen.end)].reset_index(drop=True)
    _execute(rows, development, settings=target, replicates=replicates, seed=seed)
    return target.final_evaluation_dir


def _execute(
    test: pd.DataFrame,
    development: pd.DataFrame,
    *,
    settings: config.Settings,
    replicates: int,
    seed: int,
) -> None:
    """Compute every table in memory and publish them together with the record."""
    dates = test[schema.ARRIVAL_DATE]
    loaded = champion.load_champion(settings.champion_dir)
    manifest = loaded.manifest
    registry = json.loads(settings.candidates_registry_path.read_text("utf-8"))
    candidates = {
        name: joblib.load(settings.models_dir / entry["file"])
        for name, entry in sorted(registry["candidates"].items())
    }
    scenario = joblib.load(settings.models_dir / registry["scenario"]["file"])

    rows = data_loader.coerce_features(test)
    target = test[schema.TARGET]
    compared = pd.DataFrame(
        {name: model.predict_proba(rows)[:, 1] for name, model in candidates.items()},
        index=test.index,
    )
    served = pd.Series(loaded.model.predict_proba(rows)[:, 1], index=test.index, name="champion")
    informative = pd.DataFrame(
        {SCENARIO_NAME: scenario.predict_proba(model_trainer.scenario_frame(test))[:, 1]},
        index=test.index,
    )
    decision = float(manifest["decision"]["threshold"])

    hotel_date = rows["hotel"].astype("str") + "|" + dates.astype("str")
    week = dates.dt.strftime("%G-W%V")
    bootstrap = pd.concat(
        [
            evaluator.block_bootstrap_auc(
                target,
                compared.join(informative),
                blocks,
                replicates=replicates,
                seed=seed,
                against=manifest["model"]["name"],
            ).assign(blocks=name)
            for name, blocks in (("hotel_date", hotel_date), ("week", week))
        ]
    )
    learned_from = development[development[schema.ARRIVAL_DATE] < config.CHAMPION_CUT.base_end]
    tables = {
        "metrics.csv": _metrics_table(target, compared, rows, decision),
        "scenario_special_requests.csv": _metrics_table(target, informative, rows, decision).assign(
            informative_only=True
        ),
        "segments.csv": _segments_table(target, served, rows, decision),
        "monthly.csv": _monthly_table(target, served, dates, rows),
        "bootstrap.csv": bootstrap,
        "psi.csv": _psi_table(rows, data_loader.coerce_features(learned_from)),
    }
    predictions = compared.join(informative).assign(
        champion=served,
        **{
            schema.ARRIVAL_DATE: dates.dt.strftime("%Y-%m-%d"),
            schema.TARGET: target,
            "hotel": rows["hotel"],
            "deposit_type": rows["deposit_type"],
        },
    )
    executions = _previous_executions(settings)
    forced = bool(executions)
    previous_count = len(executions)
    executions.append(
        {
            "executed_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            "code": champion.code_state(),
            "champion_sha256": manifest["model"]["sha256"],
            "champion_name": manifest["model"]["name"],
            "scenario_sha256": registry["scenario"]["sha256"],
            "test_rows": len(test),
            "forced": forced,
        }
    )
    _publish(tables, predictions, executions, settings, previous_count)


def _publish(
    tables: Mapping[str, pd.DataFrame],
    predictions: pd.DataFrame,
    executions: list[dict[str, Any]],
    settings: config.Settings,
    previous_count: int,
) -> None:
    """Write everything aside and then move the folder, so results and record appear as one."""
    folder = settings.final_evaluation_dir
    staging = folder.with_name(f"{folder.name}.staging")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        for name, table in tables.items():
            _write_table(table, staging / name)
        predictions.to_csv(staging / "predictions.csv.gz", index=False, lineterminator="\n")
        champion.write_json(
            staging / settings.final_evaluation_record.name, {"executions": executions}
        )
        if folder.exists():
            folder.rename(folder.with_name(f"{folder.name}.superseded-{previous_count}"))
        staging.rename(folder)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _write_table(table: pd.DataFrame, path: Path) -> None:
    table.round(6).to_csv(path, index=False, lineterminator="\n")


def _psi_table(test_rows: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": list(schema.INPUT_FEATURES),
            "psi": [
                evaluator.population_stability_index(reference[column], test_rows[column])
                for column in schema.INPUT_FEATURES
            ],
        }
    ).sort_values("psi", ascending=False)


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
    _assert_champion_can_change(settings, force=force)
    tune(development, settings=settings, profile=profile, seed=seed, tracker=tracker)
    return train(
        development, settings=settings, profile=profile, seed=seed, tracker=tracker, force=force
    )
