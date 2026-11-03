"""The single execution on the test, and its rehearsal on a window of 2016.

Nothing gets trained here. Only `evaluate` measures the test rows, and `prepare_evaluation` checks
everything it needs before they are read.
"""

import shutil
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from hotel_cancellation import champion, config, models
from hotel_cancellation.data import data_loader, schema
from hotel_cancellation.evaluation import evaluator
from hotel_cancellation.training import model_trainer

_SEGMENTS = ("hotel", "market_segment", "customer_type", "distribution_channel", "deposit_type")

SCENARIO_NAME = "special_requests_scenario"


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
    executions: list[dict[str, Any]] = champion.read_json(path)["executions"]
    return executions


def _assert_real_execution(settings: config.Settings, manifest: Mapping[str, Any]) -> None:
    """Demand what makes this execution the official one, so nothing can escape the record."""
    if manifest["profile"] != "full":
        raise champion.EvaluationNotReadyError(f"Profile must be full, not {manifest['profile']}.")
    if not champion.tree_is_clean():
        raise champion.EvaluationNotReadyError(
            "The git tree must be clean: commit everything first."
        )
    if not config.same_path(settings.outputs_dir, config.DEFAULT_OUTPUTS_DIR):
        raise champion.EvaluationNotReadyError(
            "Unset HOTEL_ML_OUTPUTS_DIR: the record has one place."
        )
    if not config.same_path(settings.models_dir, config.DEFAULT_MODELS_DIR):
        raise champion.EvaluationNotReadyError(
            "Unset HOTEL_ML_MODELS_DIR: the champion has one place."
        )


def _verify_entry(name: str, entry: Mapping[str, Any], settings: config.Settings) -> None:
    path = settings.models_dir / entry["file"]
    if not path.exists():
        raise champion.EvaluationNotReadyError(
            f"{name}: {path} does not exist. Run `hotel-ml train`."
        )
    if champion.sha256_of(path) != entry["sha256"]:
        raise champion.EvaluationNotReadyError(
            f"{name}: {path} does not match its registered hash."
        )


def _verify_registry(settings: config.Settings, manifest: Mapping[str, Any]) -> None:
    path = settings.candidates_registry_path
    if not path.exists():
        raise champion.EvaluationNotReadyError(
            f"{path} does not exist. Run `hotel-ml train-scenario`."
        )
    registry = champion.read_json(path)
    candidates = registry["candidates"]
    missing = sorted(set(models.specs_by_name(manifest["profile"])).difference(candidates))
    if missing:
        raise champion.EvaluationNotReadyError(f"Models missing from {path}: {', '.join(missing)}.")
    for name, entry in candidates.items():
        _verify_entry(name, entry, settings)
    _verify_entry(SCENARIO_NAME, registry["scenario"], settings)
    name, digest = manifest["model"]["name"], manifest["model"]["sha256"]
    if candidates[name]["sha256"] != digest:
        raise champion.EvaluationNotReadyError(
            f"Candidate {name} is not the champion: hashes differ."
        )


def prepare_evaluation(
    settings: config.Settings, *, force: bool = False, strict: bool = True
) -> None:
    """Check everything the single execution needs, before reading a row of the test.

    Args:
        settings: Where the models and the record are.
        force: Allow a repetition.
        strict: Demand the conditions of the official execution.

    Raises:
        champion.TestAlreadyEvaluatedError: If it already ran and `force` is false.
        champion.EvaluationNotReadyError: If a model or the environment is not the expected one.
        ChampionIntegrityError: If the champion is not what its manifest says.
    """
    executions = _previous_executions(settings)
    if executions and not force:
        raise champion.TestAlreadyEvaluatedError(
            f"The test already ran on {executions[-1]['executed_at']}. "
            "Repeat it with --force, and declare it in the report."
        )
    if not (settings.champion_dir / champion.MANIFEST_FILE).exists():
        raise champion.EvaluationNotReadyError(f"There is no champion in {settings.champion_dir}.")
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
    seed: int = config.DEFAULT_SEED,
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
        champion.TestAlreadyEvaluatedError: If it already ran and `force` is false.
        champion.EvaluationNotReadyError: If a model or the environment is not the expected one.
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
    seed: int = config.DEFAULT_SEED,
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
    development = data_loader.load_development()
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
    registry = champion.read_json(settings.candidates_registry_path)
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
