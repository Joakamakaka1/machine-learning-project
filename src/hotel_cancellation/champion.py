"""Package of the served model: the model file and a manifest that identifies it."""

import hashlib
import json
import math
import platform
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from hotel_cancellation import config
from hotel_cancellation.data import features, schema
from hotel_cancellation.evaluation import evaluator

MODEL_FILE = "model.joblib"
MANIFEST_FILE = "manifest.json"

_VERSIONED_LIBRARIES = ("scikit-learn", "xgboost", "pandas", "numpy", "joblib")
_CODE_PATHS = ("src", "pyproject.toml", "uv.lock")
COMPRESSION_LEVEL = 3
_GIT = "git"


class ChampionIntegrityError(ValueError):
    """The saved package is not what its manifest describes."""


class TestNotExecutedError(RuntimeError):
    """Rows of the test period can only be read for interpreting after the single execution."""

    __test__ = False


class TestAlreadyEvaluatedError(RuntimeError):
    """The test was already executed, and it is executed only once."""

    __test__ = False


class EvaluationNotReadyError(RuntimeError):
    """The models or the environment are not the ones the single execution demands."""


def assert_test_executed(settings: config.Settings) -> None:
    """Refuse anything that reads the test for interpreting before the single execution.

    Raises:
        TestNotExecutedError: If there is no record of the execution.
    """
    if not settings.final_evaluation_record.exists():
        raise TestNotExecutedError(
            f"{settings.final_evaluation_record} does not exist: run `hotel-ml evaluate` first."
        )


@dataclass(frozen=True)
class LoadedChampion:
    """The model together with the manifest that describes it."""

    model: Any
    manifest: dict[str, Any]


def to_jsonable(value: Any) -> Any:
    """Convert a value to what strict JSON allows: a metric that is `NaN` becomes `null`."""
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""
    content: dict[str, Any] = json.loads(path.read_text("utf-8"))
    return content


def write_json(path: Path, content: Any) -> None:
    """Write strict JSON with stable line endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_jsonable(content), indent=2, ensure_ascii=False, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")


def _git(*arguments: str) -> str:
    completed = subprocess.run(  # noqa: S603
        [_GIT, "-C", str(config.PROJECT_ROOT), *arguments],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def code_state() -> dict[str, Any]:
    """Identify the code: its commit, and whether what trains differs from that commit."""
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--", *_CODE_PATHS)),
    }


def tree_is_clean() -> bool:
    """Tell whether git sees no change at all: neither modified nor untracked files."""
    return not _git("status", "--porcelain")


def sha256_of(path: Path) -> str:
    """Hash a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def library_versions() -> dict[str, str]:
    """List the versions that can change how a saved model behaves."""
    versions = {"python": platform.python_version()}
    versions.update({library: metadata.version(library) for library in _VERSIONED_LIBRARIES})
    return versions


def _windows() -> dict[str, Any]:
    return {
        "development_end": str(config.TEST_START.date()),
        "expanding": [
            {"name": window.name, "start": str(window.start.date()), "end": str(window.end.date())}
            for window in config.EXPANDING_WINDOWS
        ],
        "calibration_selection": {
            "base_end": str(config.CALIBRATION_SELECTION.base_end.date()),
            "calibration_end": str(config.CALIBRATION_SELECTION.calibration_end.date()),
        },
        "base_end": str(config.CHAMPION_CUT.base_end.date()),
        "calibration_end": str(config.CHAMPION_CUT.calibration_end.date()),
        "test_start": str(config.TEST_START.date()),
        "test_end": str(config.TEST_END.date()),
    }


def _decision(threshold_contrast: list[dict[str, float]]) -> dict[str, Any]:
    scenario = evaluator.DEFAULT_COST_SCENARIO
    return {
        "threshold": evaluator.optimal_threshold(scenario),
        "false_positive_cost": scenario.false_positive,
        "false_negative_cost": scenario.false_negative,
        "costs_are_assumptions": True,
        "cost_scenarios": [
            {
                "false_positive_cost": option.false_positive,
                "false_negative_cost": option.false_negative,
                "threshold": evaluator.optimal_threshold(option),
            }
            for option in evaluator.COST_SCENARIOS
        ],
        "contrast_on_2016q4": threshold_contrast,
    }


def save_champion(
    model: Any,
    folder: Path,
    *,
    model_name: str,
    params: Mapping[str, Any],
    calibration_method: str,
    calibration_briers: Mapping[str, float],
    threshold_contrast: list[dict[str, float]],
    validation: Mapping[str, Any],
    selection: Sequence[Mapping[Any, Any]],
    seed: int,
    profile: str,
    development_rows: int,
) -> dict[str, Any]:
    """Write the model and its manifest.

    Args:
        model: The fitted model, exactly as it will be evaluated and served.
        folder: Package folder. Its previous contents are replaced.
        model_name: Name in the catalog.
        params: Chosen hyperparameters.
        calibration_method: Chosen calibration, or `none`.
        calibration_briers: Brier of each calibration method on 2016T4.
        threshold_contrast: Theoretical against empirical threshold on 2016T4.
        validation: Validation metrics of the model.
        selection: Table of the one standard error rule.
        seed: Seed of the search and the model.
        profile: Training profile.
        development_rows: Rows available for development.

    Returns:
        The manifest.

    Raises:
        LeakageError: If the model uses a column that must not reach it. Nothing is written.
    """
    features.assert_no_forbidden_inputs(model)
    folder.mkdir(parents=True, exist_ok=True)
    model_path = folder / MODEL_FILE
    joblib.dump(model, model_path, compress=COMPRESSION_LEVEL)
    content = model_path.read_bytes()
    manifest = {
        "created_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "code": code_state(),
        "versions": library_versions(),
        "dataset_sha256": config.RAW_DATASET_SHA256,
        "development_rows": development_rows,
        "seed": seed,
        "profile": profile,
        "model": {
            "file": MODEL_FILE,
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
            "name": model_name,
            "params": params,
            "calibration": calibration_method,
            "calibration_briers_on_2016q4": calibration_briers,
        },
        "input_features": schema.INPUT_FEATURES,
        "decision": _decision(threshold_contrast),
        "windows": _windows(),
        "validation": validation,
        "selection": selection,
    }
    write_json(folder / MANIFEST_FILE, manifest)
    jsonable: dict[str, Any] = to_jsonable(manifest)
    return jsonable


def load_champion(folder: Path) -> LoadedChampion:
    """Load the model after checking its hash and the versions of the libraries.

    The hash detects a corrupted or replaced file, not a deliberate change of both files: loading
    a `joblib` file runs code, so only a model from this repository may be loaded.

    Args:
        folder: Package folder.

    Returns:
        The model and its manifest.

    Raises:
        ChampionIntegrityError: If the file changed or a library is on another version.
    """
    manifest = read_json(folder / MANIFEST_FILE)
    model_path = folder / MODEL_FILE
    actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if actual != manifest["model"]["sha256"]:
        raise ChampionIntegrityError(
            f"{model_path} has SHA-256 {actual}, "
            f"but the manifest says {manifest['model']['sha256']}."
        )
    current = library_versions()
    different = [
        f"{library}: saved with {manifest['versions'][library]}, running {current[library]}"
        for library in _VERSIONED_LIBRARIES
        if manifest["versions"][library] != current[library]
    ]
    if different:
        raise ChampionIntegrityError("Library versions differ:\n- " + "\n- ".join(different))
    return LoadedChampion(model=joblib.load(model_path), manifest=manifest)
