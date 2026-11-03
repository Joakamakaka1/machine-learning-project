"""Saved results and the figures the interface states about them.

Everything is read from `outputs/` and `models/champion/`, so these pages need neither the API nor
a model. Nothing here evaluates anything: the costs are recomputed from saved predictions.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from hotel_cancellation import champion, config, evaluator, schema, stored_evaluation

CHAMPION_COLUMN = "champion"
TARGET_COLUMN = schema.TARGET
PREDICTIONS_FILE = "predictions.csv.gz"

_THRESHOLD_GRID = np.round(np.linspace(0.01, 0.99, 99), 2)
_LOWEST_SHOWN = 0.01
_HIGHEST_SHOWN = 0.99


def format_probability(probability: float) -> str:
    """Write a probability without presenting an extreme step of the calibrator as a certainty."""
    if probability < _LOWEST_SHOWN:
        return "<1 %"
    if probability > _HIGHEST_SHOWN:
        return ">99 %"
    return f"{probability * 100:.1f} %"


@st.cache_data(show_spinner=False)
def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_table(*parts: str) -> pd.DataFrame:
    """Read a table under `outputs/`."""
    return _read_csv(config.get_settings().outputs_dir.joinpath(*parts))


def load_evaluation_table(name: str) -> pd.DataFrame:
    """Read a table saved by the single execution."""
    return _read_csv(config.get_settings().final_evaluation_dir / name)


def load_predictions() -> pd.DataFrame:
    """Read the saved probabilities of 2017. It does not run any model."""
    return load_evaluation_table(PREDICTIONS_FILE)


@st.cache_data(show_spinner=False)
def _read_manifest(path: Path) -> dict[str, Any]:
    return champion.read_json(path)


def load_manifest() -> dict[str, Any]:
    """Read the manifest of the served model."""
    return _read_manifest(config.get_settings().champion_dir / champion.MANIFEST_FILE)


def figure_path(name: str) -> Path:
    """Locate a figure of the report."""
    return config.get_settings().figures_dir / name


def champion_name() -> str:
    """Name of the served model in the saved tables."""
    return str(load_manifest()["model"]["name"])


def champion_subsets() -> pd.DataFrame:
    """Saved metrics of the served model at its threshold, one row per subset."""
    metrics = load_evaluation_table(stored_evaluation.METRICS_FILE)
    chosen = (metrics["model"] == champion_name()) & (
        metrics["threshold_name"] == stored_evaluation.DECISION_THRESHOLD
    )
    return metrics[chosen].reset_index(drop=True)


def champion_interval() -> tuple[float, float]:
    """Saved 95 % interval of the ROC-AUC of the served model, by hotel and date blocks."""
    bootstrap = load_evaluation_table(stored_evaluation.BOOTSTRAP_FILE)
    row = bootstrap[(bootstrap["model"] == champion_name()) & (bootstrap["blocks"] == "hotel_date")]
    return float(row["ci_low"].iloc[0]), float(row["ci_high"].iloc[0])


@dataclass(frozen=True)
class ThresholdOutcome:
    """What the saved probabilities imply at one threshold."""

    threshold: float
    flagged_share: float
    recall: float
    precision: float
    false_alarms: int
    missed: int
    cost: float


def outcome_at(
    predictions: pd.DataFrame, threshold: float, scenario: evaluator.CostScenario
) -> ThresholdOutcome:
    """Compute what would happen at `threshold`, with the costs of `scenario`."""
    actual = predictions[TARGET_COLUMN].to_numpy()
    probabilities = predictions[CHAMPION_COLUMN].to_numpy()
    flagged = probabilities >= threshold
    true_alarms = int(np.count_nonzero(flagged & (actual == 1)))
    positives = int(np.count_nonzero(actual == 1))
    flagged_count = int(np.count_nonzero(flagged))
    return ThresholdOutcome(
        threshold=threshold,
        flagged_share=float(flagged.mean()),
        recall=true_alarms / positives if positives else float("nan"),
        precision=true_alarms / flagged_count if flagged_count else float("nan"),
        false_alarms=flagged_count - true_alarms,
        missed=positives - true_alarms,
        cost=evaluator.cost_per_booking(actual, probabilities, threshold, scenario),
    )


def cost_marking_all(predictions: pd.DataFrame, scenario: evaluator.CostScenario) -> float:
    """Cost without a model when every booking is marked: all the non-cancelled become alarms."""
    return float(scenario.false_positive * (predictions[TARGET_COLUMN] == 0).mean())


def cost_marking_none(predictions: pd.DataFrame, scenario: evaluator.CostScenario) -> float:
    """Cost without a model when no booking is marked: every cancellation is missed."""
    return float(scenario.false_negative * (predictions[TARGET_COLUMN] == 1).mean())


def cost_curve(predictions: pd.DataFrame, scenario: evaluator.CostScenario) -> pd.DataFrame:
    """Compute the cost and the share of marked bookings along a grid of hundredths."""
    outcomes = [outcome_at(predictions, float(value), scenario) for value in _THRESHOLD_GRID]
    return pd.DataFrame(
        {
            "threshold": _THRESHOLD_GRID,
            "cost": [outcome.cost for outcome in outcomes],
            "flagged_share": [outcome.flagged_share for outcome in outcomes],
        }
    )


def cost_range(
    predictions: pd.DataFrame, low: float, high: float, scenario: evaluator.CostScenario
) -> tuple[float, float]:
    """Lowest and highest cost between two thresholds of the grid, both included."""
    curve = cost_curve(predictions, scenario)
    costs = curve.loc[curve["threshold"].between(low, high), "cost"]
    return float(costs.min()), float(costs.max())


@dataclass(frozen=True)
class ProbabilitySteps:
    """How coarse the calibrated probabilities are."""

    distinct_values: int
    share_zero: float
    share_one: float


def probability_steps(predictions: pd.DataFrame) -> ProbabilitySteps:
    """Measure the steps of the isotonic calibrator on the saved probabilities."""
    probabilities = predictions[CHAMPION_COLUMN]
    return ProbabilitySteps(
        distinct_values=int(probabilities.nunique()),
        share_zero=float((probabilities == 0).mean()),
        share_one=float((probabilities == 1).mean()),
    )


def missing_results() -> list[str]:
    """List the saved results the pages need and cannot find."""
    settings = config.get_settings()
    needed = (
        settings.champion_dir / champion.MANIFEST_FILE,
        settings.final_evaluation_dir / stored_evaluation.METRICS_FILE,
        settings.final_evaluation_dir / PREDICTIONS_FILE,
    )
    return [str(path) for path in needed if not path.exists()]
