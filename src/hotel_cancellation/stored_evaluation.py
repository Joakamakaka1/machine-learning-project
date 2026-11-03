"""Read the results of the single execution on the test. Nothing here computes anything."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from hotel_cancellation.champion import read_json, to_jsonable

RECORD_FILE = "record.json"
METRICS_FILE = "metrics.csv"
SEGMENTS_FILE = "segments.csv"
BOOTSTRAP_FILE = "bootstrap.csv"
MONTHLY_FILE = "monthly.csv"
PSI_FILE = "psi.csv"
SCENARIO_FILE = "scenario_special_requests.csv"

DECISION_THRESHOLD = "decision"
_MAIN_BLOCKS = "hotel_date"
_SUMMARY_COLUMNS = (
    "subset",
    "rows",
    "prevalence",
    "roc_auc",
    "pr_auc",
    "brier",
    "calibration_slope",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "cost_1_3",
)


def _records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = to_jsonable(pd.read_csv(path).to_dict("records"))
    return records


@dataclass(frozen=True)
class StoredEvaluation:
    """Tables that `hotel-ml evaluate` saved, as JSON-ready rows."""

    record: dict[str, Any]
    metrics: list[dict[str, Any]]
    segments: list[dict[str, Any]]
    bootstrap: list[dict[str, Any]]
    monthly: list[dict[str, Any]]
    psi: list[dict[str, Any]]
    special_requests_scenario: list[dict[str, Any]]

    @property
    def execution(self) -> dict[str, Any]:
        """The execution that counts: the last one registered."""
        last: dict[str, Any] = self.record["executions"][-1]
        return last

    def champion_summary(self) -> dict[str, Any]:
        """Summarize the served model on the test: never the global ROC-AUC alone."""
        name = self.execution["champion_name"]
        subsets = [
            {column: row[column] for column in _SUMMARY_COLUMNS}
            for row in self.metrics
            if row["model"] == name and row["threshold_name"] == DECISION_THRESHOLD
        ]
        interval = next(
            row for row in self.bootstrap if row["model"] == name and row["blocks"] == _MAIN_BLOCKS
        )
        return {
            "executed_at": self.execution["executed_at"],
            "subsets": subsets,
            "roc_auc_ci_95": [interval["ci_low"], interval["ci_high"]],
        }


def load_stored_evaluation(folder: Path) -> StoredEvaluation | None:
    """Read the saved evaluation.

    Args:
        folder: Folder of the single execution.

    Returns:
        Its tables, or `None` if there is no record of an execution.
    """
    if not (folder / RECORD_FILE).exists():
        return None
    return StoredEvaluation(
        record=read_json(folder / RECORD_FILE),
        metrics=_records(folder / METRICS_FILE),
        segments=_records(folder / SEGMENTS_FILE),
        bootstrap=_records(folder / BOOTSTRAP_FILE),
        monthly=_records(folder / MONTHLY_FILE),
        psi=_records(folder / PSI_FILE),
        special_requests_scenario=_records(folder / SCENARIO_FILE),
    )
