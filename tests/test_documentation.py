"""The figures in the documents must be the ones in the result files, and their links must exist."""

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from hotel_cancellation import config

ROOT = config.PROJECT_ROOT
README = ROOT / "README.md"
REPORT = ROOT / "docs" / "informe_final.md"
MODEL_CARD = ROOT / "docs" / "model_card.md"
DATA_CARD = ROOT / "docs" / "data_card.md"
DEFENSE = ROOT / "docs" / "defensa.md"
REVIEW_REPORT = ROOT / "docs" / "revision_datos_y_validacion.md"
RESULTS = ROOT / "outputs" / "final_evaluation"
REVIEW = ROOT / "outputs" / "review"
DOCUMENTS = (
    README,
    REPORT,
    MODEL_CARD,
    DATA_CARD,
    DEFENSE,
    REVIEW_REPORT,
    ROOT / "docs" / "revision_protocolo.md",
    ROOT / "docs" / "protocolo.md",
    ROOT / "docs" / "enunciado.md",
    ROOT / "data" / "raw" / "README.md",
    ROOT / "docs" / "glosario.md",
    ROOT / "CLAUDE.md",
    *sorted((ROOT / "docs" / "adr").glob("*.md")),
    *sorted((ROOT / "notebooks").glob("*.ipynb")),
)

CHAMPION = "random_forest"
MODELS = {
    "random_forest": "**Random Forest (campeón)**",
    "xgboost": "XGBoost",
    "mlp": "MLP",
    "decision_tree": "Árbol de decisión",
    "logistic_regression": "Regresión logística",
    "dummy": "Dummy (baseline)",
}
SUBSETS = {
    "all": "Todas",
    "unique": "Únicas",
    "no_deposit": "`No Deposit`",
    "non_refund": "`Non Refund`",
}
_LINK = re.compile(r"\]\(([^)\s]+)\)")

Row = Mapping[str, Any]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _table(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = json.loads(pd.read_csv(path).to_json(orient="records"))
    return records


def _row(rows: Iterable[Row], **filters: Any) -> Row:
    return next(row for row in rows if all(row[key] == value for key, value in filters.items()))


def _has_row(text: str, label: str, values: Iterable[str]) -> bool:
    """Check that some table row starts with the label and holds every value as a whole cell."""
    expected = set(values)
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells[0] == label and expected <= set(cells[1:]):
            return True
    return False


def _decimals(row: Row, columns: Iterable[str]) -> list[str]:
    return [f"{row[column]:.4f}" for column in columns]


def _states(text: str, value: float) -> bool:
    """Check a figure of the review, which keeps five decimals, at the four of the documents."""
    return f"{value - 5e-6:.4f}" in text or f"{value + 5e-6:.4f}" in text


@pytest.fixture(scope="module")
def metrics() -> list[dict[str, Any]]:
    return _table(RESULTS / "metrics.csv")


@pytest.fixture(scope="module")
def intervals() -> list[dict[str, Any]]:
    return [row for row in _table(RESULTS / "bootstrap.csv") if row["blocks"] == "hotel_date"]


@pytest.mark.parametrize("path", [README, REPORT], ids=lambda path: path.name)
@pytest.mark.parametrize("model", list(MODELS))
def test_each_model_row_comes_from_the_results(
    path: Path, model: str, metrics: list[dict[str, Any]], intervals: list[dict[str, Any]]
) -> None:
    row = _row(metrics, model=model, threshold_name="decision", subset="all")
    interval = _row(intervals, model=model)
    expected = [
        f"{row['roc_auc']:.4f}",
        f"{row['pr_auc']:.4f}",
        f"[{interval['ci_low']:.4f}, {interval['ci_high']:.4f}]",
    ]
    assert _has_row(_read(path), MODELS[model], expected)


@pytest.mark.parametrize("model", list(MODELS))
def test_the_readme_accuracy_and_f1_come_from_the_results(
    model: str, metrics: list[dict[str, Any]]
) -> None:
    decision = _row(metrics, model=model, threshold_name="decision", subset="all")
    fixed = _row(metrics, model=model, threshold_name="fixed_0.5", subset="all")
    expected = _decimals(decision, ["accuracy", "f1"]) + _decimals(fixed, ["accuracy", "f1"])
    assert _has_row(_read(README), MODELS[model], expected)


@pytest.mark.parametrize("path", [README, REPORT, MODEL_CARD], ids=lambda path: path.name)
@pytest.mark.parametrize("subset", list(SUBSETS))
def test_the_champion_segments_come_from_the_results(
    path: Path, subset: str, metrics: list[dict[str, Any]]
) -> None:
    row = _row(metrics, model=CHAMPION, threshold_name="decision", subset=subset)
    expected = _decimals(
        row, ["prevalence", "roc_auc", "pr_auc", "brier", "precision", "recall", "f1", "cost_1_3"]
    )
    assert _has_row(_read(path), SUBSETS[subset], expected)


@pytest.mark.parametrize("path", [README, REPORT, MODEL_CARD], ids=lambda path: path.name)
@pytest.mark.parametrize("hotel", ["City Hotel", "Resort Hotel"])
def test_the_hotel_segments_come_from_the_results(path: Path, hotel: str) -> None:
    row = _row(_table(RESULTS / "segments.csv"), segment="hotel", value=hotel)
    assert _has_row(
        _read(path), f"`{hotel}`", _decimals(row, ["prevalence", "roc_auc", "pr_auc", "brier"])
    )


def test_the_inflation_steps_come_from_the_audit() -> None:
    readme, report = _read(README), _read(REPORT)
    for row in _table(ROOT / "outputs" / "audit" / "inflation.csv"):
        assert f"{row['roc_auc']:.4f}" in readme
        assert _has_row(report, row["label"], _decimals(row, ["roc_auc", "pr_auc"]))


def test_the_validation_table_comes_from_the_summary() -> None:
    summary = _table(ROOT / "outputs" / "validation" / "summary.csv")
    report = _read(REPORT)
    labels = {**MODELS, "random_forest": "Random Forest"}
    columns = ["roc_auc_mean", "roc_auc_std", "roc_auc_unique_mean", "pr_auc_mean", "brier_mean"]
    for model, label in labels.items():
        assert _has_row(report, label, _decimals(_row(summary, model=model), columns))


def test_the_served_decision_is_stated_as_in_the_business_table() -> None:
    business = _table(ROOT / "outputs" / "business" / "value_by_cost_scenario.csv")
    served = _row(business, threshold=0.25)
    flagged = f"{100 * served['flagged_share']:.1f} %"
    false_alarms = f"{int(served['false_alarms']):,}".replace(",", " ")
    for path in (README, REPORT, MODEL_CARD):
        assert flagged in _read(path)
    assert false_alarms in _read(REPORT)
    assert false_alarms in _read(MODEL_CARD)


def test_the_champion_hash_in_the_documents_is_the_one_in_the_manifest() -> None:
    manifest = json.loads(_read(ROOT / "models" / "champion" / "manifest.json"))
    sha256 = manifest["model"]["sha256"]
    for path in (REPORT, MODEL_CARD, ROOT / "CLAUDE.md"):
        assert sha256 in _read(path)


@pytest.mark.parametrize(
    "path",
    [README, REPORT, MODEL_CARD, DEFENSE, REVIEW_REPORT],
    ids=lambda path: path.name,
)
def test_the_figures_after_the_deployment_come_from_the_review(path: Path) -> None:
    booked = _row(_table(REVIEW / "posthoc_by_booking_date.csv"), subset="booked_from_2017_01_01")
    late = _row(
        _table(REVIEW / "posthoc_positive_kinds.csv"), kind="canceled_within_7_days_of_arrival"
    )
    text = _read(path)
    assert _states(text, booked["roc_auc"])
    assert _states(text, booked["roc_auc_no_deposit"])
    if path is not REVIEW_REPORT:
        assert _states(text, late["auc_against_stays"])


def test_the_interface_states_the_figure_after_the_deployment() -> None:
    booked = _row(_table(REVIEW / "posthoc_by_booking_date.csv"), subset="booked_from_2017_01_01")
    overview = ROOT / "src" / "hotel_cancellation" / "serving" / "ui" / "views" / "overview.py"
    assert _states(_read(overview), booked["roc_auc"])


@pytest.mark.parametrize(
    "path", [REPORT, MODEL_CARD, DATA_CARD, DEFENSE, REVIEW_REPORT], ids=lambda path: path.name
)
def test_the_figure_with_the_variables_defined_at_booking_comes_from_the_review(path: Path) -> None:
    tiers = _row(_table(REVIEW / "tiers_m12.csv"), variant="by_definition_only")
    assert _states(_read(path), tiers["roc_auc"])


def test_no_change_of_the_review_passes_the_acceptance_rule() -> None:
    decisions = _table(REVIEW / "decision_log.csv")
    assert len(decisions) == 50
    assert not any(row["accepted"] for row in decisions)
    assert "0 de 50" in _read(REVIEW_REPORT)


def test_the_interval_by_cancellation_event_comes_from_the_review() -> None:
    interval = _row(_table(REVIEW / "posthoc_intervals.csv"), blocks="hotel_status_day_agent")
    for path in (REPORT, MODEL_CARD, REVIEW_REPORT):
        text = _read(path)
        assert _states(text, interval["ci_low"])
        assert _states(text, interval["ci_high"])


def test_the_report_states_the_repository() -> None:
    assert "https://github.com/Joakamakaka1/machine-learning-project" in _read(REPORT)


@pytest.mark.parametrize("path", DOCUMENTS, ids=lambda path: path.name)
def test_relative_links_exist(path: Path) -> None:
    targets = [
        target.split("#")[0]
        for target in _LINK.findall(_read(path))
        if not target.startswith(("http", "#", "mailto:"))
    ]
    missing = [target for target in targets if not (path.parent / target).exists()]
    assert not missing
