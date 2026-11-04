"""Static figures of the report, in Spanish, from the results already saved.

Nothing here reads the raw test rows or touches a model. The figures follow one system: categorical
hues in a fixed order per model, a neutral grey for the baseline, one axis per plot and solid
hairline grids. Every value is also in the tables of `docs/informe_final.md`.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

from hotel_cancellation import champion, config
from hotel_cancellation.data import schema
from hotel_cancellation.evaluation import evaluator

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
ACCENT = "#2a78d6"
# Ordinal steps of the accent, for what has a natural order, such as the cost scenarios.
ORDINAL_RAMP = ("#86b6ef", "#2a78d6", "#104281")

# Colour follows the model, never its rank. The order passes the colour-blindness checks.
MODEL_COLORS: Mapping[str, str] = {
    "random_forest": "#2a78d6",
    "xgboost": "#eb6834",
    "mlp": "#1baf7a",
    "logistic_regression": "#eda100",
    "decision_tree": "#e87ba4",
    "dummy": MUTED,
}
MODEL_LABELS: Mapping[str, str] = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "mlp": "MLP",
    "logistic_regression": "Regresión logística",
    "decision_tree": "Árbol de decisión",
    "dummy": "Dummy (baseline)",
    "special_requests_scenario": "Escenario con peticiones especiales (informativo)",
}
_MODELS = tuple(MODEL_COLORS)
_COMPARED = tuple(name for name in _MODELS if name != "dummy")
_THRESHOLDS = np.linspace(0.01, 0.99, 99)
_CALIBRATION_BINS = 10
_PSI_MODERATE = 0.10
_PSI_LARGE = 0.25
_DARK_CELL = 0.5
_DPI = 160

# Public because the notebooks draw with the same system.
STYLE: Any = {
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": SECONDARY_INK,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": SECONDARY_INK,
    "ytick.labelcolor": SECONDARY_INK,
    "text.color": INK,
    "legend.frameon": False,
    "legend.labelcolor": SECONDARY_INK,
    "lines.linewidth": 2,
}


def _figure(width: float, height: float, rows: int = 1, columns: int = 1) -> tuple[Figure, Any]:
    figure = Figure(figsize=(width, height), layout="constrained")
    return figure, figure.subplots(rows, columns, squeeze=False)


def _available(predictions: pd.DataFrame) -> list[str]:
    return [name for name in _COMPARED if name in predictions]


def _label(name: str, champion_name: str) -> str:
    label = MODEL_LABELS[name]
    return f"{label} (campeón)" if name == champion_name else label


def _line_width(name: str, champion_name: str) -> float:
    return 2.8 if name == champion_name else 1.6


def roc_figure(predictions: pd.DataFrame, champion_name: str) -> Figure:
    """Compare the ROC curves of every model and the baseline."""
    figure, axes = _figure(7, 6)
    axis = axes[0, 0]
    actual = predictions[schema.TARGET]
    axis.plot([0, 1], [0, 1], color=MUTED, linewidth=1.4, label="Dummy (baseline) · 0.500")
    for name in _available(predictions):
        false_positive_rate, true_positive_rate, _ = roc_curve(actual, predictions[name])
        area = float(np.trapezoid(true_positive_rate, false_positive_rate))
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            color=MODEL_COLORS[name],
            linewidth=_line_width(name, champion_name),
            label=f"{_label(name, champion_name)} · {area:.3f}",
            zorder=3 if name == champion_name else 2,
        )
    axis.set_title("Curvas ROC en el test de 2017")
    axis.set_xlabel("Proporción de falsas alarmas")
    axis.set_ylabel("Proporción de cancelaciones detectadas")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_aspect("equal")
    axis.legend(loc="lower right", title="Modelo · ROC-AUC")
    return figure


def precision_recall_figure(predictions: pd.DataFrame, champion_name: str) -> Figure:
    """Compare the precision and recall of every model and the baseline."""
    figure, axes = _figure(7, 5.2)
    axis = axes[0, 0]
    actual = predictions[schema.TARGET]
    prevalence = float(actual.mean())
    axis.axhline(
        prevalence, color=MUTED, linewidth=1.4, label=f"Dummy (baseline) · {prevalence:.3f}"
    )
    for name in _available(predictions):
        precision, recall, _ = precision_recall_curve(actual, predictions[name])
        area = float(average_precision_score(actual, predictions[name]))
        axis.plot(
            recall,
            precision,
            color=MODEL_COLORS[name],
            linewidth=_line_width(name, champion_name),
            label=f"{_label(name, champion_name)} · {area:.3f}",
            zorder=3 if name == champion_name else 2,
        )
    axis.set_title("Curvas de precisión y recall en el test de 2017")
    axis.set_xlabel("Recall (cancelaciones detectadas)")
    axis.set_ylabel("Precisión (avisos acertados)")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.legend(loc="lower left", title="Modelo · PR-AUC")
    return figure


def intervals_figure(bootstrap: pd.DataFrame, champion_name: str) -> Figure:
    """Show the ROC-AUC of each model with its two bootstrap intervals.

    The baseline stays out of the scale, because including it would hide the intervals.
    """
    panels = (("hotel_date", "Bloques de hotel y fecha"), ("week", "Bloques por semana"))
    order = [*_COMPARED[::-1], "special_requests_scenario"]
    figure, axes = _figure(11, 3.8, 1, 2)
    for axis, (blocks, title) in zip(axes[0], panels, strict=True):
        table = bootstrap[bootstrap["blocks"] == blocks].set_index("model")
        names = [name for name in order if name in table.index]
        for position, name in enumerate(names):
            row = table.loc[name]
            color = ACCENT if name == champion_name else MUTED
            axis.plot([row["ci_low"], row["ci_high"]], [position, position], color=color, zorder=2)
            axis.plot(row["roc_auc"], position, "o", color=color, markersize=7, zorder=3)
        axis.set_yticks(range(len(names)), [_label(name, champion_name) for name in names])
        axis.set_title(title)
        axis.grid(visible=False, axis="y")
    figure.axes[0].sharex(figure.axes[1])
    figure.supxlabel(
        "ROC-AUC con intervalo del 95 %. El Dummy, con 0.500, queda fuera de escala",
        fontsize=10,
        color=SECONDARY_INK,
    )
    return figure


def confusion_figure(matrices: Mapping[str, tuple[int, int, int, int]]) -> Figure:
    """Draw confusion matrices with counts and percentages of all the bookings.

    Args:
        matrices: For each panel title, the counts `(tn, fp, fn, tp)`.

    Returns:
        The figure, with a panel per matrix.
    """
    columns = len(matrices)
    figure, axes = _figure(3.3 * columns + 0.6, 3.5, 1, columns)
    classes = ["No cancela", "Cancela"]
    color_map = mpl.colors.LinearSegmentedColormap.from_list("azul", ["#eaf2fc", "#104281"])
    for axis, (title, (tn, fp, fn, tp)) in zip(axes[0], matrices.items(), strict=True):
        cells = np.array([[tn, fp], [fn, tp]], dtype=float)
        total = cells.sum()
        shares = cells / total
        axis.imshow(shares, cmap=color_map, vmin=0, vmax=max(0.6, float(shares.max())))
        for row in range(2):
            for column in range(2):
                dark = shares[row, column] > _DARK_CELL * max(0.6, float(shares.max()))
                axis.text(
                    column,
                    row,
                    f"{int(cells[row, column]):,}\n{shares[row, column]:.1%}".replace(
                        ",", " "
                    ).replace("%", " %"),
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="#ffffff" if dark else INK,
                )
        axis.set_xticks([0, 1], classes)
        axis.set_yticks([0, 1], classes)
        axis.set_xlabel("Predicción")
        axis.set_ylabel("Realidad")
        axis.set_title(title)
        axis.grid(visible=False)
        for spine in axis.spines.values():
            spine.set_visible(False)
    return figure


def _confusion_counts(actual: pd.Series, probabilities: pd.Series, threshold: float) -> Any:
    predicted = probabilities >= threshold
    positive = actual == 1
    return (
        int((~predicted & ~positive).sum()),
        int((predicted & ~positive).sum()),
        int((~predicted & positive).sum()),
        int((predicted & positive).sum()),
    )


def _scenario_title(scenario: evaluator.CostScenario, threshold: float) -> str:
    ratio = f"{scenario.false_positive:g}:{scenario.false_negative:g}"
    return f"Costes {ratio}, umbral {threshold:.3f}"


def calibration_figure(predictions: pd.DataFrame, champion_name: str) -> Figure:
    """Compare predicted and observed cancellation per model, and show the served probabilities."""
    figure, axes = _figure(10, 6.4, 2, 3)
    actual = predictions[schema.TARGET]
    for axis, name in zip(axes.flat, _available(predictions), strict=False):
        frame = pd.DataFrame({"p": predictions[name], "y": actual})
        frame["bin"] = pd.qcut(frame["p"], _CALIBRATION_BINS, duplicates="drop")
        means = frame.groupby("bin", observed=True)[["p", "y"]].mean()
        axis.plot([0, 1], [0, 1], color=MUTED, linewidth=1)
        axis.plot(means["p"], means["y"], "-o", color=MODEL_COLORS[name], markersize=5)
        axis.set_title(_label(name, champion_name), fontsize=11)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xlabel("Probabilidad predicha")
        axis.set_ylabel("Cancelación observada")
    last = axes[1, 2]
    last.hist(predictions[champion_name], bins=25, color=ACCENT, rwidth=0.9)
    last.set_title("Probabilidades del campeón", fontsize=11)
    last.set_xlabel("Probabilidad predicha")
    last.set_ylabel("Reservas")
    last.grid(visible=False, axis="x")
    figure.suptitle(
        "Calibración en el test de 2017, por deciles de probabilidad",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    return figure


def importance_figure(table: pd.DataFrame) -> Figure:
    """Compare the three importance methods, each one ordered by its own value."""
    panels = (
        ("forest_importance", "feature_importances_", "Reducción media de impureza"),
        ("permutation_mean", "Permutación", "Caída del ROC-AUC en 2017"),
        ("shap_mean_abs", "SHAP", "|SHAP| medio, sin calibrar"),
    )
    figure, axes = _figure(12, 6.2, 1, 3)
    for axis, (column, title, label) in zip(axes[0], panels, strict=True):
        ordered = table.sort_values(column)
        axis.barh(ordered["variable"], ordered[column], color=ACCENT, height=0.62)
        axis.set_title(title)
        axis.set_xlabel(label)
        axis.grid(visible=False, axis="y")
    figure.suptitle(
        "Importancia por variable original: interpretación, no causalidad",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    return figure


def monthly_auc_figure(monthly: pd.DataFrame) -> Figure:
    """Show the ROC-AUC of the champion month by month."""
    figure, axes = _figure(8, 3.6)
    axis = axes[0, 0]
    axis.plot(monthly["month"], monthly["roc_auc"], "-o", color=ACCENT, markersize=8)
    axis.set_title("ROC-AUC del campeón, mes a mes")
    axis.set_ylabel("ROC-AUC")
    axis.set_xlabel("Mes de llegada")
    axis.set_ylim(0.5, 1)
    aucs = monthly["roc_auc"].to_numpy()
    for extreme in (int(aucs.argmax()), int(aucs.argmin())):
        row = monthly.iloc[extreme]
        axis.annotate(
            f"{row['roc_auc']:.3f}",
            (row["month"], row["roc_auc"]),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            color=SECONDARY_INK,
        )
    return figure


def monthly_prevalence_figure(monthly: pd.DataFrame) -> Figure:
    """Show the observed cancellation month by month, in its own chart."""
    figure, axes = _figure(8, 3.6)
    axis = axes[0, 0]
    axis.bar(monthly["month"], monthly["prevalence"], color=ACCENT, width=0.55)
    axis.set_title("Cancelación observada, mes a mes")
    axis.set_ylabel("Prevalencia")
    axis.set_xlabel("Mes de llegada")
    axis.set_ylim(0, 0.6)
    axis.grid(visible=False, axis="x")
    return figure


def drift_figure(psi: pd.DataFrame) -> Figure:
    """Show how far each variable moved from what the base model learned from."""
    figure, axes = _figure(8, 5.8)
    axis = axes[0, 0]
    ordered = psi.sort_values("psi")
    axis.barh(ordered["feature"], ordered["psi"], color=ACCENT, height=0.62)
    for level, text in ((_PSI_MODERATE, "0.10 moderada"), (_PSI_LARGE, "0.25 grande")):
        axis.axvline(level, color=MUTED, linewidth=1)
        axis.text(level, len(ordered) - 0.4, f" {text}", color=SECONDARY_INK, va="bottom")
    axis.set_title("Deriva entre el entrenamiento y 2017 (PSI)", pad=18)
    axis.set_xlabel("PSI")
    axis.grid(visible=False, axis="y")
    return figure


def cost_figure(predictions: pd.DataFrame, champion_name: str) -> Figure:
    """Show the cost of the champion at each threshold, marking the threshold of each scenario."""
    figure, axes = _figure(8, 4.8)
    axis = axes[0, 0]
    actual = predictions[schema.TARGET]
    served = predictions[champion_name]
    for scenario, color in zip(evaluator.COST_SCENARIOS, ORDINAL_RAMP, strict=True):
        costs = [
            evaluator.cost_per_booking(actual, served, float(threshold), scenario)
            for threshold in _THRESHOLDS
        ]
        best = evaluator.optimal_threshold(scenario)
        at_best = evaluator.cost_per_booking(actual, served, best, scenario)
        ratio = f"{scenario.false_positive:g}:{scenario.false_negative:g}"
        axis.plot(_THRESHOLDS, costs, color=color, label=f"Costes {ratio}, umbral {best:.3f}")
        axis.plot(best, at_best, "o", color=color, markersize=9, zorder=3)
    axis.set_title("Coste del campeón según el umbral, con costes supuestos")
    axis.set_xlabel("Umbral de decisión")
    axis.set_ylabel("Coste por reserva")
    axis.set_xlim(0, 1)
    axis.set_ylim(bottom=0)
    axis.legend(title="Escenario (supuesto); el punto marca t*")
    return figure


def inflation_figure(table: pd.DataFrame) -> Figure:
    """Show the four steps from the inflated figure to the official one, on an axis from zero."""
    figure, axes = _figure(10, 3.6)
    axis = axes[0, 0]
    ordered = table.iloc[::-1]
    colors = [ACCENT if step == "official_model" else MUTED for step in ordered["step"]]
    axis.barh(ordered["label"], ordered["roc_auc"], color=colors, height=0.55)
    for position, value in enumerate(ordered["roc_auc"]):
        axis.text(value + 0.008, position, f"{value:.3f}", va="center", color=INK)
    axis.set_xlim(0, 1.08)
    axis.set_title("De la cifra inflada a la oficial: ROC-AUC en cada escalón")
    axis.set_xlabel("ROC-AUC")
    axis.grid(visible=False, axis="y")
    return figure


def _save(figure: Figure, folder: Path, name: str) -> Path:
    path = folder / f"{name}.png"
    figure.savefig(path, dpi=_DPI)
    return path


def save_all(settings: config.Settings) -> list[Path]:
    """Write every figure that the saved results allow, and the business value table.

    Args:
        settings: Where the results are and where to write.

    Returns:
        The written figures.

    Raises:
        TestNotExecutedError: If the single execution has not happened yet.
    """
    champion.assert_test_executed(settings)
    results = settings.final_evaluation_dir
    predictions = pd.read_csv(results / "predictions.csv.gz")
    record = champion.read_json(settings.final_evaluation_record)
    champion_name = record["executions"][-1]["champion_name"]
    decision = float(evaluator.optimal_threshold(evaluator.DEFAULT_COST_SCENARIO))
    monthly = pd.read_csv(results / "monthly.csv")
    actual = predictions[schema.TARGET]
    served = predictions[champion_name]

    folder = settings.figures_dir
    folder.mkdir(parents=True, exist_ok=True)
    settings.business_dir.mkdir(parents=True, exist_ok=True)
    evaluator.business_value_table(actual, served).round(6).to_csv(
        settings.business_dir / "value_by_cost_scenario.csv", index=False, lineterminator="\n"
    )
    with mpl.rc_context(STYLE):
        scenarios = {
            _scenario_title(scenario, evaluator.optimal_threshold(scenario)): _confusion_counts(
                actual, served, evaluator.optimal_threshold(scenario)
            )
            for scenario in evaluator.COST_SCENARIOS
        }
        models = {
            _label(name, champion_name): _confusion_counts(actual, predictions[name], decision)
            for name in _available(predictions)
        }
        figures = {
            "roc": roc_figure(predictions, champion_name),
            "precision_recall": precision_recall_figure(predictions, champion_name),
            "intervalos_roc_auc": intervals_figure(
                pd.read_csv(results / "bootstrap.csv"), champion_name
            ),
            "matrices_confusion_campeon": confusion_figure(scenarios),
            "matrices_confusion_modelos": confusion_figure(models),
            "calibracion": calibration_figure(predictions, champion_name),
            "roc_auc_por_mes": monthly_auc_figure(monthly),
            "prevalencia_por_mes": monthly_prevalence_figure(monthly),
            "deriva_psi": drift_figure(pd.read_csv(results / "psi.csv")),
            "coste_umbral": cost_figure(predictions, champion_name),
        }
        importance = settings.interpretation_dir / "importance.csv"
        if importance.exists():
            figures["importancia_variables"] = importance_figure(pd.read_csv(importance))
        inflation = settings.audit_dir / "inflation.csv"
        if inflation.exists():
            figures["inflacion"] = inflation_figure(pd.read_csv(inflation))
        return [_save(figure, folder, name) for name, figure in figures.items()]
