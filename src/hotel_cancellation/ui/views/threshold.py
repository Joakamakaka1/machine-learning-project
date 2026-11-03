"""Threshold and cost: what the served threshold implies, and what other costs would imply."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hotel_cancellation import evaluator
from hotel_cancellation.ui import layout, results

# The same tokens as the figures of the report: one accent, and grey for the reference.
_ACCENT = "#2a78d6"
_REFERENCE = "#898781"
_INK = "#52514e"
_GRID = "#e1e0d9"
_NO_MODEL_LABEL = "Sin modelo: marcar todas las reservas"

_DESCRIBED_LOW = 0.30
_DESCRIBED_HIGH = 0.40
_NEXT_POINT = 0.26
_SCENARIOS = {
    "scenario": "Costes FP:FN (supuestos)",
    "threshold": "Umbral",
    "recall": "Recall",
    "precision": "Precisión",
    "true_alarms": "Cancelaciones detectadas",
    "false_alarms": "Falsas alarmas",
    "missed_cancellations": "Cancelaciones perdidas",
    "flagged_share": "Reservas marcadas",
    "cost_per_booking": "Coste por reserva",
    "cost_without_model": _NO_MODEL_LABEL,
    "saving_per_booking": "Ahorro por reserva",
}
_SCENARIO_COUNTS = ("true_alarms", "false_alarms", "missed_cancellations")
_CONTRAST = {
    "scenario": "Costes FP:FN (supuestos)",
    "threshold": "Umbral teórico",
    "cost_at_threshold": "Coste con el umbral teórico",
    "empirical_best_threshold": "Mejor umbral observado",
    "cost_at_empirical_best": "Coste con el mejor umbral",
}


def _scenario_name(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["false_positive_cost"]
        .map("{:g}".format)
        .str.cat(frame["false_negative_cost"].map("{:g}".format), sep=":")
    )


def _cost_chart(
    curve: pd.DataFrame, no_model_cost: float, served: float, chosen: float
) -> go.Figure:
    figure = go.Figure()
    # The calibrator works in steps, and a step line does not invent values between them.
    figure.add_scatter(
        x=curve["threshold"],
        y=curve["cost"],
        name="Coste del modelo",
        mode="lines",
        line={"color": _ACCENT, "width": 2, "shape": "hv"},
        hovertemplate="%{y:.3f}",
    )
    figure.add_scatter(
        x=curve["threshold"],
        y=[no_model_cost] * len(curve),
        name=_NO_MODEL_LABEL,
        mode="lines",
        line={"color": _REFERENCE, "width": 2, "dash": "dash"},
        hovertemplate="%{y:.3f}",
    )
    marks = [(served, "Umbral servido", "top left")]
    if chosen != served:
        marks.append((chosen, "Umbral elegido", "top right"))
    for value, text, position in marks:
        figure.add_vline(
            x=value,
            line={"color": _INK, "width": 1, "dash": "dot"},
            annotation={"text": text, "font": {"color": _INK}},
            annotation_position=position,
        )
    figure.update_layout(
        xaxis={"title": "Umbral", "range": [0, 1], "gridcolor": _GRID, "tickformat": ".1f"},
        yaxis={"title": "Coste por reserva (supuesto)", "gridcolor": _GRID, "rangemode": "tozero"},
        legend={"orientation": "h", "y": 1.14, "x": 0},
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": _INK},
        margin={"t": 56, "r": 16, "l": 16, "b": 40},
        height=420,
    )
    return figure


manifest = results.load_manifest()
predictions = results.load_predictions()
served_threshold = float(manifest["decision"]["threshold"])
default_scenario = evaluator.DEFAULT_COST_SCENARIO
served = results.outcome_at(predictions, served_threshold, default_scenario)
no_model = results.cost_marking_all(predictions, default_scenario)
steps = results.probability_steps(predictions)
next_point = results.outcome_at(predictions, _NEXT_POINT, default_scenario)
described_low, described_high = results.cost_range(
    predictions, _DESCRIBED_LOW, _DESCRIBED_HIGH, default_scenario
)

st.title("Umbral y coste")
layout.costs_are_assumed()

st.header("Qué hace el umbral servido")
st.markdown(
    f"El umbral {served.threshold:.2f} sale de la fórmula `C_FP / (C_FP + C_FN)` con el supuesto "
    "de que una falsa alarma cuesta 1 y una cancelación perdida, 3. El coste se fijó antes de ver "
    "2017. Esto es lo que supuso en el test:"
)
flagged_column, recall_column, precision_column, cost_column = st.columns(4)
flagged_column.metric("Reservas marcadas", f"{served.flagged_share * 100:.1f} %")
recall_column.metric("Recall", layout.format_number(served.recall, 3))
precision_column.metric("Precisión", layout.format_number(served.precision, 3))
cost_column.metric("Coste por reserva", layout.format_number(served.cost, 3))
st.markdown(
    f"Punto de comparación, **{_NO_MODEL_LABEL}**: {layout.format_number(no_model)} por "
    "reserva. El modelo mejora esa cifra sin marcarlas todas, pero sigue marcando muchas: "
    f"{layout.format_count(served.false_alarms)} falsas alarmas."
)

st.subheader("El umbral cae justo antes de un escalón")
st.markdown(
    f"{layout.steps_explanation(steps)} Con {served.threshold:.2f} se marca el "
    f"{served.flagged_share * 100:.1f} % de las reservas, y con {_NEXT_POINT:.2f}, el "
    f"{next_point.flagged_share * 100:.1f} %: un escalón grande queda justo por encima del umbral."
)
st.warning(
    f"Con costes 1:3, entre {_DESCRIBED_LOW:.2f} y {_DESCRIBED_HIGH:.2f} el coste habría sido de "
    f"{described_low:.3f} a {described_high:.3f}, frente a {served.cost:.3f}. **Elegir ahora un "
    "umbral mirando 2017 no sería una evaluación limpia**: ese año ya no serviría para medirlo. "
    f"Por eso el umbral servido sigue siendo {served.threshold:.2f}.",
    icon="⚠️",
)

st.header("Escenarios de coste")
business = results.load_table("business", "value_by_cost_scenario.csv")
scenario_table = layout.number_table(
    business.assign(scenario=_scenario_name(business)), _SCENARIOS, counts=_SCENARIO_COUNTS
)
st.dataframe(scenario_table, hide_index=True)

st.subheader("Contraste en 2016T4, con datos de desarrollo")
st.markdown(
    "Es la comparación que no contamina el test. El mejor umbral observado cambia de un periodo a "
    "otro, y por eso no se usa para decidir."
)
contrast = pd.DataFrame(manifest["decision"]["contrast_on_2016q4"])
st.dataframe(
    layout.number_table(contrast.assign(scenario=_scenario_name(contrast)), _CONTRAST),
    hide_index=True,
)

st.header("Prueba otros costes")
st.caption(
    "Solo describe lo que habría pasado en 2017 con las probabilidades guardadas. No cambia el "
    "modelo ni el umbral que sirve la API."
)
fp_column, fn_column, threshold_column = st.columns(3)
false_positive = fp_column.slider(
    "Coste de una falsa alarma (supuesto)", 1.0, 10.0, 1.0, 0.5, key="false_positive_cost"
)
false_negative = fn_column.slider(
    "Coste de una cancelación perdida (supuesto)", 1.0, 10.0, 3.0, 0.5, key="false_negative_cost"
)
chosen_threshold = threshold_column.slider(
    "Umbral", 0.01, 0.99, served_threshold, 0.01, key="chosen_threshold"
)

scenario = evaluator.CostScenario(false_positive, false_negative)
chosen = results.outcome_at(predictions, chosen_threshold, scenario)
curve = results.cost_curve(predictions, scenario)
mark_all = results.cost_marking_all(predictions, scenario)
mark_none = results.cost_marking_none(predictions, scenario)

theory_column, cost_chosen_column, flagged_chosen_column = st.columns(3)
theory_column.metric("Umbral teórico", f"{evaluator.optimal_threshold(scenario):.3f}")
cost_chosen_column.metric("Coste por reserva con estos controles", f"{chosen.cost:.3f}")
flagged_chosen_column.metric("Reservas marcadas", f"{chosen.flagged_share * 100:.1f} %")
st.markdown(f"**{_NO_MODEL_LABEL}**: {mark_all:.3f} por reserva.")
if mark_none < mark_all:
    st.caption(
        f"Con estos costes sale más barato no marcar ninguna reserva ({mark_none:.3f}) que "
        "marcarlas todas."
    )

st.plotly_chart(_cost_chart(curve, mark_all, served_threshold, chosen_threshold), width="stretch")
with st.expander("Ver la curva como tabla"):
    st.dataframe(
        layout.number_table(
            curve,
            {
                "threshold": "Umbral",
                "cost": "Coste por reserva",
                "flagged_share": "Reservas marcadas",
            },
        ),
        hide_index=True,
    )

layout.image_card(
    "matrices_confusion_campeon.png",
    title="Matrices de confusión del campeón",
    description="Cada escenario de coste, con su umbral teórico.",
)
