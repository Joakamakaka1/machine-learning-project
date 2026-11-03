"""Evaluation: the saved tables and figures of the single execution on 2017."""

import streamlit as st

from hotel_cancellation.evaluation import stored_evaluation
from hotel_cancellation.serving.ui import layout, results

_MODEL_ORDER = (
    "random_forest",
    "xgboost",
    "mlp",
    "decision_tree",
    "logistic_regression",
    "dummy",
)
_METRICS = {
    "model": "Modelo",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "brier": "Brier",
    "calibration_slope": "Pendiente",
    "accuracy": "Accuracy",
    "precision": "Precisión",
    "recall": "Recall",
    "f1": "F1",
    "cost_1_3": "Coste 1:3 (supuesto)",
}
_SUBSET_METRICS = {
    "subset": "Filas analizadas",
    "rows": "Filas",
    "prevalence": "Prevalencia",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "brier": "Brier",
    "precision": "Precisión",
    "recall": "Recall",
    "f1": "F1",
    "cost_1_3": "Coste 1:3 (supuesto)",
}
_SEGMENT_METRICS = {
    "segment": "Segmento",
    "value": "Valor",
    "rows": "Filas",
    "prevalence": "Prevalencia",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "recall": "Recall",
    "f1": "F1",
    "cost_1_3": "Coste 1:3 (supuesto)",
}
_MONTHLY = {"month": "Mes", "rows": "Filas", "prevalence": "Prevalencia", "roc_auc": "ROC-AUC"}
_PSI = {"feature": "Variable", "psi": "PSI"}
_SCENARIO = {
    "subset": "Filas analizadas",
    "rows": "Filas",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "brier": "Brier",
    "recall": "Recall",
    "f1": "F1",
    "cost_1_3": "Coste 1:3 (supuesto)",
}


def _by_model(threshold_name: str):  # type: ignore[no-untyped-def]
    metrics = results.load_evaluation_table(stored_evaluation.METRICS_FILE)
    chosen = metrics[(metrics["subset"] == "all") & (metrics["threshold_name"] == threshold_name)]
    ordered = chosen.set_index("model").loc[list(_MODEL_ORDER)].reset_index()
    return layout.label_models(ordered)


subsets = results.champion_subsets()
steps = results.probability_steps(results.load_predictions())

st.title("Evaluación")
st.markdown(
    "Cifras guardadas de la **única ejecución** sobre 2017. Esta página no calcula nada sobre el "
    "test: lee lo que se guardó. Toda probabilidad es la calibrada."
)

st.header("El campeón, por subconjunto")
layout.roc_auc_by_subset(subsets)
layout.deposit_warning(subsets)
st.dataframe(
    layout.number_table(
        subsets.assign(subset=subsets["subset"].map(layout.SUBSET_LABELS)),
        _SUBSET_METRICS,
        counts=("rows",),
    ),
    hide_index=True,
)

st.header("Comparación de modelos")
st.markdown(f"Con el umbral servido, {subsets['threshold'].iloc[0]:.2f}, y costes 1:3 supuestos.")
st.dataframe(layout.number_table(_by_model("decision"), _METRICS), hide_index=True)
st.markdown(
    "El campeón **no** es el más barato con su propio umbral. La selección se hizo por ROC-AUC en "
    "validación, como fija el protocolo, y el modelo no se ha cambiado después."
)
st.subheader("Con un umbral fijo de 0.5")
st.caption("Sirve para comparar con otros trabajos. No es el umbral servido.")
st.dataframe(layout.number_table(_by_model("fixed_0.5"), _METRICS), hide_index=True)

layout.image_card(
    "roc.png",
    title="Curvas ROC",
    description="Cada curva enfrenta las cancelaciones detectadas con las falsas alarmas.",
)
layout.image_card(
    "precision_recall.png",
    title="Precisión y recall",
    description="Muestra qué parte de lo marcado acierta al detectar más cancelaciones.",
)
layout.image_card(
    "intervalos_roc_auc.png",
    title="ROC-AUC con su intervalo",
    description="XGBoost no se distingue del campeón: la diferencia pareada incluye el cero.",
)
layout.image_card(
    "matrices_confusion_modelos.png",
    title="Matrices de confusión",
    description="Aciertos y errores de cada modelo con el umbral servido.",
)

st.header("Calibración")
layout.image_card(
    "calibracion.png",
    title="Probabilidad estimada y frecuencia real",
    description=layout.steps_explanation(steps),
)

st.header("Segmentos")
st.dataframe(
    layout.number_table(
        results.load_evaluation_table(stored_evaluation.SEGMENTS_FILE),
        _SEGMENT_METRICS,
        counts=("rows",),
    ),
    hide_index=True,
)

st.header("Deriva temporal")
layout.image_card(
    "roc_auc_por_mes.png",
    title="ROC-AUC por mes",
    description="Baja a lo largo del periodo. El intervalo del bootstrap no cubre esta deriva.",
)
layout.image_card(
    "prevalencia_por_mes.png",
    title="Prevalencia por mes",
    description="La proporción de cancelaciones cambia de un mes a otro.",
)
st.dataframe(
    layout.number_table(
        results.load_evaluation_table(stored_evaluation.MONTHLY_FILE), _MONTHLY, counts=("rows",)
    ),
    hide_index=True,
)
layout.image_card(
    "deriva_psi.png",
    title="Deriva por variable",
    description=(
        "`agent` es la que más deriva. Es una limitación de usar identificadores en un modelo que "
        "no se reentrena."
    ),
)
st.dataframe(
    layout.number_table(results.load_evaluation_table(stored_evaluation.PSI_FILE), _PSI),
    hide_index=True,
)

st.header("Escenario con `total_of_special_requests`")
scenario = results.load_evaluation_table(stored_evaluation.SCENARIO_FILE)
scenario_rows = scenario[scenario["threshold_name"] == stored_evaluation.DECISION_THRESHOLD]
scenario_roc_auc = scenario_rows.set_index("subset")["roc_auc"]["all"]
st.warning(
    f"**Es solo informativo.** Con esta variable, el ROC-AUC sería de "
    f"{layout.format_number(scenario_roc_auc)}, pero el modelo servido no la usa: el diagnóstico "
    "indica que se acumula durante la vida de la reserva, de modo que quizá no se conoce al "
    "confirmarla. No puede cambiar el campeón.",
    icon="⚠️",
)
st.dataframe(
    layout.number_table(
        scenario_rows.assign(subset=scenario_rows["subset"].map(layout.SUBSET_LABELS)),
        _SCENARIO,
        counts=("rows",),
    ),
    hide_index=True,
)
