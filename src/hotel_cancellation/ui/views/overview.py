"""Overview: what the model does and how far its figures can be trusted."""

import streamlit as st

from hotel_cancellation import evaluator
from hotel_cancellation.ui import layout, results

manifest = results.load_manifest()
subsets = results.champion_subsets()
predictions = results.load_predictions()
served = results.outcome_at(
    predictions, manifest["decision"]["threshold"], evaluator.DEFAULT_COST_SCENARIO
)

st.title("Cancelación de reservas de hotel")
st.markdown(
    "Este proyecto estima, **en el momento de confirmar una reserva**, la probabilidad de que se "
    "cancele. El modelo es un **Random Forest calibrado con isotónica**, entrenado con llegadas "
    "anteriores a 2016-10-01 y calibrado en el último trimestre de 2016. Se evaluó una sola vez "
    f"con las {layout.format_count(len(predictions))} llegadas de enero a agosto de 2017, sin "
    "reentrenar."
)

st.header("Resultado en 2017")
layout.roc_auc_by_subset(subsets)
layout.deposit_warning(subsets)

st.markdown(
    "- **Baseline:** un `Dummy` da un ROC-AUC de 0.5000.\n"
    f"- **Validación en 2016:** {manifest['validation']['mean']['roc_auc']:.4f} ± "
    f"{manifest['validation']['std']['roc_auc']:.4f} entre ventanas.\n"
    "- **El test no es estrictamente ciego:** una auditoría previa ya había visto 2017, y el "
    "informe declara qué decisiones salen de ella."
)

st.header("Decisión servida")
threshold_column, flagged_column, recall_column, precision_column = st.columns(4)
threshold_column.metric("Umbral", f"{served.threshold:.2f}")
flagged_column.metric("Reservas marcadas", f"{served.flagged_share * 100:.1f} %")
recall_column.metric("Recall", layout.format_number(served.recall, 3))
precision_column.metric("Precisión", layout.format_number(served.precision, 3))
st.markdown(
    "El umbral sale de suponer que perder una cancelación cuesta tres veces más que una falsa "
    "alarma. Marca muchas reservas, y más de la mitad de las marcadas no se cancela. La página "
    "**Umbral y coste** lo detalla."
)
layout.costs_are_assumed()

layout.image_card(
    "inflacion.png",
    title="De una cifra inflada a la cifra oficial",
    description=(
        "Una partición aleatoria da 0.952: casi un tercio de las filas de test ya estaba en train, "
        "por los duplicados. Evaluar en el futuro baja a 0.868, y quitar `country` y "
        "`booking_changes`, que se conocen tarde, a 0.818. Tampoco es el rendimiento al confirmar: "
        "con las reservas confirmadas desde 2017-01-01, el campeón da 0.7917."
    ),
)
