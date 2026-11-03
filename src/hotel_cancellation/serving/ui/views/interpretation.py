"""Interpretation: what weighs in the model in general, which does not say what causes what."""

import streamlit as st

from hotel_cancellation.serving.ui import layout, results

_IMPORTANCE = {
    "variable": "Variable",
    "forest_importance": "feature_importances_",
    "permutation_mean": "Permutación (caída de ROC-AUC)",
    "permutation_std": "± desviación",
    "shap_mean_abs": "|SHAP| medio",
}

st.title("Interpretabilidad")
st.markdown(
    "**Sirve para interpretar, no decide nada y no indica causas.** Se calculó sobre 2017 con el "
    "modelo ya congelado, y los valores se agregan por variable original."
)
layout.deposit_warning(results.champion_subsets())

layout.image_card(
    "importancia_variables.png",
    title="Importancia de las variables",
    description=(
        "- **`feature_importances_`:** sale del modelo base, y favorece a las variables con muchas "
        "categorías.\n"
        "- **Permutación:** caída del ROC-AUC del campeón al barajar cada variable en 2017.\n"
        "- **SHAP:** 2000 filas de 2017, en la probabilidad sin calibrar del modelo base."
    ),
)
st.dataframe(
    layout.number_table(results.load_table("interpretation", "importance.csv"), _IMPORTANCE),
    hide_index=True,
)
st.info(
    "**No hay explicación por reserva.** Solo se ofrecería con XGBoost o con regresión logística, "
    "y el campeón es un Random Forest. Además, describiría el modelo base y no la probabilidad "
    "calibrada que se sirve."
)
