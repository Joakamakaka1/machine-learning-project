"""Presentation helpers shared by the pages.

The warnings that keep the interface honest live here, so that every page says the same thing.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from hotel_cancellation import config
from hotel_cancellation.ui import client, results

# Chart and text columns: with (3, 2) the chart takes about 60 % of the width.
IMAGE_COLUMN_RATIO: tuple[int, int] = (3, 2)

MODEL_LABELS: Mapping[str, str] = {
    "random_forest": "Random Forest (campeón)",
    "xgboost": "XGBoost",
    "mlp": "MLP",
    "decision_tree": "Árbol de decisión",
    "logistic_regression": "Regresión logística",
    "dummy": "Dummy (baseline)",
    "special_requests_scenario": "Escenario con peticiones especiales (informativo)",
}
SUBSET_LABELS: Mapping[str, str] = {
    "all": "Todas",
    "unique": "Únicas",
    "no_deposit": "No Deposit",
    "non_refund": "Non Refund",
}
_NOT_DEFINED = "n/d"
_HEALTH_KEY = "api_health"


def image_card(name: str, *, title: str, description: str) -> None:
    """Show a saved figure next to its description, or say that it cannot be found."""
    path: Path = results.figure_path(name)
    st.subheader(title)
    if not path.exists():
        st.warning(f"No se encuentra la figura `{name}`.")
        return
    image_column, text_column = st.columns(IMAGE_COLUMN_RATIO)
    image_column.image(str(path), width="stretch")
    text_column.markdown(description)


def format_number(value: Any, decimals: int = 4) -> str:
    """Write a number, or `n/d` for a metric that is not defined."""
    return _NOT_DEFINED if pd.isna(value) else f"{float(value):.{decimals}f}"


def format_count(value: Any) -> str:
    """Write a whole number with a thin separator, as the report does."""
    return f"{int(value):,}".replace(",", " ")


def number_table(
    frame: pd.DataFrame, columns: Mapping[str, str], *, counts: tuple[str, ...] = ()
) -> pd.DataFrame:
    """Build a table for reading: Spanish headers, fixed decimals and `n/d`.

    Args:
        frame: Saved rows.
        columns: Column of `frame` and its header, in the order to show.
        counts: Columns that hold whole numbers.

    Returns:
        A frame of text.
    """
    shown = {}
    for column, header in columns.items():
        values = frame[column]
        if column in counts:
            shown[header] = values.map(format_count)
        elif pd.api.types.is_numeric_dtype(values):
            shown[header] = values.map(format_number)
        else:
            shown[header] = values.astype("str")
    return pd.DataFrame(shown)


def label_models(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace the names of the models by their labels."""
    return frame.assign(model=frame["model"].map(lambda name: MODEL_LABELS.get(name, name)))


def roc_auc_by_subset(subsets: pd.DataFrame) -> None:
    """Show the ROC-AUC on the four subsets. The global figure never goes alone."""
    low, high = results.champion_interval()
    by_subset = subsets.set_index("subset")["roc_auc"]
    columns = st.columns(len(SUBSET_LABELS))
    for column, (subset, label) in zip(columns, SUBSET_LABELS.items(), strict=True):
        column.metric(f"ROC-AUC, {label}", format_number(by_subset[subset]))
    st.caption(
        f"Intervalo del 95 % del ROC-AUC con todas las reservas: "
        f"[{format_number(low)}, {format_number(high)}], con bootstrap por bloques de hotel y "
        "fecha. No cubre la deriva temporal."
    )


def deposit_warning(subsets: pd.DataFrame) -> None:
    """Warn that `deposit_type` is not causal and say where the real difficulty is."""
    by_subset = subsets.set_index("subset")["roc_auc"]
    st.warning(
        f"**`deposit_type` no es causal.** `Non Refund` da un ROC-AUC de "
        f"{format_number(by_subset['non_refund'])} y es casi trivial: reúne tandas de reservas "
        "casi idénticas que se cancelan juntas. Pedir un depósito no reembolsable no provoca la "
        f"cancelación. El caso difícil es `No Deposit`, con "
        f"{format_number(by_subset['no_deposit'])}.",
        icon="⚠️",
    )


def costs_are_assumed() -> None:
    """Remind the reader that the costs are assumptions."""
    st.info(
        "Los costes son un **supuesto**, no datos de los hoteles. Cambian qué umbral conviene, "
        "pero no la calidad del modelo."
    )


def steps_explanation(steps: results.ProbabilitySteps) -> str:
    """Explain why the probabilities jump in steps."""
    return (
        f"La probabilidad sale de un calibrador isotónico, que trabaja en escalones: en 2017 solo "
        f"da {steps.distinct_values} valores distintos. Un {steps.share_zero * 100:.2f} % de las "
        f"reservas vale exactamente 0, y un {steps.share_one * 100:.2f} %, exactamente 1. No son "
        "certezas, y por eso se muestran como «<1 %» y «>99 %»."
    )


def check_api() -> None:
    """Ask for the state of the API, once per run: a name that does not resolve takes seconds."""
    try:
        st.session_state[_HEALTH_KEY] = client.default_client().health()
    except client.ApiUnavailableError:
        st.session_state[_HEALTH_KEY] = None


def api_status(container: Any, *, verbose: bool) -> None:
    """Draw the state that `check_api` found.

    Args:
        container: Where to draw it.
        verbose: With instructions to start it, for the prediction page.
    """
    address = config.get_settings().api_url
    health = st.session_state[_HEALTH_KEY]
    if health is None:
        if verbose:
            container.error(
                f"**La API no está disponible** en `{address}`. Puedes consultar el resto de "
                "páginas, pero no predecir. Arráncala con `poe up` o con `poe api`.",
                icon="🚫",
            )
        else:
            container.error("API no disponible", icon="🚫")
        return
    if verbose:
        container.success(f"API conectada. Modelo: `{health['model_version']}`.", icon="✅")
    else:
        container.success("API conectada", icon="✅")
