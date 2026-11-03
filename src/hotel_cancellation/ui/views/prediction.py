"""Prediction: a form that asks the API about one booking."""

from collections.abc import Mapping
from typing import Any

import streamlit as st

from hotel_cancellation.data import schema
from hotel_cancellation.ui import booking, client, layout, results

_COLUMNS = 2
_FORM = "booking_form"
_YES_NO = {0: "No", 1: "Sí"}
_LABELS: Mapping[str, str] = {
    "canceled": "Se marca",
    "not_canceled": "No se marca",
}
_BANDS: Mapping[str, str] = {"low": "Bajo", "medium": "Medio", "high": "Alto"}
_BAND_MEANINGS: Mapping[str, str] = {
    "low": "por debajo del umbral",
    "medium": "marcada, aunque es más probable que no se cancele",
    "high": "es más probable que se cancele que lo contrario",
}
# Machine-readable errors of the API, in the language of the interface.
_REJECTIONS: Mapping[str, str] = {
    "no_guests": "La reserva necesita al menos un huésped.",
    "no_nights": "La reserva necesita al menos una noche.",
}


def _draw_field(target: Any, field: booking.FormField) -> Any:
    key = f"field_{field.name}"
    help_text = field.help_text or None
    default = booking.default_of(field)
    if field.kind is booking.Kind.CATEGORY:
        options = booking.options_of(field)
        return target.selectbox(
            field.label, options, index=options.index(default), help=help_text, key=key
        )
    if field.kind is booking.Kind.FLAG:
        return target.selectbox(
            field.label,
            list(_YES_NO),
            index=default,
            format_func=_YES_NO.get,
            help=help_text,
            key=key,
        )
    if field.kind is booking.Kind.AGENT:
        return target.text_input(field.label, value=default, help=help_text, key=key)
    low, high = booking.limits_of(field)
    if field.kind is booking.Kind.MONEY:
        return target.number_input(
            field.label,
            min_value=low,
            max_value=high,
            value=float(default),
            step=1.0,
            help=help_text,
            key=key,
        )
    return target.number_input(
        field.label,
        min_value=int(low),
        max_value=int(high),
        value=int(default),
        step=1,
        help=help_text,
        key=key,
    )


def _draw_form() -> dict[str, Any] | None:
    values: dict[str, Any] = {}
    with st.form(_FORM):
        st.caption("El formulario se abre con una reserva de ejemplo. Cámbiala y pulsa Predecir.")
        for section, fields in booking.FORM_SECTIONS.items():
            st.markdown(f"**{section}**")
            columns = st.columns(_COLUMNS)
            for position, field in enumerate(fields):
                values[field.name] = _draw_field(columns[position % _COLUMNS], field)
        submitted = st.form_submit_button("Predecir", type="primary", width="stretch")
    return booking.build_payload(values) if submitted else None


def _show_rejection(errors: list[dict[str, Any]]) -> None:
    labels = {field.name: field.label for field in booking.all_fields()}
    lines = []
    for error in errors:
        known = _REJECTIONS.get(error.get("type", ""))
        if known:
            lines.append(known)
            continue
        location = error.get("loc", [])
        name = location[-1] if location else ""
        lines.append(f"{labels.get(name, name)}: el valor no es válido.")
    st.error("La API ha rechazado la reserva.\n\n- " + "\n- ".join(lines), icon="🚫")


def _show_result(result: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    probability = float(result["probability"])
    band = result["risk_band"]
    st.subheader("Resultado")
    probability_column, decision_column, band_column = st.columns(3)
    probability_column.metric(
        "Probabilidad de cancelación", results.format_probability(probability)
    )
    decision_column.metric(
        "Decisión", _LABELS[result["label"]], help="Se marca desde el umbral: posible cancelación."
    )
    band_column.metric("Riesgo", _BANDS[band], help=_BAND_MEANINGS[band])
    st.caption(
        f"Umbral {float(result['threshold']):.2f}. Riesgo {_BANDS[band].lower()}: "
        f"{_BAND_MEANINGS[band]}. Modelo `{result['model_version']}`."
    )
    st.info(layout.steps_explanation(results.probability_steps(results.load_predictions())))
    if result["label"] == "canceled":
        st.markdown(
            "**Marcar no significa que se vaya a cancelar.** Con este umbral, en 2017, algo menos "
            "de la mitad de las reservas marcadas se canceló."
        )
    if payload["deposit_type"] == "Non Refund":
        layout.deposit_warning(results.champion_subsets())
    st.caption(
        "No hay explicación por reserva: el modelo servido es un Random Forest, y la página "
        "Interpretabilidad resume qué pesa en general."
    )
    with st.expander("Ver la respuesta de la API"):
        st.json(dict(result))


st.title("Predicción")
st.markdown(
    f"Rellena las {len(schema.INPUT_FEATURES)} variables con los valores **al confirmar la "
    "reserva**. La API estima la probabilidad de que se cancele."
)
layout.api_status(st, verbose=True)

booking_payload = _draw_form()
if booking_payload is not None:
    try:
        with st.spinner("Consultando el modelo..."):
            answer = client.default_client().predict(booking_payload)
    except client.BookingRejectedError as rejection:
        _show_rejection(rejection.errors)
    except client.ApiUnavailableError:
        st.error("No se puede predecir: la API no responde o ha fallado.", icon="🚫")
    else:
        _show_result(answer, booking_payload)
