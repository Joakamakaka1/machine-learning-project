"""Data: where the bookings come from, how they are cleaned and which variables are out."""

from collections.abc import Mapping

import pandas as pd
import streamlit as st

from hotel_cancellation.data import schema
from hotel_cancellation.ui import booking, layout, results

# Spanish wording of `schema.FORBIDDEN_FEATURES`. A test checks that both list the same columns.
_EXCLUDED: Mapping[str, tuple[str, str]] = {
    "reservation_status": ("Al terminar la reserva", "Repite el objetivo"),
    "reservation_status_date": ("Al terminar la reserva", "Fecha del resultado"),
    "country": ("Se corrige en el check-in", "Fuga documentada por los autores del dataset"),
    "required_car_parking_spaces": ("En el check-in o después", "No se conoce al confirmar"),
    "assigned_room_type": ("En el check-in o después", "No se conoce al confirmar"),
    "booking_changes": ("Durante la vida de la reserva", "Cuenta cambios por definición"),
    "total_of_special_requests": (
        "Durante la vida de la reserva",
        "Se acumula según el diagnóstico. Se publica un escenario informativo con ella",
    ),
    "arrival_date_year": ("Al reservar", "Calendario: no generaliza y añade varianza"),
    "arrival_date_month": ("Al reservar", "Calendario: no generaliza y añade varianza"),
    "arrival_date_week_number": ("Al reservar", "Calendario: no generaliza y añade varianza"),
    "arrival_date_day_of_month": ("Al reservar", "Calendario: no generaliza y añade varianza"),
    "company": ("Al reservar", "Sin efecto en validación: el modelo más simple la deja fuera"),
}
_CLEANING = {
    "Sin huéspedes": "180",
    "Sin noches": "715",
    "`adr` negativo": "1",
    "`adr` de 5400 o más": "1",
    "Filas eliminadas (una fila puede cumplir más de un motivo)": "827",
    "Filas que quedan, de 119 390": "118 563",
}
_PARTITIONS = {
    "Parte": ["Desarrollo", "Test"],
    "Llegadas": ["2015-07-01 a 2016-12-31", "2017-01-01 a 2017-08-31"],
    "Uso": ["Toda decisión", "Una sola ejecución, al final"],
}

manifest = results.load_manifest()
test_rows = len(results.load_predictions())
labels = {field.name: field.label for field in booking.all_fields()}

st.title("Datos")
st.markdown(
    "Reservas de dos hoteles portugueses, con llegadas entre el 1 de julio de 2015 y el 31 de "
    "agosto de 2017.\n\n"
    "**Fuente:** Antonio, N., de Almeida, A. y Nunes, L. (2019). *Hotel booking demand datasets*. "
    "Data in Brief, 22, 41-49. <https://doi.org/10.1016/j.dib.2018.11.126>. Licencia CC BY 4.0.\n\n"
    "El archivo guarda **el valor final de cada reserva, no su historial**. Por eso importa cuándo "
    "se conoce cada variable."
)

st.header("Limpieza y particiones")
cleaning_column, partition_column = st.columns(2)
cleaning_column.dataframe(
    pd.DataFrame({"Motivo": list(_CLEANING), "Filas": list(_CLEANING.values())}),
    hide_index=True,
)
partition_column.dataframe(
    pd.DataFrame(
        {
            **_PARTITIONS,
            "Filas limpias": [
                layout.format_count(manifest["development_rows"]),
                layout.format_count(test_rows),
            ],
        }
    ),
    hide_index=True,
)

st.header(f"Variables de entrada ({len(schema.INPUT_FEATURES)})")
st.markdown(
    "Se usan como si se conocieran al confirmar la reserva, pero no consta que sean los valores de "
    "ese momento: el artículo del dataset los referencia al día anterior a la llegada."
)
st.dataframe(
    pd.DataFrame(
        {
            "Variable": list(schema.INPUT_FEATURES),
            "Significado": [labels[column] for column in schema.INPUT_FEATURES],
        }
    ),
    hide_index=True,
)

st.header(f"Variables que quedan fuera ({len(schema.FORBIDDEN_FEATURES)})")
st.dataframe(
    pd.DataFrame(
        {
            "Variable": list(_EXCLUDED),
            "Cuándo se conoce": [known for known, _ in _EXCLUDED.values()],
            "Motivo": [reason for _, reason in _EXCLUDED.values()],
        }
    ),
    hide_index=True,
)

st.header("Diagnóstico de acumulación")
st.markdown(
    "Una reserva cancelada deja de cambiar al cancelarse. Si una variable se acumula, crece con la "
    "parte de vida transcurrida hasta ese momento. `booking_changes`, que cuenta cambios por "
    "definición, hace de control. Se usan solo llegadas anteriores a 2017."
)
cohorts = results.load_table("experiments", "feature_availability_cohorts.csv")
compared = cohorts[
    cohorts["variable"].isin(["total_of_special_requests", "booking_changes"])
    & (cohorts["cohort"] == "online_ta")
]
st.dataframe(
    layout.number_table(
        compared,
        {
            "variable": "Variable",
            "rows": "Reservas canceladas en Online TA",
            "rho": "Correlación de Spearman con la vida transcurrida",
            "nonzero_0_5": "Con valor, 0-5 % de vida",
            "nonzero_75_100": "Con valor, 75-100 % de vida",
        },
        counts=("rows",),
    ),
    hide_index=True,
)
st.caption(
    "El diagnóstico es observacional: quien cancela enseguida puede ser un cliente distinto. Por "
    "eso se publican los dos escenarios, con y sin `total_of_special_requests`."
)
