import pandas as pd
import streamlit as st

from src.predictor import BookingPredictor

st.set_page_config(page_title="Predictor de Cancelaciones de Hotel", layout="wide")

st.title("Predictor de Cancelaciones de Hotel")
st.markdown(
    "Esta aplicación predice la probabilidad de que un cliente cancele su "
    "reserva de hotel basándose en el mejor modelo de ML entrenado."
)


# Cargar el predictor
@st.cache_resource
def load_predictor():
    return BookingPredictor()


try:
    predictor = load_predictor()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error al cargar el modelo. ¿Ya ejecutaste el entrenamiento? Detalle: {e}")

if model_loaded:
    st.sidebar.header("Datos de la Reserva")

    # Inputs en la barra lateral organizados
    hotel = st.sidebar.selectbox("Tipo de Hotel", ["Resort Hotel", "City Hotel"])
    lead_time = st.sidebar.slider("Anticipación (días)", 0, 365, 30)
    adr = st.sidebar.number_input(
        "Precio Diario Promedio (ADR - €)", min_value=0.0, max_value=1000.0, value=100.0
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Fechas y Duración")
        arrival_date_year = st.selectbox("Año de Llegada", [2015, 2016, 2017, 2018])
        arrival_date_month = st.selectbox(
            "Mes de Llegada",
            [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ],
        )
        arrival_date_day_of_month = st.number_input("Día de Llegada", 1, 31, 15)
        arrival_date_week_number = st.number_input("Número de Semana del Año", 1, 53, 27)
        stays_in_weekend_nights = st.number_input("Noches de Fin de Semana", 0, 10, 2)
        stays_in_week_nights = st.number_input("Noches de Semana", 0, 20, 5)

    with col2:
        st.subheader("Huéspedes e Historial")
        adults = st.number_input("Adultos", 1, 10, 2)
        children = st.number_input("Niños", 0, 10, 0)
        babies = st.number_input("Bebés", 0, 5, 0)
        meal = st.selectbox("Régimen de Comida", ["BB", "HB", "FB", "SC", "Undefined"])
        country = st.text_input("País de Origen (Código ISO, ej: PRT, ESP, GBR)", "ESP")
        market_segment = st.selectbox(
            "Segmento de Mercado",
            [
                "Direct",
                "Online TA",
                "Offline TA/TO",
                "Groups",
                "Corporate",
                "Complementary",
                "Aviation",
            ],
        )
        distribution_channel = st.selectbox(
            "Canal de Distribución", ["Direct", "TA/TO", "Corporate", "Corporate", "GDS"]
        )

    st.sidebar.subheader("Políticas e Historial")
    deposit_type = st.sidebar.selectbox(
        "Tipo de Depósito", ["No Deposit", "Non Refund", "Refundable"]
    )
    is_repeated_guest = st.sidebar.checkbox("¿Es cliente repetidor?", value=False)
    previous_cancellations = st.sidebar.number_input("Cancelaciones previas", 0, 20, 0)
    previous_bookings_not_canceled = st.sidebar.number_input(
        "Reservas previas no canceladas", 0, 50, 0
    )
    booking_changes = st.sidebar.number_input("Modificaciones hechas a la reserva", 0, 10, 0)
    required_car_parking_spaces = st.sidebar.number_input("Plazas de parking requeridas", 0, 5, 0)
    total_of_special_requests = st.sidebar.number_input("Peticiones especiales totales", 0, 10, 1)
    days_in_waiting_list = st.sidebar.number_input("Días en lista de espera", 0, 365, 0)
    customer_type = st.sidebar.selectbox(
        "Tipo de Cliente", ["Transient", "Transient-Party", "Contract", "Group"]
    )
    reserved_room_type = st.sidebar.text_input("Tipo de Habitación Reservada (ej: A, D, C)", "A")
    assigned_room_type = st.sidebar.text_input("Tipo de Habitación Asignada (ej: A, D, C)", "A")
    agent = st.sidebar.text_input("ID Agente (o NULL)", "NULL")

    # Botón para predecir
    if st.button("Predecir Probabilidad de Cancelación", type="primary"):
        # Construir el registro de entrada
        input_data = pd.DataFrame(
            [
                {
                    "hotel": hotel,
                    "lead_time": lead_time,
                    "arrival_date_year": arrival_date_year,
                    "arrival_date_month": arrival_date_month,
                    "arrival_date_week_number": arrival_date_week_number,
                    "arrival_date_day_of_month": arrival_date_day_of_month,
                    "stays_in_weekend_nights": stays_in_weekend_nights,
                    "stays_in_week_nights": stays_in_week_nights,
                    "adults": adults,
                    "children": float(children),
                    "babies": babies,
                    "meal": meal,
                    "country": country,
                    "market_segment": market_segment,
                    "distribution_channel": distribution_channel,
                    "is_repeated_guest": 1 if is_repeated_guest else 0,
                    "previous_cancellations": previous_cancellations,
                    "previous_bookings_not_canceled": previous_bookings_not_canceled,
                    "reserved_room_type": reserved_room_type,
                    "assigned_room_type": assigned_room_type,
                    "booking_changes": booking_changes,
                    "deposit_type": deposit_type,
                    "agent": agent,
                    "days_in_waiting_list": days_in_waiting_list,
                    "customer_type": customer_type,
                    "adr": adr,
                    "required_car_parking_spaces": required_car_parking_spaces,
                    "total_of_special_requests": total_of_special_requests,
                }
            ]
        )

        # Inferencia
        pred = predictor.predict(input_data)[0]
        prob = predictor.predict_proba(input_data)[0]

        st.subheader("Resultado de la Predicción")

        # Mostrar resultado con colores atractivos
        if pred == 1:
            st.error(f"**ALTA PROBABILIDAD DE CANCELACIÓN:** {prob:.2%}")
            st.write("El modelo predice que esta reserva **será cancelada**.")
        else:
            st.success(f"**BAJA PROBABILIDAD DE CANCELACIÓN:** {prob:.2%}")
            st.write("El modelo predice que el cliente **realizará el check-in** con éxito.")
