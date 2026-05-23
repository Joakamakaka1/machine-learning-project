"""
- Archivo encargado de:
- API consuma el modelo seleccionado
"""

import joblib
import pandas as pd

from src.config import MODELS_DIR


# Clase para cargar el modelo y hacer predicciones
class BookingPredictor:
    def __init__(self):
        # Cargar el mejor modelo guardado
        self.model_path = MODELS_DIR / "best_model.pkl"
        if not self.model_path.exists():
            raise FileNotFoundError(
                "No se encuentra el archivo best_model.pkl. Corre primero el entrenamiento."
            )

        self.model = joblib.load(self.model_path)

    # Función interna para limpiar y preparar los datos antes de la predicción
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        # Lista de columnas numéricas conocidas en el dataset
        num_cols = [
            "lead_time",
            "arrival_date_year",
            "arrival_date_week_number",
            "arrival_date_day_of_month",
            "stays_in_weekend_nights",
            "stays_in_week_nights",
            "adults",
            "children",
            "babies",
            "is_repeated_guest",
            "previous_cancellations",
            "previous_bookings_not_canceled",
            "booking_changes",
            "agent",
            "days_in_waiting_list",
            "adr",
            "required_car_parking_spaces",
            "total_of_special_requests",
        ]

        for col in num_cols:
            if col in df.columns:
                # Convertir a string, limpiar valores nulos textuales y coaccionar a numérico
                df[col] = pd.to_numeric(
                    df[col]
                    .astype(str)
                    .replace(["NULL", "None", "nan", "NaN", "nan.0", "", " "], None),
                    errors="coerce",
                )
        return df

    # Funcion encargada de:
    # - Predecir la cancelación de una reserva
    # - Recibe un DataFrame con las características y devuelve predicciones (0 o 1)
    def predict(self, data: pd.DataFrame):
        cleaned_data = self._clean_data(data)
        predictions = self.model.predict(cleaned_data)
        return predictions.tolist()

    # Funcion encargada de:
    # - Predecir la probabilidad de cancelación de una reserva
    # - Recibe un DataFrame con las características y devuelve probabilidades (0.0 a 1.0)
    def predict_proba(self, data: pd.DataFrame):
        cleaned_data = self._clean_data(data)
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(cleaned_data)[:, 1]
        else:
            # Se aproxima si el modelo no tiene predict_proba (eg. SVM sin probabilidad)
            probabilities = self.model.predict(cleaned_data)
        return probabilities.tolist()
