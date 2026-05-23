import pandas as pd

from src.predictor import BookingPredictor


class PredictionService:
    def __init__(self):
        # El predictor de src se encarga de buscar models/best_model.pkl
        self.predictor = BookingPredictor()

    def predict_booking(self, booking_data: dict) -> tuple[int, float]:
        """Realiza la predicción a partir de los datos recibidos en la API."""
        # Convertimos los datos a un DataFrame de una fila
        df_input = pd.DataFrame([booking_data])

        # Inferencia
        pred = self.predictor.predict(df_input)[0]
        prob = self.predictor.predict_proba(df_input)[0]

        return int(pred), float(prob)


# Instanciamos el servicio de predicción globalmente para reusar en endpoints
try:
    prediction_service = PredictionService()
except Exception as e:
    prediction_service = None
    print(f"Advertencia: No se pudo inicializar el servicio de predicción: {e}")
