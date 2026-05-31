"""
- Archivo encargado de:
- API consuma el modelo seleccionado
"""

import joblib
import pandas as pd

from src.config import MODELS_DIR


class BookingPredictor:
    """Carga el mejor modelo entrenado y expone métodos para realizar predicciones."""
    def __init__(self):
        """Carga el mejor modelo guardado desde disco."""
        self.model_path = MODELS_DIR / "best_model.pkl"
        if not self.model_path.exists():
            raise FileNotFoundError(
                "No se encuentra el archivo best_model.pkl. Corre primero el entrenamiento."
            )

        self.model = joblib.load(self.model_path)

    def predict(self, data: pd.DataFrame):
        """Predice la cancelación de una reserva.

        El Pipeline guardado ya incluye FeatureEngineer + preprocesado completo,
        por lo que no es necesario ningún paso adicional antes de llamar a predict.

        Args:
            data: DataFrame con las características de la reserva.

        Returns:
            list: Predicciones binarias (0 = no cancela, 1 = cancela).
        """
        predictions = self.model.predict(data)
        return predictions.tolist()

    def predict_proba(self, data: pd.DataFrame):
        """Predice la probabilidad de cancelación de una reserva.

        Args:
            data: DataFrame con las características de la reserva.

        Returns:
            list: Probabilidades de cancelación entre 0.0 y 1.0.
        """
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(data)[:, 1]
        else:
            # Se aproxima si el modelo no tiene predict_proba (eg. SVM sin probabilidad)
            probabilities = self.model.predict(data)
        return probabilities.tolist()
