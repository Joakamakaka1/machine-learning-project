"""
Carga del modelo entrenado para servir predicciones.
"""

import joblib

from src.config import MODELS_DIR


def load_model(model_name="best_model"):
    """Carga un pipeline entrenado desde disco.

    El Pipeline guardado ya incluye FeatureEngineer + preprocesado completo,
    por lo que no es necesario ningún paso adicional antes de llamar a predict.

    Args:
        model_name: Nombre del modelo (sin extensión .pkl).

    Returns:
        Pipeline sklearn entrenado.

    Raises:
        FileNotFoundError: Si el archivo .pkl no existe.
    """
    path = MODELS_DIR / f"{model_name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra {path}. Ejecuta primero el entrenamiento.")
    return joblib.load(path)
