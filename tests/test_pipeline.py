import pandas as pd
from src.predictor import BookingPredictor


def test_clean_data_coercion(monkeypatch):
    """Prueba que el método de limpieza de datos convierta strings nulos en NaN y coaccione números."""
    # Mockear __init__ para evitar cargar el modelo real y no requerir dependencias del disco
    monkeypatch.setattr(BookingPredictor, "__init__", lambda self: None)

    predictor = BookingPredictor()

    # DataFrame con strings sucios en columnas numéricas y una columna categórica
    test_df = pd.DataFrame(
        [
            {
                "lead_time": "120",
                "agent": "NULL",
                "adr": "None",
                "hotel": "Resort Hotel",
                "total_of_special_requests": " ",
            }
        ]
    )

    cleaned_df = predictor._clean_data(test_df)

    # Validar la coacción correcta a numérico y nulo
    assert cleaned_df.loc[0, "lead_time"] == 120.0
    assert pd.isna(cleaned_df.loc[0, "agent"])
    assert pd.isna(cleaned_df.loc[0, "adr"])
    assert pd.isna(cleaned_df.loc[0, "total_of_special_requests"])
    assert cleaned_df.loc[0, "hotel"] == "Resort Hotel"  # Columna de texto no se altera
