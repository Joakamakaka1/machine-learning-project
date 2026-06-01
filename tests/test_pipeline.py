import pandas as pd
import pytest

from src.data_loader import FeatureEngineer
from src.predictor import load_model


def test_load_model_missing_file():
    """Verifica que load_model lanza FileNotFoundError si el modelo no existe."""
    with pytest.raises(FileNotFoundError):
        load_model("modelo_inexistente_xyz")


def test_feature_engineer_creates_derived_columns():
    """Verifica que FeatureEngineer crea las columnas derivadas correctamente."""
    df = pd.DataFrame(
        [
            {
                "stays_in_weekend_nights": 2,
                "stays_in_week_nights": 3,
                "adults": 2,
                "children": 1.0,
                "babies": 0,
                "adr": 100.0,
                "reserved_room_type": "A",
                "assigned_room_type": "B",
                "is_repeated_guest": 0,
                "previous_cancellations": 0,
                "previous_bookings_not_canceled": 0,
                "country": "PRT",
                "agent": 9.0,
                "lead_time": 10,
            }
        ]
    )

    fe = FeatureEngineer()
    result = fe.transform(df)

    assert result.loc[0, "total_nights"] == 5
    assert result.loc[0, "total_guests"] == 3.0
    assert result.loc[0, "adr_per_night"] == 20.0
    assert result.loc[0, "is_new_guest"] == 1
    assert result.loc[0, "is_domestic"] == 1
    assert result.loc[0, "is_agent_booking"] == 1
    assert result.loc[0, "guest_composition"] == "family"
    assert result.loc[0, "stay_category"] == "mixed"
    assert abs(result.loc[0, "log_lead_time"] - np.log1p(10)) < 1e-5


def test_feature_engineer_no_division_by_zero():
    """Verifica que adr_per_night no divide por cero cuando total_nights == 0."""
    import numpy as np
    df = pd.DataFrame(
        [
            {
                "stays_in_weekend_nights": 0,
                "stays_in_week_nights": 0,
                "adults": 1,
                "children": 0.0,
                "babies": 0,
                "adr": 50.0,
                "reserved_room_type": "A",
                "assigned_room_type": "A",
                "is_repeated_guest": 0,
                "previous_cancellations": 0,
                "previous_bookings_not_canceled": 0,
                "country": "ESP",
                "agent": None,
                "lead_time": 0,
            }
        ]
    )

    fe = FeatureEngineer()
    result = fe.transform(df)

    # Cuando total_nights == 0, adr_per_night debe ser igual a adr
    assert result.loc[0, "adr_per_night"] == 50.0

