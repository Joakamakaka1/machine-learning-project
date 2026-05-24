from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from api.main import app
import api.router as router_module

client = TestClient(app)


def test_health_check():
    """Valida el endpoint de salud de la API."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OK"
    assert "project_name" in data
    assert "model_loaded" in data


def test_predict_endpoint_success(monkeypatch):
    """Valida que el endpoint de predicción retorne resultados exitosos."""
    # Mockear el predictor para aislar la llamada de ML en test
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = [1]
    mock_predictor.predict_proba.return_value = [0.854]
    monkeypatch.setattr(router_module, "_predictor", mock_predictor)

    booking_payload = {
        "hotel": "City Hotel",
        "lead_time": 120,
        "arrival_date_year": 2016,
        "arrival_date_month": "October",
        "arrival_date_week_number": 42,
        "arrival_date_day_of_month": 12,
        "stays_in_weekend_nights": 2,
        "stays_in_week_nights": 5,
        "adults": 2,
        "children": 0.0,
        "babies": 0,
        "meal": "BB",
        "country": "ESP",
        "market_segment": "Online TA",
        "distribution_channel": "TA/TO",
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "reserved_room_type": "A",
        "assigned_room_type": "A",
        "booking_changes": 0,
        "deposit_type": "No Deposit",
        "agent": "9",
        "days_in_waiting_list": 0,
        "customer_type": "Transient",
        "adr": 110.0,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 1,
    }

    response = client.post("/api/v1/predict", json=booking_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1
    assert data["probability"] == 0.854
    assert data["label"] == "Cancelado"
    assert "85.40%" in data["message"]


def test_predict_endpoint_model_not_loaded(monkeypatch):
    """Valida el comportamiento si el modelo no está cargado (503 Service Unavailable)."""
    # Forzar que el predictor sea None para simular modelo no cargado
    monkeypatch.setattr(router_module, "_predictor", None)

    booking_payload = {
        "hotel": "City Hotel",
        "lead_time": 10,
        "arrival_date_year": 2017,
        "arrival_date_month": "May",
        "arrival_date_week_number": 20,
        "arrival_date_day_of_month": 15,
        "stays_in_weekend_nights": 1,
        "stays_in_week_nights": 2,
        "adults": 2,
        "children": 0.0,
        "babies": 0,
        "meal": "BB",
        "country": "ESP",
        "market_segment": "Direct",
        "distribution_channel": "Direct",
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "reserved_room_type": "A",
        "assigned_room_type": "A",
        "booking_changes": 0,
        "deposit_type": "No Deposit",
        "agent": "NULL",
        "days_in_waiting_list": 0,
        "customer_type": "Transient",
        "adr": 90.0,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 0,
    }

    response = client.post("/api/v1/predict", json=booking_payload)
    assert response.status_code == 503
    data = response.json()
    assert "detail" in data
    assert data["status_code"] == 503
