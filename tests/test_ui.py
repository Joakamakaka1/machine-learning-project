"""Tests of every page with `AppTest`. The API is simulated, so nothing has to be running."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from hotel_cancellation import config, schema
from hotel_cancellation.api.schemas import Booking

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

from hotel_cancellation import ui
from hotel_cancellation.ui import booking, client

pytestmark = pytest.mark.slow

APP = str(Path(ui.__file__).parent / "app.py")
PAGES = (
    "views/overview.py",
    "views/prediction.py",
    "views/evaluation.py",
    "views/threshold.py",
    "views/interpretation.py",
    "views/data.py",
)
PREDICT_RESPONSE = {
    "probability": 0.0,
    "threshold": 0.25,
    "label": "not_canceled",
    "risk_band": "low",
    "model_version": "random_forest-41d3a28592ae",
}
HEALTH_RESPONSE = {
    "status": "ok",
    "model_loaded": True,
    "model_version": "random_forest-41d3a28592ae",
}

Handler = Callable[[httpx.Request], httpx.Response]


def _fake_api(
    monkeypatch: pytest.MonkeyPatch, predict: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Simulate the API over HTTP and keep the bookings it receives."""
    received: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/predict"):
            received.append(json.loads(request.content))
            return httpx.Response(200, json=predict or PREDICT_RESPONSE)
        return httpx.Response(200, json=HEALTH_RESPONSE)

    _install(monkeypatch, handler)
    return received


def _install(monkeypatch: pytest.MonkeyPatch, handler: Handler) -> None:
    monkeypatch.setattr(
        client,
        "default_client",
        lambda: client.ApiClient("http://api.test", transport=httpx.MockTransport(handler)),
    )


def _api_down(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    _install(monkeypatch, handler)


def _open(page: str | None = None) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=120).run()
    if page is not None:
        app.switch_page(page).run()
    assert not app.exception, [item.message for item in app.exception]
    return app


def _text(app: AppTest) -> str:
    blocks = [
        *app.title,
        *app.header,
        *app.subheader,
        *app.markdown,
        *app.caption,
        *app.warning,
        *app.info,
        *app.error,
        *app.success,
    ]
    metrics = [f"{item.label} {item.value}" for item in app.metric]
    return "\n".join([*(str(item.value) for item in blocks), *metrics])


@pytest.fixture(autouse=True)
def fake_api(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    return _fake_api(monkeypatch)


def test_every_page_opens_without_exceptions() -> None:
    for page in PAGES:
        assert _open(page).title, page


def test_every_page_but_the_prediction_works_without_the_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _api_down(monkeypatch)
    for page in PAGES:
        assert _open(page).title, page


def test_overview_never_shows_the_global_roc_auc_alone() -> None:
    text = _text(_open())
    for figure in ("0.8175", "0.7608", "0.7529", "0.9998", "[0.8058, 0.8297]"):
        assert figure in text
    assert "no es causal" in text


def test_the_form_opens_with_a_valid_booking_and_every_input() -> None:
    fields = booking.all_fields()
    assert {field.name for field in fields} == set(schema.INPUT_FEATURES)
    defaults = {field.name: schema.EXAMPLE_BOOKING[field.name] for field in fields}
    Booking.model_validate(booking.build_payload(defaults))


def test_the_form_reads_the_allowed_values_from_the_schema() -> None:
    app = _open("views/prediction.py")
    options = {str(box.key).removeprefix("field_"): set(box.options) for box in app.selectbox}
    for column, allowed in schema.ALLOWED_VALUES.items():
        if column in schema.INPUT_FEATURES:
            assert options[column] == set(allowed), column


def test_submitting_sends_the_form_and_shows_the_answer(
    fake_api: list[dict[str, Any]],
) -> None:
    app = _open("views/prediction.py")
    app.number_input(key="field_lead_time").set_value(120)
    app.button(key="FormSubmitter:booking_form-Predecir").click().run()
    assert not app.exception
    assert fake_api[0]["lead_time"] == 120
    assert fake_api[0]["hotel"] == schema.EXAMPLE_BOOKING["hotel"]
    Booking.model_validate(fake_api[0])
    text = _text(app)
    assert "<1 %" in text
    assert "0.0 %" not in text
    assert "No se marca" in text
    assert "Bajo" in text


def test_an_empty_agent_is_sent_as_null(fake_api: list[dict[str, Any]]) -> None:
    app = _open("views/prediction.py")
    app.text_input(key="field_agent").set_value("")
    app.button(key="FormSubmitter:booking_form-Predecir").click().run()
    assert fake_api[0]["agent"] is None


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ({"probability": 1.0, "label": "canceled", "risk_band": "high"}, ">99 %"),
        ({"probability": 0.2504, "label": "canceled", "risk_band": "medium"}, "25.0 %"),
    ],
)
def test_a_flagged_booking_is_explained(
    monkeypatch: pytest.MonkeyPatch, answer: dict[str, Any], expected: str
) -> None:
    _fake_api(monkeypatch, {**PREDICT_RESPONSE, **answer})
    app = _open("views/prediction.py")
    app.button(key="FormSubmitter:booking_form-Predecir").click().run()
    text = _text(app)
    assert expected in text
    assert "Se marca" in text
    assert "escalones" in text


def test_a_rejected_booking_is_explained_in_spanish(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/predict"):
            errors = [{"type": "no_guests", "loc": ["body"], "msg": "a booking needs guests"}]
            return httpx.Response(422, json={"detail": errors})
        return httpx.Response(200, json=HEALTH_RESPONSE)

    _install(monkeypatch, handler)
    app = _open("views/prediction.py")
    app.button(key="FormSubmitter:booking_form-Predecir").click().run()
    assert not app.exception
    assert "al menos un huésped" in _text(app)


def test_the_prediction_warns_if_the_api_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    _api_down(monkeypatch)
    app = _open("views/prediction.py")
    assert "no está disponible" in _text(app)
    app.button(key="FormSubmitter:booking_form-Predecir").click().run()
    assert not app.exception
    assert "No se puede predecir" in _text(app)


def test_an_internal_error_of_the_api_shows_no_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/predict"):
            return httpx.Response(500, json={"detail": "Internal server error"})
        return httpx.Response(200, json=HEALTH_RESPONSE)

    _install(monkeypatch, handler)
    app = _open("views/prediction.py")
    app.button(key="FormSubmitter:booking_form-Predecir").click().run()
    assert "Internal server error" not in _text(app)
    assert "No se puede predecir" in _text(app)


def test_evaluation_declares_the_scenario_as_informative_only() -> None:
    text = _text(_open("views/evaluation.py"))
    assert "0.8369" in text
    assert "informativo" in text
    assert "no la usa" in text
    assert "128" in text
    assert "no es causal" in text
    assert "0.7529" in text


def test_threshold_tells_the_truth_about_the_served_threshold() -> None:
    app = _open("views/threshold.py")
    text = _text(app)
    for figure in ("77.3 %", "0.941", "0.473", "0.476", "0.430", "0.439", "0.6117"):
        assert figure in text, figure
    assert "Sin modelo: marcar todas las reservas" in text
    assert "supuesto" in text
    assert "no sería una evaluación limpia" in text


def test_the_contrast_on_development_data_keeps_both_costs() -> None:
    app = _open("views/threshold.py")
    contrast = next(table.value for table in app.dataframe if "Umbral teórico" in table.value)
    assert contrast["Coste con el umbral teórico"].tolist()[0] == "0.3529"
    assert contrast["Coste con el mejor umbral"].tolist()[0] == "0.3519"


def test_the_threshold_controls_recompute_the_cost() -> None:
    app = _open("views/threshold.py")
    before = _text(app)
    assert "Coste por reserva con estos controles 0.476" in before
    app.slider(key="chosen_threshold").set_value(0.3).run()
    assert not app.exception
    assert "Coste por reserva con estos controles 0.430" in _text(app)
    app.slider(key="false_negative_cost").set_value(5.0).run()
    assert "Umbral teórico 0.167" in _text(app)


def test_interpretation_warns_and_offers_no_explanation_per_booking() -> None:
    text = _text(_open("views/interpretation.py"))
    assert "no es causal" in text
    assert "por reserva" in text
    assert "deposit_type" in text


def test_data_cites_the_source_and_explains_the_exclusions() -> None:
    app = _open("views/data.py")
    assert "CC BY 4.0" in _text(app)
    tables = [table.value for table in app.dataframe]
    cells = " ".join(str(cell) for table in tables for cell in table.to_numpy().ravel())
    assert "118 563" in cells
    assert "78 054" in cells
    assert "40 509" in cells
    excluded = next(table for table in tables if "Variable" in table and len(table) == 12)
    assert set(excluded["Variable"]) == set(schema.FORBIDDEN_FEATURES)


def test_settings_point_to_the_local_api_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HOTEL_ML_API_URL", raising=False)
    assert config.Settings(_env_file=None).api_url == "http://127.0.0.1:8000"  # type: ignore[call-arg]
