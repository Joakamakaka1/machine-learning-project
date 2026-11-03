"""Tests of the API through HTTP, with the real lifespan and the real champion."""

import json
import shutil
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from hotel_cancellation import champion, config
from hotel_cancellation.api import deps
from hotel_cancellation.api.main import create_app
from hotel_cancellation.api.schemas import MAX_BATCH_SIZE
from hotel_cancellation.data import data_loader, schema

CHAMPION_SHA256 = "41d3a28592ae93367b8a88ca80ed60bb1e5ac2356ec2344d4984c0770f13d7f1"
PREFIX = "/api/v1"


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(create_app()) as test_client:
        yield test_client


def _booking(**changes: Any) -> dict[str, Any]:
    return {**schema.EXAMPLE_BOOKING, **changes}


def test_health_says_the_model_is_loaded(client: TestClient) -> None:
    response = client.get(f"{PREFIX}/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_version": "random_forest-41d3a28592ae",
    }


def test_predict_answers_with_the_agreed_fields(client: TestClient) -> None:
    response = client.post(f"{PREFIX}/predict", json=_booking())
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"probability", "threshold", "label", "risk_band", "model_version"}
    assert 0.0 <= body["probability"] <= 1.0
    assert body["threshold"] == 0.25
    assert body["label"] in {"canceled", "not_canceled"}
    assert body["risk_band"] in {"low", "medium", "high"}
    assert body["model_version"] == "random_forest-41d3a28592ae"


def test_a_booking_without_agent_is_valid(client: TestClient) -> None:
    assert client.post(f"{PREFIX}/predict", json=_booking(agent=None)).status_code == 200


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"adults": 0, "children": 0, "babies": 0}, id="no-guests"),
        pytest.param({"stays_in_weekend_nights": 0, "stays_in_week_nights": 0}, id="no-nights"),
        pytest.param({"adr": -1.0}, id="negative-adr"),
        pytest.param({"adr": config.ADR_UPPER_LIMIT}, id="adr-at-the-limit"),
        pytest.param({"adr": "110"}, id="adr-as-text"),
        pytest.param({"lead_time": 1.5}, id="fractional-count"),
        pytest.param({"lead_time": -1}, id="negative-count"),
        pytest.param({"lead_time": 10**400}, id="count-too-large-for-a-float"),
        pytest.param({"is_repeated_guest": 2}, id="flag-out-of-range"),
        pytest.param({"is_repeated_guest": True}, id="flag-as-boolean"),
        pytest.param({"hotel": "Motel"}, id="unknown-hotel"),
        pytest.param({"deposit_type": "no deposit"}, id="unknown-deposit"),
        pytest.param({"agent": "9.0"}, id="agent-read-as-a-number"),
        pytest.param({"agent": 9}, id="agent-as-a-number"),
        pytest.param({"country": "PRT"}, id="forbidden-column"),
        pytest.param({"total_of_special_requests": 1}, id="scenario-column"),
    ],
)
def test_impossible_or_foreign_bookings_are_rejected(
    client: TestClient, changes: dict[str, Any]
) -> None:
    assert client.post(f"{PREFIX}/predict", json=_booking(**changes)).status_code == 422


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"adults": 0, "children": 0, "babies": 0}, "no_guests"),
        ({"stays_in_weekend_nights": 0, "stays_in_week_nights": 0}, "no_nights"),
    ],
)
def test_impossible_bookings_say_why_with_a_stable_code(
    client: TestClient, changes: dict[str, Any], code: str
) -> None:
    response = client.post(f"{PREFIX}/predict", json=_booking(**changes))
    assert [error["type"] for error in response.json()["detail"]] == [code]


def test_every_input_feature_is_required(client: TestClient) -> None:
    for column in schema.INPUT_FEATURES:
        incomplete = _booking()
        del incomplete[column]
        assert client.post(f"{PREFIX}/predict", json=incomplete).status_code == 422, column


def test_the_contract_is_built_from_the_schema(client: TestClient) -> None:
    components = client.get("/openapi.json").json()["components"]["schemas"]
    properties = components["Booking"]["properties"]
    assert set(properties) == set(schema.INPUT_FEATURES)
    for column in schema.CATEGORICAL_FEATURES:
        if column in schema.ALLOWED_VALUES:
            assert set(properties[column]["enum"]) == schema.ALLOWED_VALUES[column]


def test_batch_keeps_the_order_and_matches_single_predictions(client: TestClient) -> None:
    risky = _booking(deposit_type="Non Refund", lead_time=200)
    response = client.post(f"{PREFIX}/predict/batch", json={"bookings": [_booking(), risky]})
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert predictions[0] == client.post(f"{PREFIX}/predict", json=_booking()).json()
    assert predictions[1] == client.post(f"{PREFIX}/predict", json=risky).json()


def test_batch_accepts_its_limit_and_rejects_more(client: TestClient) -> None:
    at_limit = {"bookings": [_booking()] * MAX_BATCH_SIZE}
    over_limit = {"bookings": [_booking()] * (MAX_BATCH_SIZE + 1)}
    assert client.post(f"{PREFIX}/predict/batch", json=at_limit).status_code == 200
    assert client.post(f"{PREFIX}/predict/batch", json=over_limit).status_code == 422


def test_batch_rejects_an_empty_list(client: TestClient) -> None:
    assert client.post(f"{PREFIX}/predict/batch", json={"bookings": []}).status_code == 422


def test_model_info_identifies_the_model_and_marks_the_costs_as_assumed(
    client: TestClient,
) -> None:
    info = client.get(f"{PREFIX}/model-info").json()
    assert info["model"]["name"] == "random_forest"
    assert info["model"]["calibration"] == "isotonic"
    assert info["model"]["sha256"] == CHAMPION_SHA256
    assert info["threshold"] == 0.25
    assert info["costs"]["are_assumptions"] is True
    assert info["costs"]["false_positive"] == 1.0
    assert info["costs"]["false_negative"] == 3.0
    assert info["input_features"] == list(schema.INPUT_FEATURES)


def test_model_info_gives_validation_and_test_metrics_with_every_subset(
    client: TestClient,
) -> None:
    info = client.get(f"{PREFIX}/model-info").json()
    assert info["validation"]["mean"]["roc_auc"] == pytest.approx(0.8172, abs=1e-4)
    assert info["validation"]["std"]["roc_auc"] == pytest.approx(0.0231, abs=1e-4)
    roc_auc = {row["subset"]: row["roc_auc"] for row in info["test"]["subsets"]}
    assert roc_auc == pytest.approx(
        {"all": 0.8175, "unique": 0.7608, "no_deposit": 0.7529, "non_refund": 0.9998}, abs=1e-4
    )
    assert info["test"]["roc_auc_ci_95"] == pytest.approx([0.8058, 0.8297], abs=1e-4)
    assert info["test"]["executed_at"].startswith("2026-11-02")


def test_evaluate_returns_what_the_single_execution_saved(client: TestClient) -> None:
    evaluation = client.get(f"{PREFIX}/evaluate").json()
    assert len(evaluation["record"]["executions"]) == 1
    assert evaluation["record"]["executions"][0]["forced"] is False
    assert evaluation["costs_are_assumptions"] is True
    rows = [
        row
        for row in evaluation["metrics"]
        if (row["model"], row["threshold_name"], row["subset"])
        == ("random_forest", "decision", "all")
    ]
    assert rows[0]["roc_auc"] == pytest.approx(0.8175, abs=1e-4)
    assert rows[0]["cost_1_3"] == pytest.approx(0.4761, abs=1e-4)
    assert {row["month"] for row in evaluation["monthly"]} >= {"2017-01", "2017-08"}
    assert evaluation["special_requests_scenario"]["informative_only"] is True
    assert evaluation["special_requests_scenario"]["rows"][0]["roc_auc"] == pytest.approx(
        0.8369, abs=1e-4
    )


def test_evaluate_never_reads_the_test_rows_nor_predicts(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("/evaluate must not compute anything")

    monkeypatch.setattr(data_loader, "load_raw", _forbidden)
    predictor = client.app.state.predictor  # type: ignore[attr-defined]
    monkeypatch.setattr(predictor, "predict", _forbidden)
    assert client.get(f"{PREFIX}/evaluate").status_code == 200


def test_there_is_no_way_to_train_through_the_api(client: TestClient) -> None:
    assert client.post(f"{PREFIX}/train").status_code == 404
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths) == {
        f"{PREFIX}/health",
        f"{PREFIX}/model-info",
        f"{PREFIX}/predict",
        f"{PREFIX}/predict/batch",
        f"{PREFIX}/evaluate",
    }


def test_an_internal_error_gives_no_detail_to_the_client() -> None:
    class Broken:
        def predict(self, bookings: Any) -> Any:
            raise RuntimeError("secret path C:/models/champion")

    app = create_app()
    app.dependency_overrides[deps.get_predictor] = Broken
    with TestClient(app, raise_server_exceptions=False) as broken_client:
        response = broken_client.post(f"{PREFIX}/predict", json=_booking())
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}


@pytest.mark.parametrize(
    "origin", ["http://localhost:8501", "http://127.0.0.1:8501", "http://localhost"]
)
def test_cors_allows_local_origins(client: TestClient, origin: str) -> None:
    response = client.get(f"{PREFIX}/health", headers={"Origin": origin})
    assert response.headers["access-control-allow-origin"] == origin


@pytest.mark.parametrize(
    "origin", ["https://example.com", "http://localhost.example.com", "http://127.0.0.1.nip.io"]
)
def test_cors_refuses_any_other_origin(client: TestClient, origin: str) -> None:
    response = client.get(f"{PREFIX}/health", headers={"Origin": origin})
    assert "access-control-allow-origin" not in response.headers


def _settings_on(models_dir: Path, outputs_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOTEL_ML_MODELS_DIR", str(models_dir))
    monkeypatch.setenv("HOTEL_ML_OUTPUTS_DIR", str(outputs_dir))


def _copy_champion(target: Path) -> Path:
    models_dir = target / "models"
    shutil.copytree(config.DEFAULT_MODELS_DIR / "champion", models_dir / "champion")
    return models_dir


def test_startup_fails_clearly_if_the_model_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models_dir = _copy_champion(tmp_path)
    with (models_dir / "champion" / champion.MODEL_FILE).open("ab") as model_file:
        model_file.write(b"\0")
    _settings_on(models_dir, config.DEFAULT_OUTPUTS_DIR, monkeypatch)
    with pytest.raises(champion.ChampionIntegrityError, match="SHA-256"), TestClient(create_app()):
        pass


def test_startup_fails_clearly_if_a_library_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models_dir = _copy_champion(tmp_path)
    manifest_path = models_dir / "champion" / champion.MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["versions"]["scikit-learn"] = "0.0.1"
    manifest_path.write_text(json.dumps(manifest), "utf-8")
    _settings_on(models_dir, config.DEFAULT_OUTPUTS_DIR, monkeypatch)
    with (
        pytest.raises(champion.ChampionIntegrityError, match="scikit-learn"),
        TestClient(create_app()),
    ):
        pass


def test_without_a_saved_evaluation_the_service_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _settings_on(config.DEFAULT_MODELS_DIR, tmp_path / "outputs", monkeypatch)
    with TestClient(create_app()) as no_evaluation_client:
        assert no_evaluation_client.get(f"{PREFIX}/evaluate").status_code == 404
        assert no_evaluation_client.get(f"{PREFIX}/model-info").json()["test"] is None
        assert no_evaluation_client.post(f"{PREFIX}/predict", json=_booking()).status_code == 200


@pytest.mark.parametrize("number", ["NaN", "Infinity", "1e400"])
def test_a_number_that_json_cannot_hold_is_rejected_as_invalid(
    client: TestClient, number: str
) -> None:
    body = json.dumps(_booking(adr=0.0)).replace('"adr": 0.0', f'"adr": {number}')
    response = client.post(
        f"{PREFIX}/predict", content=body, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


def test_a_rejection_does_not_repeat_what_it_received(client: TestClient) -> None:
    response = client.post(f"{PREFIX}/predict", json=_booking(hotel="secret-value"))
    assert response.status_code == 422
    assert "secret-value" not in response.text
    assert response.json()["detail"][0]["loc"] == ["body", "hotel"]


def test_the_manifest_cannot_point_to_another_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models_dir = _copy_champion(tmp_path)
    folder = models_dir / "champion"
    shutil.copy(folder / champion.MODEL_FILE, folder / "other.joblib")
    (folder / champion.MODEL_FILE).unlink()
    manifest_path = folder / champion.MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["model"]["file"] = "other.joblib"
    manifest_path.write_text(json.dumps(manifest), "utf-8")
    _settings_on(models_dir, config.DEFAULT_OUTPUTS_DIR, monkeypatch)
    with pytest.raises(FileNotFoundError), TestClient(create_app()):
        pass


def test_an_evaluation_of_another_model_is_not_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs_dir = tmp_path / "outputs"
    folder = outputs_dir / "final_evaluation"
    shutil.copytree(config.DEFAULT_OUTPUTS_DIR / "final_evaluation", folder)
    record = json.loads((folder / "record.json").read_text("utf-8"))
    record["executions"][-1]["champion_sha256"] = "another-model"
    (folder / "record.json").write_text(json.dumps(record), "utf-8")
    _settings_on(config.DEFAULT_MODELS_DIR, outputs_dir, monkeypatch)
    with TestClient(create_app()) as other_model_client:
        assert other_model_client.get(f"{PREFIX}/evaluate").status_code == 404
        assert other_model_client.get(f"{PREFIX}/model-info").json()["test"] is None


def test_an_undefined_validation_metric_does_not_break_the_model_info(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models_dir = _copy_champion(tmp_path)
    manifest_path = models_dir / "champion" / champion.MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["validation"]["mean"]["roc_auc_non_refund"] = None
    manifest_path.write_text(json.dumps(manifest), "utf-8")
    _settings_on(models_dir, config.DEFAULT_OUTPUTS_DIR, monkeypatch)
    with TestClient(create_app()) as undefined_metric_client:
        response = undefined_metric_client.get(f"{PREFIX}/model-info")
    assert response.status_code == 200
    assert response.json()["validation"]["mean"]["roc_auc_non_refund"] is None


def test_the_runtime_imports_nothing_from_the_training_stack() -> None:
    script = textwrap.dedent(
        """
        import sys

        from fastapi.testclient import TestClient

        from hotel_cancellation.data import schema
        from hotel_cancellation.api.main import create_app

        with TestClient(create_app()) as client:
            booking = dict(schema.EXAMPLE_BOOKING)
            assert client.post("/api/v1/predict", json=booking).status_code == 200
            assert client.get("/api/v1/evaluate").status_code == 200

        training = {"tensorflow", "keras", "mlflow", "shap", "imblearn", "matplotlib"}
        print(sorted(training & {name.partition(".")[0] for name in sys.modules}))
        """
    )
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "[]"
