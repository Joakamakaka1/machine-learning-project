import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from hotel_cancellation import champion, config, data_loader, evaluator, features, schema
from tests.conftest import BASE_BOOKING


@pytest.fixture
def fitted(synthetic_development: pd.DataFrame) -> Pipeline:
    pipeline = features.build_pipeline(
        features.build_preprocessor(features.Family.SCALED), LogisticRegression(max_iter=500)
    )
    return pipeline.fit(
        data_loader.coerce_features(synthetic_development), synthetic_development[schema.TARGET]
    )


@pytest.fixture
def fields() -> dict[str, Any]:
    return {
        "model_name": "logistic_regression",
        "params": {"model__C": 1.0},
        "calibration_method": "sigmoid",
        "calibration_briers": {"none": 0.2, "sigmoid": 0.19, "isotonic": 0.195},
        "threshold_contrast": [{"threshold": 0.25, "cost_at_threshold": 0.5}],
        "validation": {
            "windows": {"2016Q1": {"roc_auc": 0.81, "roc_auc_non_refund": float("nan")}}
        },
        "selection": [{"model": "logistic_regression", "within_rule": True}],
        "seed": 42,
        "profile": "fast",
        "development_rows": 100,
    }


@pytest.fixture
def package(tmp_path: Path, fitted: Pipeline, fields: dict[str, Any]) -> Path:
    folder = tmp_path / "champion"
    champion.save_champion(fitted, folder, **fields)
    return folder


def _manifest(folder: Path) -> dict[str, Any]:
    def refuse(value: str) -> None:
        raise AssertionError(f"{value} is not valid JSON")

    content = (folder / champion.MANIFEST_FILE).read_text("utf-8")
    manifest: dict[str, Any] = json.loads(content, parse_constant=refuse)
    return manifest


def test_manifest_holds_the_hash_of_the_saved_model(package: Path) -> None:
    content = (package / champion.MODEL_FILE).read_bytes()
    manifest = _manifest(package)
    assert manifest["model"]["sha256"] == hashlib.sha256(content).hexdigest()
    assert manifest["model"]["size_bytes"] == len(content)
    assert manifest["model"]["name"] == "logistic_regression"
    assert manifest["model"]["calibration"] == "sigmoid"


def test_manifest_holds_the_versions_that_restore_it(package: Path) -> None:
    versions = _manifest(package)["versions"]
    assert {"python", "scikit-learn", "xgboost", "pandas", "numpy", "joblib"} <= set(versions)


def test_manifest_declares_the_threshold_and_its_costs(package: Path) -> None:
    decision = _manifest(package)["decision"]
    assert decision["threshold"] == pytest.approx(0.25)
    assert decision["false_positive_cost"] == 1.0
    assert decision["false_negative_cost"] == 3.0
    assert [s["threshold"] for s in decision["cost_scenarios"]] == pytest.approx(
        [evaluator.optimal_threshold(s) for s in evaluator.COST_SCENARIOS]
    )
    assert decision["costs_are_assumptions"] is True


def test_manifest_declares_the_windows_of_the_protocol(package: Path) -> None:
    windows = _manifest(package)["windows"]
    assert windows["base_end"] == "2016-10-01"
    assert windows["calibration_end"] == "2017-01-01"
    assert [w["name"] for w in windows["expanding"]] == [w.name for w in config.EXPANDING_WINDOWS]
    assert windows["test_start"] == "2017-01-01"


def test_manifest_holds_the_commit_the_validation_and_the_inputs(package: Path) -> None:
    manifest = _manifest(package)
    assert len(manifest["code"]["commit"]) == 40
    assert isinstance(manifest["code"]["dirty"], bool)
    assert manifest["validation"]["windows"]["2016Q1"]["roc_auc"] == 0.81
    # JSON has no NaN: a metric that could not be computed is written as null.
    assert manifest["validation"]["windows"]["2016Q1"]["roc_auc_non_refund"] is None
    assert tuple(manifest["input_features"]) == schema.INPUT_FEATURES
    assert manifest["dataset_sha256"] == config.RAW_DATASET_SHA256


def test_loading_returns_the_same_predictions(
    package: Path, fitted: Pipeline, synthetic_development: pd.DataFrame
) -> None:
    loaded = champion.load_champion(package)
    rows = data_loader.coerce_features(synthetic_development.head(10))
    pd.testing.assert_frame_equal(
        pd.DataFrame(loaded.model.predict_proba(rows)), pd.DataFrame(fitted.predict_proba(rows))
    )
    assert loaded.manifest["model"]["name"] == "logistic_regression"


def test_loading_detects_a_modified_model(package: Path) -> None:
    (package / champion.MODEL_FILE).write_bytes(b"modified")
    with pytest.raises(champion.ChampionIntegrityError, match="SHA-256"):
        champion.load_champion(package)


def test_loading_detects_other_library_versions(package: Path) -> None:
    manifest = _manifest(package)
    manifest["versions"]["scikit-learn"] = "0.0.1"
    (package / champion.MANIFEST_FILE).write_text(json.dumps(manifest), "utf-8")
    with pytest.raises(champion.ChampionIntegrityError, match="scikit-learn"):
        champion.load_champion(package)


@pytest.mark.slow
def test_the_versioned_champion_loads_and_behaves() -> None:
    folder = config.get_settings().champion_dir
    loaded = champion.load_champion(folder)
    features.assert_no_forbidden_inputs(loaded.model)
    assert loaded.manifest["profile"] == "full"
    assert loaded.manifest["code"]["dirty"] is False

    booking = {**BASE_BOOKING, "lead_time": 120}
    rows = data_loader.coerce_features(
        pd.DataFrame(
            [
                booking,
                {**booking, "deposit_type": "Non Refund"},
                {**booking, "lead_time": 1},
                {**booking, "agent": "never-seen", "meal": None},
            ]
        )
    )
    probabilities = loaded.model.predict_proba(rows)[:, 1]
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert probabilities[1] > probabilities[0] > probabilities[2]


def test_saving_refuses_a_model_that_uses_a_forbidden_column(
    tmp_path: Path, synthetic_development: pd.DataFrame, fields: dict[str, Any]
) -> None:
    leaking = features.build_pipeline(
        features.build_preprocessor(features.Family.SCALED, numeric=("total_of_special_requests",)),
        LogisticRegression(max_iter=100),
    ).fit(synthetic_development, synthetic_development[schema.TARGET])
    with pytest.raises(features.LeakageError, match="total_of_special_requests"):
        champion.save_champion(leaking, tmp_path / "champion", **fields)
    assert not (tmp_path / "champion" / champion.MANIFEST_FILE).exists()
