"""Tests of the predictor over the real champion, which is versioned with the code."""

import json
import shutil
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from hotel_cancellation import champion, config, data_loader, schema
from hotel_cancellation.predictor import BookingPredictor, RiskBand, risk_band

CHAMPION_DIR = config.DEFAULT_MODELS_DIR / "champion"
CHAMPION_SHA256 = "41d3a28592ae93367b8a88ca80ed60bb1e5ac2356ec2344d4984c0770f13d7f1"


@pytest.fixture(scope="module")
def predictor() -> BookingPredictor:
    return BookingPredictor.from_folder(CHAMPION_DIR)


def test_it_exposes_what_identifies_the_served_model(predictor: BookingPredictor) -> None:
    assert predictor.threshold == 0.25
    assert predictor.sha256 == CHAMPION_SHA256
    assert predictor.model_version == "random_forest-41d3a28592ae"


def test_a_prediction_carries_probability_threshold_label_band_and_version(
    predictor: BookingPredictor,
) -> None:
    prediction = predictor.predict([schema.EXAMPLE_BOOKING])[0]
    assert 0.0 <= prediction.probability <= 1.0
    assert prediction.threshold == 0.25
    assert prediction.label in {"canceled", "not_canceled"}
    assert prediction.risk_band in set(RiskBand)
    assert prediction.model_version == "random_forest-41d3a28592ae"


def test_the_label_follows_the_served_threshold(predictor: BookingPredictor) -> None:
    almost_certain = {**schema.EXAMPLE_BOOKING, "deposit_type": "Non Refund", "lead_time": 200}
    prediction = predictor.predict([almost_certain])[0]
    assert prediction.probability > 0.9
    assert prediction.label == "canceled"
    assert prediction.risk_band is RiskBand.HIGH


def test_a_booking_without_agent_is_predicted_like_any_other(predictor: BookingPredictor) -> None:
    direct = {**schema.EXAMPLE_BOOKING, "agent": None}
    alone = predictor.predict([direct])[0]
    mixed = predictor.predict([schema.EXAMPLE_BOOKING, direct])[1]
    assert alone.probability == mixed.probability


def test_predictions_keep_the_order_of_the_bookings(predictor: BookingPredictor) -> None:
    risky = {**schema.EXAMPLE_BOOKING, "deposit_type": "Non Refund", "lead_time": 200}
    together = predictor.predict([schema.EXAMPLE_BOOKING, risky, schema.EXAMPLE_BOOKING])
    assert together[0].probability == together[2].probability
    assert together[1].probability == predictor.predict([risky])[0].probability


def test_no_bookings_means_no_predictions(predictor: BookingPredictor) -> None:
    assert predictor.predict([]) == []


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.0, RiskBand.LOW),
        (0.2499, RiskBand.LOW),
        (0.25, RiskBand.MEDIUM),
        (0.4999, RiskBand.MEDIUM),
        (0.5, RiskBand.HIGH),
        (1.0, RiskBand.HIGH),
    ],
)
def test_risk_bands_split_at_the_threshold_and_at_one_half(
    probability: float, expected: RiskBand
) -> None:
    assert risk_band(probability, threshold=0.25) is expected


def _copy_champion(target: Path) -> Path:
    folder = target / "champion"
    shutil.copytree(CHAMPION_DIR, folder)
    return folder


def _edit_manifest(folder: Path, change: Any) -> None:
    path = folder / champion.MANIFEST_FILE
    manifest = json.loads(path.read_text("utf-8"))
    change(manifest)
    path.write_text(json.dumps(manifest), "utf-8")


def test_loading_fails_if_the_model_file_changed(tmp_path: Path) -> None:
    folder = _copy_champion(tmp_path)
    with (folder / champion.MODEL_FILE).open("ab") as model_file:
        model_file.write(b"\0")
    with pytest.raises(champion.ChampionIntegrityError, match="SHA-256"):
        BookingPredictor.from_folder(folder)


def test_loading_fails_if_a_library_is_on_another_version(tmp_path: Path) -> None:
    folder = _copy_champion(tmp_path)
    _edit_manifest(folder, lambda manifest: manifest["versions"].update({"scikit-learn": "0.0.1"}))
    with pytest.raises(champion.ChampionIntegrityError, match="scikit-learn"):
        BookingPredictor.from_folder(folder)


def test_loading_fails_if_the_manifest_lists_other_input_features(tmp_path: Path) -> None:
    folder = _copy_champion(tmp_path)
    _edit_manifest(folder, lambda manifest: manifest["input_features"].append("country"))
    with pytest.raises(champion.ChampionIntegrityError, match="input features"):
        BookingPredictor.from_folder(folder)


@pytest.mark.slow
def test_it_gives_the_probabilities_stored_by_the_single_evaluation(
    predictor: BookingPredictor,
) -> None:
    """Only compares predictions: the rows of 2017 produce no new metric here."""
    test_rows = data_loader.split_temporal(data_loader.clean(data_loader.load_raw())).test
    stored = pd.read_csv(config.get_settings().final_evaluation_dir / "predictions.csv.gz")
    sample = test_rows.sample(n=300, random_state=42)
    rows = cast(
        list[dict[str, Any]], data_loader.coerce_features(sample).astype(object).to_dict("records")
    )
    predictions = predictor.predict(rows)
    actual = [prediction.probability for prediction in predictions]
    assert actual == pytest.approx(stored.loc[sample.index, "champion"].tolist(), abs=1e-12)
