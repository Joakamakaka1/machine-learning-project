"""Predictions of the served model, with the decision the project took for it."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from hotel_cancellation import champion, data_loader, features, schema

Label = Literal["canceled", "not_canceled"]
CANCELED: Label = "canceled"
NOT_CANCELED: Label = "not_canceled"

# More likely than not. It is not fitted to any data, unlike a threshold chosen on a test.
HIGH_RISK_PROBABILITY = 0.5


class RiskBand(StrEnum):
    """Reading of a probability against the decision threshold and one half."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def risk_band(probability: float, threshold: float) -> RiskBand:
    """Place a probability in its band.

    Args:
        probability: Calibrated probability of cancellation.
        threshold: Probability from which the model flags a booking.

    Returns:
        `LOW` below the threshold, `HIGH` from one half on and `MEDIUM` in between, where a flagged
        booking is still more likely to stay than to cancel.
    """
    if probability < threshold:
        return RiskBand.LOW
    return RiskBand.HIGH if probability >= HIGH_RISK_PROBABILITY else RiskBand.MEDIUM


@dataclass(frozen=True)
class BookingPrediction:
    """The outcome for one booking."""

    probability: float
    threshold: float
    label: Label
    risk_band: RiskBand
    model_version: str


class BookingPredictor:
    """Predicts with the champion after checking that it is the expected one."""

    def __init__(self, loaded: champion.LoadedChampion) -> None:
        """Keep a model whose inputs agree with the schema.

        Args:
            loaded: The model and its manifest.

        Raises:
            ChampionIntegrityError: If the manifest lists other input features.
            LeakageError: If a forbidden column reaches the model.
        """
        if tuple(loaded.manifest["input_features"]) != schema.INPUT_FEATURES:
            raise champion.ChampionIntegrityError(
                f"The manifest lists other input features: {loaded.manifest['input_features']}."
            )
        features.assert_no_forbidden_inputs(loaded.model)
        self._model = loaded.model
        self.manifest = loaded.manifest
        self.threshold = float(loaded.manifest["decision"]["threshold"])
        self.sha256 = str(loaded.manifest["model"]["sha256"])
        self.model_version = f"{loaded.manifest['model']['name']}-{self.sha256[:12]}"

    @classmethod
    def from_folder(cls, folder: Path) -> "BookingPredictor":
        """Load the package in `folder`, failing if its hash or a library version differ."""
        return cls(champion.load_champion(folder))

    def predict(self, bookings: Sequence[Mapping[str, Any]]) -> list[BookingPrediction]:
        """Predict each booking, in the order received.

        Args:
            bookings: Values of the input features. A missing `agent` is `None`.

        Returns:
            One prediction per booking.
        """
        if not bookings:
            return []
        frame = data_loader.coerce_features(pd.DataFrame(list(bookings)))
        probabilities = self._model.predict_proba(frame)[:, 1]
        return [self._decide(float(probability)) for probability in probabilities]

    def _decide(self, probability: float) -> BookingPrediction:
        return BookingPrediction(
            probability=probability,
            threshold=self.threshold,
            label=CANCELED if probability >= self.threshold else NOT_CANCELED,
            risk_band=risk_band(probability, self.threshold),
            model_version=self.model_version,
        )
