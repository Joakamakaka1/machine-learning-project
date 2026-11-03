"""Check that a running API gives the probabilities saved by the single evaluation.

The model file was created on Windows. Pointed at the container, this proves that it behaves the
same on Linux, through the whole path of a request. It only compares predictions row by row: it
measures nothing and writes nothing.

    uv run python -m scripts.verify_served_predictions --rows 2000
"""

import argparse
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import batched
from typing import Any

import httpx
import pandas as pd

from hotel_cancellation import config, data_loader, schema
from hotel_cancellation.api.schemas import MAX_BATCH_SIZE

_STORED_COLUMN = "champion"
_TOLERANCE = 1e-12
_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True)
class Comparison:
    """How far the served probabilities are from the saved ones."""

    rows: int
    identical: int
    largest_difference: float
    model_version: str

    @property
    def agrees(self) -> bool:
        """Tell whether every row agrees, allowing only for the rounding of a floating point."""
        return self.rows > 0 and self.largest_difference <= _TOLERANCE


def _load_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    test = data_loader.split_temporal(data_loader.clean(data_loader.load_raw())).test
    stored = pd.read_csv(config.get_settings().final_evaluation_dir / "predictions.csv.gz")
    same_rows = (
        len(test) == len(stored)
        and test[schema.TARGET].to_numpy().tolist() == stored[schema.TARGET].to_numpy().tolist()
        and test["hotel"].to_numpy().tolist() == stored["hotel"].to_numpy().tolist()
    )
    if not same_rows:
        raise ValueError("The saved predictions do not follow the order of the test rows.")
    return test, stored


def _as_request(row: pd.Series) -> dict[str, Any]:
    """Write a row as the API demands it: counts as integers and a missing agent as null."""
    booking: dict[str, Any] = {}
    for column in schema.INPUT_FEATURES:
        value = row[column]
        if column == "adr":
            booking[column] = float(value)
        elif column in schema.NUMERIC_FEATURES:
            booking[column] = int(value)
        else:
            booking[column] = None if pd.isna(value) else str(value)
    return booking


def compare(client: httpx.Client, *, rows: int, seed: int) -> Comparison:
    """Send a sample of the rows of 2017 to the API and compare what comes back.

    Args:
        client: Client whose base address is the API.
        rows: Rows in the sample.
        seed: Seed of the sample.

    Returns:
        The comparison.
    """
    test, stored = _load_rows()
    sample = test.sample(n=rows, random_state=seed)
    features = data_loader.coerce_features(sample)
    bookings = [_as_request(row) for _, row in features.iterrows()]
    served: list[float] = []
    version = ""
    for batch in batched(bookings, MAX_BATCH_SIZE):
        response = client.post("/api/v1/predict/batch", json={"bookings": list(batch)})
        response.raise_for_status()
        predictions = response.json()["predictions"]
        served.extend(prediction["probability"] for prediction in predictions)
        version = predictions[0]["model_version"]
    expected = stored.loc[sample.index, _STORED_COLUMN].tolist()
    differences = [abs(left - right) for left, right in zip(served, expected, strict=True)]
    return Comparison(
        rows=len(served),
        identical=sum(difference == 0 for difference in differences),
        largest_difference=max(differences, default=math.inf),
        model_version=version,
    )


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the comparison against a running API.

    Returns:
        0 if every row agrees, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    options = parser.parse_args(arguments)
    with httpx.Client(
        base_url=options.api_url, timeout=_TIMEOUT_SECONDS, trust_env=False
    ) as client:
        result = compare(client, rows=options.rows, seed=options.seed)
    print(f"Modelo: {result.model_version}")
    print(f"Filas comparadas: {result.rows}")
    print(f"Filas idénticas: {result.identical}")
    print(f"Mayor diferencia absoluta: {result.largest_difference:.3e}")
    print("Coinciden" if result.agrees else "NO coinciden")
    return 0 if result.agrees else 1


if __name__ == "__main__":
    sys.exit(main())
