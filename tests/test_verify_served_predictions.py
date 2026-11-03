"""Test of the comparison between a served model and the saved probabilities."""

import pytest
from fastapi.testclient import TestClient

from hotel_cancellation.api.main import create_app
from scripts.verify_served_predictions import Comparison, compare


@pytest.mark.slow
def test_the_real_service_gives_the_saved_probabilities() -> None:
    with TestClient(create_app()) as client:
        comparison = compare(client, rows=1200, seed=7)
    assert comparison.rows == 1200
    assert comparison.agrees
    assert comparison.model_version == "random_forest-41d3a28592ae"


def test_a_difference_or_an_empty_comparison_does_not_agree() -> None:
    assert not Comparison(10, 9, 1e-3, "model").agrees
    assert not Comparison(0, 0, 0.0, "model").agrees
    assert Comparison(10, 9, 1e-15, "model").agrees
