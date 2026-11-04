"""Tests of how the notebooks write their figures in prose."""

from hotel_cancellation.evaluation.notebook_helpers import count, percent


def test_percent_keeps_one_decimal() -> None:
    assert percent(0.7734) == "77.3 %"


def test_count_separates_thousands_with_spaces() -> None:
    assert count(40509) == "40 509"
    assert count(40509.0) == "40 509"
    assert count(118563) == "118 563"
