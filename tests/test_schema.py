"""Tests of the single source of truth for columns and allowed values."""

from hotel_cancellation.data import schema

EXPECTED_NUMERIC = (
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "days_in_waiting_list",
    "adr",
)
EXPECTED_CATEGORICAL = (
    "hotel",
    "meal",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
    "agent",
)
EXPECTED_FORBIDDEN = {
    "reservation_status",
    "reservation_status_date",
    "country",
    "required_car_parking_spaces",
    "assigned_room_type",
    "booking_changes",
    "total_of_special_requests",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "company",
}


def test_input_features_are_the_nineteen_agreed_variables() -> None:
    assert schema.NUMERIC_FEATURES == EXPECTED_NUMERIC
    assert schema.CATEGORICAL_FEATURES == EXPECTED_CATEGORICAL
    assert schema.INPUT_FEATURES == EXPECTED_NUMERIC + EXPECTED_CATEGORICAL
    assert len(schema.INPUT_FEATURES) == 19


def test_no_forbidden_variable_can_be_an_input_feature() -> None:
    assert set(schema.FORBIDDEN_FEATURES) == EXPECTED_FORBIDDEN
    assert not set(schema.INPUT_FEATURES) & set(schema.FORBIDDEN_FEATURES)
    assert schema.TARGET not in schema.INPUT_FEATURES


def test_every_forbidden_variable_states_its_reason() -> None:
    for column, reason in schema.FORBIDDEN_FEATURES.items():
        assert len(reason.split()) >= 3, column


def test_every_raw_column_is_classified_exactly_once() -> None:
    classified = [*schema.INPUT_FEATURES, *schema.FORBIDDEN_FEATURES, schema.TARGET]
    assert len(classified) == len(set(classified))
    assert set(classified) == set(schema.RAW_COLUMNS)
    assert len(schema.RAW_COLUMNS) == 32


def test_arrival_date_is_not_an_input_feature() -> None:
    assert schema.ARRIVAL_DATE not in schema.RAW_COLUMNS
    assert schema.ARRIVAL_DATE not in schema.INPUT_FEATURES


def test_allowed_values_are_declared_for_low_cardinality_categoricals() -> None:
    categoricals_with_free_ids = {"agent"}
    assert set(EXPECTED_CATEGORICAL) - categoricals_with_free_ids <= set(schema.ALLOWED_VALUES)
    assert schema.ALLOWED_VALUES["hotel"] == {"City Hotel", "Resort Hotel"}
    assert schema.ALLOWED_VALUES["deposit_type"] == {"No Deposit", "Non Refund", "Refundable"}
    assert "agent" not in schema.ALLOWED_VALUES


def test_the_example_booking_holds_exactly_the_input_features() -> None:
    assert set(schema.EXAMPLE_BOOKING) == set(schema.INPUT_FEATURES)
    for column, allowed in schema.ALLOWED_VALUES.items():
        if column in schema.EXAMPLE_BOOKING:
            assert schema.EXAMPLE_BOOKING[column] in allowed


def test_agent_ids_are_integers_written_as_text() -> None:
    assert schema.ID_PATTERN.fullmatch("9")
    assert schema.ID_PATTERN.fullmatch("240")
    assert not schema.ID_PATTERN.fullmatch("9.0")
    assert not schema.ID_PATTERN.fullmatch("NULL")
