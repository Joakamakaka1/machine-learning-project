import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from hotel_cancellation.data import data_loader, features, schema


@pytest.fixture
def rows(synthetic_development: pd.DataFrame) -> pd.DataFrame:
    return data_loader.coerce_features(synthetic_development)


@pytest.fixture
def target(synthetic_development: pd.DataFrame) -> pd.Series:
    return synthetic_development[schema.TARGET]


def test_only_the_scaled_family_scales_the_numbers(rows: pd.DataFrame) -> None:
    scaled = features.build_preprocessor(features.Family.SCALED).fit(rows)
    plain = features.build_preprocessor(features.Family.UNSCALED).fit(rows)
    position = list(scaled.get_feature_names_out()).index("numeric__lead_time")
    assert scaled.transform(rows)[:, position].mean() == pytest.approx(0, abs=1e-9)
    assert plain.transform(rows)[:, position].max() > 200


def test_preprocessor_encodes_missing_and_unseen_values_without_nans(rows: pd.DataFrame) -> None:
    preprocessor = features.build_preprocessor(features.Family.UNSCALED).fit(rows)
    unseen = rows.head(3).assign(
        agent=pd.Series(["never-seen", None, "9"], dtype="str"),
        adr=[np.nan, 10.0, 10.0],
    )
    encoded = preprocessor.transform(unseen)
    assert encoded.shape[0] == 3
    assert not np.isnan(encoded).any()


def test_rare_categories_are_grouped(rows: pd.DataFrame) -> None:
    rare = rows.copy()
    rare.loc[rare.index[:5], "agent"] = "rare"
    preprocessor = features.build_preprocessor(features.Family.UNSCALED).fit(rare)
    names = list(preprocessor.get_feature_names_out())
    assert "categorical__agent_rare" not in names
    assert "categorical__agent_infrequent_sklearn" in names


def _logistic(columns: tuple[str, ...]) -> Pipeline:
    return features.build_pipeline(
        features.build_preprocessor(features.Family.SCALED, numeric=columns),
        LogisticRegression(max_iter=500),
    )


def test_guard_accepts_the_input_features(rows: pd.DataFrame, target: pd.Series) -> None:
    pipeline = _logistic(schema.NUMERIC_FEATURES).fit(rows, target)
    features.assert_no_forbidden_inputs(pipeline)


@pytest.mark.parametrize("column", sorted(schema.FORBIDDEN_FEATURES))
def test_guard_fails_for_every_forbidden_column(column: str) -> None:
    pipeline = _logistic(("lead_time", column))
    with pytest.raises(features.LeakageError, match=column):
        features.assert_no_forbidden_inputs(pipeline)


def test_guard_fails_for_a_column_that_is_not_an_input() -> None:
    with pytest.raises(features.LeakageError, match="arrival_date"):
        features.assert_no_forbidden_inputs(_logistic(("lead_time", schema.ARRIVAL_DATE)))


def test_guard_sees_through_the_calibration_wrapper(
    synthetic_development: pd.DataFrame, target: pd.Series
) -> None:
    columns = ("lead_time", "total_of_special_requests")
    leaking = _logistic(columns).fit(synthetic_development, target)
    calibrated = CalibratedClassifierCV(FrozenEstimator(leaking), method="sigmoid")
    calibrated.fit(synthetic_development, target)
    assert roc_auc_score(target, calibrated.predict_proba(synthetic_development)[:, 1]) > 0.99
    with pytest.raises(features.LeakageError, match="total_of_special_requests"):
        features.assert_no_forbidden_inputs(calibrated)


def test_guard_rejects_a_model_it_cannot_inspect() -> None:
    with pytest.raises(features.LeakageError, match="preprocess"):
        features.assert_no_forbidden_inputs(LogisticRegression())


def test_forbidden_columns_in_the_frame_do_not_change_the_predictions(
    synthetic_development: pd.DataFrame, target: pd.Series
) -> None:
    pipeline = features.build_pipeline(
        features.build_preprocessor(features.Family.SCALED), LogisticRegression(max_iter=500)
    ).fit(synthetic_development, target)
    tampered = synthetic_development.assign(
        total_of_special_requests=0, country="XXX", booking_changes=99
    )
    features.assert_no_forbidden_inputs(pipeline)
    np.testing.assert_array_equal(
        pipeline.predict_proba(synthetic_development), pipeline.predict_proba(tampered)
    )
    assert roc_auc_score(target, pipeline.predict_proba(tampered)[:, 1]) < 0.95
