import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import roc_auc_score

from hotel_cancellation import data_loader, features, models, schema


@pytest.fixture
def rows(synthetic_development: pd.DataFrame) -> pd.DataFrame:
    return data_loader.coerce_features(synthetic_development)


@pytest.fixture
def target(synthetic_development: pd.DataFrame) -> pd.Series:
    return synthetic_development[schema.TARGET]


def test_catalog_lists_the_five_models_from_simplest_to_most_complex() -> None:
    names = [spec.name for spec in models.model_catalog("full")]
    assert names == ["logistic_regression", "decision_tree", "random_forest", "xgboost", "mlp"]
    assert tuple(names) == models.SIMPLICITY_ORDER


def test_catalog_respects_the_search_budget_of_the_protocol() -> None:
    catalog = {spec.name: spec for spec in models.model_catalog("full")}
    assert catalog["mlp"].n_iter == 6
    assert all(spec.n_iter <= 20 for spec in catalog.values())


def test_only_linear_models_and_the_mlp_scale_their_inputs() -> None:
    family = {spec.name: spec.family for spec in models.model_catalog("full")}
    assert family["logistic_regression"] is features.Family.SCALED
    assert family["mlp"] is features.Family.SCALED
    assert family["xgboost"] is features.Family.UNSCALED


def test_models_do_not_ask_for_threads_or_class_weights() -> None:
    for spec in models.model_catalog("full"):
        params = spec.pipeline(seed=0).get_params()
        assert params.get("model__n_jobs") in (None, 1), spec.name
        assert params.get("model__class_weight") is None, spec.name
        assert params.get("model__scale_pos_weight") is None, spec.name


@pytest.mark.parametrize("profile", ["full", "fast"])
def test_every_search_space_value_is_a_valid_parameter(profile: models.ProfileName) -> None:
    for spec in (*models.model_catalog(profile), models.BASELINE):
        pipeline = spec.pipeline(seed=0)
        for parameter, values in spec.space.items():
            assert len(values) > 0
            for value in values:
                pipeline.set_params(**{parameter: value})
        features.assert_no_forbidden_inputs(pipeline)


@pytest.mark.parametrize("spec", models.model_catalog("fast"), ids=lambda spec: spec.name)
def test_fast_models_learn_the_synthetic_signal(
    spec: models.ModelSpec, rows: pd.DataFrame, target: pd.Series
) -> None:
    fitted = spec.pipeline(seed=0).fit(rows, target)
    assert roc_auc_score(target, fitted.predict_proba(rows)[:, 1]) > 0.7


def test_baseline_ignores_the_inputs(rows: pd.DataFrame, target: pd.Series) -> None:
    fitted = models.BASELINE.pipeline(seed=0).fit(rows, target)
    probabilities = fitted.predict_proba(rows)[:, 1]
    assert np.ptp(probabilities) == 0
    assert probabilities[0] == pytest.approx(target.mean())


def test_catalog_can_be_imported_without_loading_tensorflow() -> None:
    code = (
        "import sys; import hotel_cancellation.models as m; m.model_catalog('full');"
        "sys.exit(int('tensorflow' in sys.modules or 'keras' in sys.modules))"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0  # noqa: S603


@pytest.mark.slow
class TestKerasAdapter:
    @pytest.fixture
    def classifier(self) -> models.KerasMLPClassifier:
        pytest.importorskip("keras")
        return models.KerasMLPClassifier(
            hidden_layers=(8,), batch_size=32, epochs=5, patience=2, random_state=0
        )

    @pytest.fixture
    def encoded(self, rows: pd.DataFrame, target: pd.Series) -> np.ndarray:
        preprocessor = features.build_preprocessor(features.Family.SCALED)
        return np.asarray(preprocessor.fit_transform(rows, target))

    def test_gives_probabilities_of_both_classes(
        self, classifier: models.KerasMLPClassifier, encoded: np.ndarray, target: pd.Series
    ) -> None:
        fitted = classifier.fit(encoded, target.to_numpy())
        probabilities = fitted.predict_proba(encoded)
        assert probabilities.shape == (len(encoded), 2)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
        assert set(np.unique(fitted.predict(encoded))) <= {0, 1}
        assert roc_auc_score(target, probabilities[:, 1]) > 0.6

    def test_can_be_cloned_with_its_parameters(self, classifier: models.KerasMLPClassifier) -> None:
        assert clone(classifier).get_params() == classifier.get_params()

    def test_is_reproducible_with_the_same_seed(
        self, classifier: models.KerasMLPClassifier, encoded: np.ndarray, target: pd.Series
    ) -> None:
        first = clone(classifier).fit(encoded, target.to_numpy()).predict_proba(encoded)
        second = clone(classifier).fit(encoded, target.to_numpy()).predict_proba(encoded)
        np.testing.assert_allclose(first, second, atol=1e-5)

    def test_survives_joblib_inside_the_calibration_wrapper(
        self,
        classifier: models.KerasMLPClassifier,
        rows: pd.DataFrame,
        target: pd.Series,
        tmp_path: Path,
    ) -> None:
        pipeline = features.build_pipeline(
            features.build_preprocessor(features.Family.SCALED), classifier
        ).fit(rows, target)
        calibrated = CalibratedClassifierCV(FrozenEstimator(pipeline), method="sigmoid")
        calibrated.fit(rows, target)

        path = tmp_path / "model.joblib"
        joblib.dump(calibrated, path)
        restored = joblib.load(path)
        np.testing.assert_allclose(
            restored.predict_proba(rows), calibrated.predict_proba(rows), atol=1e-6
        )
