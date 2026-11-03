"""Model catalog, search spaces and the Keras adapter.

TensorFlow is imported only when an MLP is trained or loaded: the API image does not include it.
"""

import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from hotel_cancellation.data import features

ProfileName = Literal["full", "fast"]

# Selection breaks ties in favour of the first model of this list that stays within the rule.
SIMPLICITY_ORDER: tuple[str, ...] = (
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
    "mlp",
)

SearchSpace = Mapping[str, Sequence[Any]]

_MAX_SEARCH_ITERATIONS = 20
_MLP_CONFIGURATIONS = 6
_PREDICT_BATCH_SIZE = 4096


class KerasMLPClassifier(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Dense binary classifier with a scikit-learn interface.

    `keras.wrappers.SKLearnClassifier` does not define `predict_proba`, and calibration needs it.
    Rows must arrive in arrival order: early stopping watches the last `validation_fraction`, so
    the model validates on the most recent bookings.

    Args:
        hidden_layers: Units of each dense layer.
        dropout: Dropout after each dense layer.
        learning_rate: Adam learning rate.
        batch_size: Rows per training step.
        epochs: Maximum epochs.
        patience: Epochs without improvement before stopping.
        validation_fraction: Last share of rows used to stop early.
        random_state: Seed of the weights, the dropout and the shuffling.
    """

    def __init__(
        self,
        hidden_layers: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        epochs: int = 50,
        patience: int = 5,
        validation_fraction: float = 0.1,
        random_state: int = 0,
    ) -> None:
        """Store the parameters untouched, as scikit-learn requires."""
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:  # noqa: N803
        """Train the network, keeping the weights of its best validation epoch."""
        keras = _import_keras()
        keras.utils.set_random_seed(self.random_state)
        inputs = np.asarray(X, dtype="float32")
        outcomes = np.asarray(y, dtype="float32")

        network = keras.Sequential([keras.Input(shape=(inputs.shape[1],))])
        for units in self.hidden_layers:
            network.add(keras.layers.Dense(units, activation="relu"))
            network.add(keras.layers.Dropout(self.dropout, seed=self.random_state))
        network.add(keras.layers.Dense(1, activation="sigmoid"))
        network.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
        )
        stop_early = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.patience, restore_best_weights=True
        )
        history = network.fit(
            inputs,
            outcomes,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_fraction,
            callbacks=[stop_early],
            verbose=0,
        )
        self.model_ = network
        self.classes_ = np.array([0, 1])
        self.epochs_trained_ = len(history.history["loss"])
        return self

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:  # noqa: N803
        """Return the probability of each class, one column per class."""
        inputs = np.asarray(X, dtype="float32")
        positive = self.model_.predict(inputs, batch_size=_PREDICT_BATCH_SIZE, verbose=0)
        positive = positive.ravel().astype("float64")
        return np.column_stack([1 - positive, positive])

    def predict(self, X: ArrayLike) -> NDArray[np.int_]:  # noqa: N803
        """Return the most probable class."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def __getstate__(self) -> dict[str, Any]:
        """Replace the Keras model, which does not pickle, with the bytes of its `.keras` file."""
        state = super().__getstate__()
        if "model_" not in state:
            return dict(state)
        state = dict(state)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.keras"
            state.pop("model_").save(path)
            state["_model_bytes"] = path.read_bytes()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Rebuild the Keras model from the bytes of its `.keras` file."""
        content = state.pop("_model_bytes", None)
        super().__setstate__(state)
        if content is None:
            return
        keras = _import_keras()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.keras"
            path.write_bytes(content)
            self.model_ = keras.saving.load_model(path, compile=False)


def _import_keras() -> Any:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import keras

    return keras


@dataclass(frozen=True)
class ModelSpec:
    """A model of the catalog and how to search it."""

    name: str
    family: features.Family
    estimator: Callable[[int], Any]
    space: SearchSpace = field(default_factory=dict)
    n_iter: int = 1
    parallel_search: bool = True

    def pipeline(self, seed: int) -> Pipeline:
        """Build the unfitted pipeline with its preprocessing."""
        return features.build_pipeline(
            features.build_preprocessor(self.family), self.estimator(seed)
        )


def _logistic_regression(seed: int) -> LogisticRegression:
    return LogisticRegression(max_iter=2000, random_state=seed)


def _decision_tree(seed: int) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(random_state=seed)


def _random_forest(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=seed)


def _xgboost(seed: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=2,
        tree_method="hist",
        n_jobs=1,
        random_state=seed,
    )


def _mlp(seed: int) -> KerasMLPClassifier:
    return KerasMLPClassifier(random_state=seed)


def _dummy(_: int) -> DummyClassifier:
    return DummyClassifier(strategy="prior")


BASELINE = ModelSpec(
    name="dummy",
    family=features.Family.UNSCALED,
    estimator=_dummy,
    space={"model__strategy": ["prior"]},
)


def _full_catalog() -> tuple[ModelSpec, ...]:
    logistic_c = [float(value) for value in np.logspace(-3, 2, 12)]
    return (
        ModelSpec(
            name="logistic_regression",
            family=features.Family.SCALED,
            estimator=_logistic_regression,
            space={"model__C": logistic_c},
            n_iter=len(logistic_c),
        ),
        ModelSpec(
            name="decision_tree",
            family=features.Family.UNSCALED,
            estimator=_decision_tree,
            space={
                "model__max_depth": [4, 5, 6, 7, 8, 10, 12, 14],
                "model__min_samples_leaf": [5, 10, 20, 50, 100, 200, 500],
                "model__criterion": ["gini", "entropy"],
                "model__max_features": [None, "sqrt", 0.5],
            },
            n_iter=_MAX_SEARCH_ITERATIONS,
        ),
        ModelSpec(
            name="random_forest",
            family=features.Family.UNSCALED,
            estimator=_random_forest,
            space={
                "model__n_estimators": [200, 300, 400],
                "model__max_depth": [8, 10, 12, 14, 16, 20],
                "model__min_samples_leaf": [2, 5, 10, 20, 50],
                "model__max_features": ["sqrt", 0.2, 0.3, 0.5],
            },
            n_iter=_MAX_SEARCH_ITERATIONS,
        ),
        ModelSpec(
            name="xgboost",
            family=features.Family.UNSCALED,
            estimator=_xgboost,
            space={
                "model__n_estimators": [200, 300, 400, 600, 800],
                "model__max_depth": [3, 4, 5, 6, 7, 8],
                "model__learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
                "model__subsample": [0.7, 0.8, 0.9, 1.0],
                "model__colsample_bytree": [0.5, 0.6, 0.8, 1.0],
                "model__min_child_weight": [1, 3, 5, 10, 20],
                "model__reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
            },
            n_iter=_MAX_SEARCH_ITERATIONS,
        ),
        ModelSpec(
            name="mlp",
            family=features.Family.SCALED,
            estimator=_mlp,
            space={
                "model__hidden_layers": [(64,), (64, 32), (128, 64), (256, 128)],
                "model__dropout": [0.0, 0.2, 0.4],
                "model__learning_rate": [3e-4, 1e-3],
                "model__batch_size": [256, 512],
            },
            n_iter=_MLP_CONFIGURATIONS,
            parallel_search=False,
        ),
    )


def _fast_forest(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=20, max_depth=6, n_jobs=1, random_state=seed)


def _fast_xgboost(seed: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=20, max_depth=3, tree_method="hist", n_jobs=1, random_state=seed
    )


def _fast_catalog() -> tuple[ModelSpec, ...]:
    """Small versions for the tests. There is no MLP, because the fast tests run without Keras."""
    return (
        ModelSpec(
            name="logistic_regression",
            family=features.Family.SCALED,
            estimator=_logistic_regression,
            space={"model__C": [0.1, 1.0]},
            n_iter=2,
        ),
        ModelSpec(
            name="decision_tree",
            family=features.Family.UNSCALED,
            estimator=_decision_tree,
            space={"model__max_depth": [3, 5], "model__min_samples_leaf": [20]},
            n_iter=2,
        ),
        ModelSpec(
            name="random_forest",
            family=features.Family.UNSCALED,
            estimator=_fast_forest,
            space={"model__min_samples_leaf": [5, 20]},
            n_iter=2,
        ),
        ModelSpec(
            name="xgboost",
            family=features.Family.UNSCALED,
            estimator=_fast_xgboost,
            space={"model__learning_rate": [0.05, 0.1]},
            n_iter=2,
        ),
    )


def model_catalog(profile: ProfileName = "full") -> tuple[ModelSpec, ...]:
    """List the models to compare, from simplest to most complex.

    Args:
        profile: `full` is what the protocol specifies, `fast` is for the tests.

    Returns:
        The specifications, without the baseline.
    """
    return _full_catalog() if profile == "full" else _fast_catalog()
