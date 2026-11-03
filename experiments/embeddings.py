"""Do categorical embeddings beat one-hot in the MLP? It stays an experiment unless they do.

The network shares the hidden layers, dropout and learning rate of the tuned MLP, so the only
difference with the one-hot version is how the categories enter.
"""

import json
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from experiments import common
from hotel_cancellation import config, models
from hotel_cancellation.data import data_loader, features, schema

_MISSING = "__missing__"
_UNKNOWN_INDEX = -1
_MAX_EMBEDDING_SIZE = 16
_VALIDATION_FRACTION = 0.1
_EPOCHS = 50
_PATIENCE = 5
_PREDICT_BATCH_SIZE = 4096


def _categories(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[list(schema.CATEGORICAL_FEATURES)].astype("object").fillna(_MISSING)


def embedding_mlp(
    *,
    hidden_layers: tuple[int, ...] = (64, 32),
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    epochs: int = _EPOCHS,
) -> common.FitPredict:
    """Build a variant: an embedding per category, concatenated with the scaled numbers.

    Args:
        hidden_layers: Units of each dense layer.
        dropout: Dropout after each dense layer.
        learning_rate: Adam learning rate.
        batch_size: Rows per training step.
        epochs: Maximum epochs. Early stopping watches the most recent rows.

    Returns:
        The function that trains on a window and predicts its validation rows.
    """

    def fit_predict(train: pd.DataFrame, valid: pd.DataFrame) -> NDArray[np.float64]:
        keras: Any = models._import_keras()
        keras.utils.set_random_seed(common.SEED)

        train_rows = data_loader.coerce_features(train)
        valid_rows = data_loader.coerce_features(valid)
        numeric = Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
        ordinal = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=_UNKNOWN_INDEX,
            min_frequency=features.MIN_CATEGORY_FREQUENCY,
        )
        columns = list(schema.NUMERIC_FEATURES)
        train_numbers = numeric.fit_transform(train_rows[columns]).astype("float32")
        valid_numbers = numeric.transform(valid_rows[columns]).astype("float32")
        # Index 0 is what the training rows have not seen.
        train_codes = ordinal.fit_transform(_categories(train_rows)).astype("int32") + 1
        valid_codes = ordinal.transform(_categories(valid_rows)).astype("int32") + 1

        numeric_input = keras.Input(shape=(len(columns),), name="numeric")
        inputs = [numeric_input]
        parts = [numeric_input]
        for position, name in enumerate(schema.CATEGORICAL_FEATURES):
            size = int(train_codes[:, position].max()) + 1
            category_input = keras.Input(shape=(1,), name=name, dtype="int32")
            embedded = keras.layers.Embedding(size, min(_MAX_EMBEDDING_SIZE, (size + 1) // 2))
            inputs.append(category_input)
            parts.append(keras.layers.Flatten()(embedded(category_input)))
        output = keras.layers.Concatenate()(parts)
        for units in hidden_layers:
            output = keras.layers.Dense(units, activation="relu")(output)
            output = keras.layers.Dropout(dropout, seed=common.SEED)(output)
        output = keras.layers.Dense(1, activation="sigmoid")(output)

        network = keras.Model(inputs, output)
        network.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
        )

        def as_inputs(numbers: NDArray[Any], codes: NDArray[Any]) -> dict[str, NDArray[Any]]:
            categories = {
                name: codes[:, position]
                for position, name in enumerate(schema.CATEGORICAL_FEATURES)
            }
            return {"numeric": numbers, **categories}

        network.fit(
            as_inputs(train_numbers, train_codes),
            train[schema.TARGET].to_numpy(dtype="float32"),
            epochs=epochs,
            batch_size=batch_size,
            validation_split=_VALIDATION_FRACTION,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=_PATIENCE, restore_best_weights=True
                )
            ],
            verbose=0,
        )
        predicted = network.predict(
            as_inputs(valid_numbers, valid_codes), batch_size=_PREDICT_BATCH_SIZE, verbose=0
        )
        return np.asarray(predicted.ravel(), dtype="float64")

    return fit_predict


def _tuned_mlp_params() -> dict[str, Any]:
    content = json.loads(config.get_settings().best_params_path.read_text("utf-8"))
    params = content["models"]["mlp"]["params"]
    return {key.removeprefix("model__"): value for key, value in params.items()}


def _one_hot_mlp(**params: Any) -> common.FitPredict:
    def build() -> models.KerasMLPClassifier:
        return models.KerasMLPClassifier(random_state=common.SEED, **params)

    return common.tabular_predictor(build, family=features.Family.SCALED)


def embeddings(development: pd.DataFrame) -> common.ExperimentResult:
    """Compare the MLP with one-hot, the MLP with embeddings and the reference model.

    Args:
        development: Development rows.

    Returns:
        The tuned MLP with one-hot as the reference of the gaps.
    """
    params = _tuned_mlp_params()
    params["hidden_layers"] = tuple(params["hidden_layers"])
    return common.compare(
        "embeddings",
        development,
        [
            common.Variant("mlp_one_hot", _one_hot_mlp(**params)),
            common.Variant("mlp_embeddings", embedding_mlp(**params)),
            common.Variant("xgboost_reference", common.tabular_predictor(common.reference_xgboost)),
        ],
    )
