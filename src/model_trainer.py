"""Módulo de entrenamiento, optimización y serialización de modelos predictivos.

Integra clasificadores scikit-learn y redes neuronales Keras en pipelines
homogéneos, protegiendo contra fugas de información y facilitando la persistencia.
"""

import json
import logging

import joblib
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from xgboost import XGBClassifier

from src.config import MODELS_DIR, NN_BATCH_SIZE, NN_EPOCHS, OUTPUTS_DIR, RANDOM_STATE

logger = logging.getLogger(__name__)


class KerasMLPClassifier(ClassifierMixin, BaseEstimator):
    """Adaptador de red neuronal Keras para compatibilidad con la API de scikit-learn.

    Permite integrar Keras en Pipelines de sklearn estándar.
    """

    def __init__(
        self,
        epochs=10,
        batch_size=64,
        learning_rate=1e-3,
        validation_split=0.1,
        random_state=RANDOM_STATE,
        verbose=0,
    ):
        """Inicializa el clasificador con sus hiperparámetros."""
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.random_state = random_state
        self.verbose = verbose

    def _build_model(self, n_features):
        """Construye y compila la arquitectura MLP en Keras."""
        tf.keras.utils.set_random_seed(self.random_state)

        model = keras.Sequential(
            [
                keras.layers.Input(shape=(n_features,)),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, X, y):
        """Entrena el modelo de Keras calculando pesos de clase para balanceo."""
        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32")

        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        self.model_ = self._build_model(self.n_features_in_)

        # Parada temprana por pérdida de validación
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )

        # Ajuste por desbalanceo de clases
        class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

        self.model_.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=[early_stopping],
            class_weight=class_weight_dict,
            verbose=self.verbose,
        )
        return self

    def predict_proba(self, X):
        """Calcula probabilidades para cada clase."""
        X = np.asarray(X, dtype="float32")
        proba = self.model_.predict(X, verbose=0).flatten()
        return np.column_stack([1 - proba, proba])

    def predict(self, X):
        """Predice etiquetas binarias usando umbral 0.5."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _build_algorithms(nn_epochs, nn_batch_size):
    """Devuelve las instancias de los 5 clasificadores configurados."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=700,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            colsample_bylevel=0.8,
            tree_method="hist",
            scale_pos_weight=1.7,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "neural_network": KerasMLPClassifier(
            epochs=nn_epochs, batch_size=nn_batch_size, random_state=RANDOM_STATE
        ),
    }


def _tune_xgboost(preprocessor, clf, X_train, y_train):
    """Optimiza XGBoost usando RandomizedSearchCV con validación cruzada."""
    n_iter = 30
    cv_folds = 3
    total_fits = n_iter * cv_folds

    logger.info(
        "Iniciando RandomizedSearchCV: %d iteraciones x %d folds = %d fits totales",
        n_iter,
        cv_folds,
        total_fits,
    )

    param_dist = {
        "classifier__n_estimators": [300, 500, 700],
        "classifier__max_depth": [4, 6, 8, 10],
        "classifier__learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
        "classifier__subsample": [0.6, 0.7, 0.8, 0.9],
        "classifier__colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "classifier__colsample_bylevel": [0.6, 0.8, 1.0],
        "classifier__min_child_weight": [1, 3, 5, 7],
        "classifier__gamma": [0, 0.05, 0.1, 0.3, 0.5],
        "classifier__reg_alpha": [0, 0.01, 0.1, 0.5, 1.0],
        "classifier__reg_lambda": [1, 1.5, 2, 3],
    }

    base_pipeline = Pipeline([("preprocessor", clone(preprocessor)), ("classifier", clone(clf))])

    search = RandomizedSearchCV(
        base_pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv_folds,
        scoring="roc_auc",
        n_jobs=1,
        random_state=RANDOM_STATE,
        verbose=1,
        refit=True,
    )

    search.fit(X_train, y_train)

    best_params = {k.replace("classifier__", ""): v for k, v in search.best_params_.items()}
    logger.info("Mejores parametros encontrados: %s", best_params)
    logger.info("Mejor ROC-AUC CV: %.4f", search.best_score_)

    with open(OUTPUTS_DIR / "best_hiperparametros.json", "w") as f:
        json.dump(best_params, f, indent=2)

    return search.best_estimator_


def train_all_models(
    preprocessor,
    X_train,
    y_train,
    nn_epochs=NN_EPOCHS,
    nn_batch_size=NN_BATCH_SIZE,
    tune_xgboost=False,
):
    """Entrena los 5 clasificadores en Pipelines independientes y los guarda en disco."""
    algorithms = _build_algorithms(nn_epochs, nn_batch_size)
    trained = {}

    for name, clf in algorithms.items():
        logger.info("Entrenando modelo: %s", name)

        if name == "xgboost" and tune_xgboost:
            pipeline = _tune_xgboost(preprocessor, clf, X_train, y_train)
        else:
            pipeline = Pipeline([("preprocessor", clone(preprocessor)), ("classifier", clf)])
            pipeline.fit(X_train, y_train)

        trained[name] = pipeline

        joblib.dump(pipeline, MODELS_DIR / f"{name}.pkl")
        logger.info("Guardado en %s", MODELS_DIR / f"{name}.pkl")

    return trained
