"""
- Archivo que gestionará:
- El entrenamiento de los modelos tradicionales y de la red neuronal de Keras
- Encapsulará todo en Pipelines de scikit-learn, incluyendo la red neuronal,
  para que compita en igualdad de condiciones con el resto de modelos.
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

from src import config
from src.config import MODELS_DIR
from utils.tuning_utils import _TuningProgressPipeline

logger = logging.getLogger(__name__)


class KerasMLPClassifier(ClassifierMixin, BaseEstimator):
    """Wrapper de la red neuronal Keras con interfaz compatible con scikit-learn.

    Implementa fit, predict y predict_proba, integrándose en un pipeline de sklearn
    exactamente igual que los modelos clásicos. Esto permite:
    - Serializar el pipeline completo (preprocesado + red) en un único .pkl.
    - Evaluar la red con el mismo código que el resto de modelos.
    - Incluirla en la selección del mejor modelo.

    El preprocesado lo aplica el Pipeline externo; este wrapper recibe los datos
    ya transformados (matriz de floats).
    """
    def __init__(
        self,
        epochs=10,
        batch_size=64,
        learning_rate=1e-3,
        validation_split=0.1,
        random_state=config.RANDOM_STATE,
        verbose=0,
    ):
        """
        Inicializa el clasificador con los hiperparámetros de la red neuronal
        con valores por defecto razonables.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.random_state = random_state
        self.verbose = verbose

    def _build_model(self, n_features):
        """Construye la arquitectura de la red neuronal.

        Arquitectura ampliada con BatchNormalization para estabilizar
        el entrenamiento, siguiendo el patrón visto en clase 4.
        """

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
        """Entrena la red neuronal con los datos de entrenamiento."""
        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32")
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        self.model_ = self._build_model(self.n_features_in_)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )
        # Compensar desbalanceo de clases (~37% cancelaciones vs ~63% no canceladas)
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
        """Devuelve probabilidades de clase para cada muestra."""
        X = np.asarray(X, dtype="float32")
        proba = self.model_.predict(X, verbose=0).flatten()
        return np.column_stack([1 - proba, proba])

    def predict(self, X):
        """Predice la clase de cada muestra usando un umbral de 0.5."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class ModelTrainer:
    """Gestiona el entrenamiento de todos los modelos de forma uniforme.

    Los modelos se envuelven en un Pipeline sklearn, lo que garantiza competición
    en igualdad de condiciones y serialización homogénea (un único .pkl por modelo).
    """
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.models = {}

    def _tune_xgboost(self, clf, X_train, y_train):
        """Búsqueda aleatoria de hiperparámetros para XGBoost con validación cruzada.

        Entrena el pipeline completo (preprocesado + clasificador) en cada fold para
        evitar fuga de datos y guarda el mejor modelo y sus parámetros en outputs/.

        Paralelismo: n_jobs=1 en RSCV (fits secuenciales) + n_jobs=-1 en XGBoost
        (paralelismo interno vía OpenMP por fit). Esto evita el oversubscription de
        CPU que ocurre cuando RSCV lanza procesos en paralelo y cada uno intenta
        usar todos los cores, causando que el pipeline nunca termine.
        """
        n_iter = 30
        cv_folds = 3
        total_fits = n_iter * cv_folds
        logger.info(
            "Iniciando RandomizedSearchCV: %d iteraciones x %d folds = %d fits totales",
            n_iter,
            cv_folds,
            total_fits,
        )
        logger.info(
            "Estrategia: n_jobs=1 en RSCV (secuencial) + n_jobs=-1 en XGBoost (OpenMP por fit)"
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
        # Clonar el clasificador para no mutar el original; tree_method='hist' y
        # n_jobs=-1 se conservan desde la instancia base definida en train_all_models.
        _TuningProgressPipeline.reset(total_fits)
        base_pipeline = _TuningProgressPipeline(
            [("preprocessor", self.preprocessor), ("classifier", clone(clf))]
        )
        search = RandomizedSearchCV(
            base_pipeline,
            param_dist,
            n_iter=n_iter,
            cv=cv_folds,
            scoring="roc_auc",
            n_jobs=1,
            random_state=config.RANDOM_STATE,
            verbose=0,
            refit=True,
        )
        search.fit(X_train, y_train)
        best_params = {k.replace("classifier__", ""): v for k, v in search.best_params_.items()}
        logger.info("Mejores parametros encontrados: %s", best_params)
        logger.info("Mejor ROC-AUC CV: %.4f", search.best_score_)
        with open(MODELS_DIR.parent / "outputs" / "best_hiperparametros.json", "w") as f:
            json.dump(best_params, f, indent=2)
        return search.best_estimator_

    def _build_algorithms(self, nn_epochs, nn_batch_size):
        """Devuelve el diccionario de algoritmos configurados listos para entrenar.

        Separar la configuración del bucle de entrenamiento sigue el principio de
        responsabilidad única: este método solo sabe QUÉ modelos entrenar y con qué
        parámetros; train_all_models sabe CÓMO iterar y serializar.
        """
        return {
            # class_weight='balanced' compensa el desbalance (~37% cancelaciones)
            # en los tres modelos clásicos, igual que scale_pos_weight en XGBoost
            # y class_weight_dict en la red neuronal.
            "logistic_regression": LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=config.RANDOM_STATE
            ),
            "decision_tree": DecisionTreeClassifier(
                max_depth=12,
                min_samples_leaf=50,
                class_weight="balanced",
                random_state=config.RANDOM_STATE,
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_leaf=20,
                class_weight="balanced",
                n_jobs=-1,
                random_state=config.RANDOM_STATE,
            ),
            # tree_method='hist' acelera el entrenamiento; scale_pos_weight compensa
            # el desbalance de clases (~63% no cancela / ~37% cancela ≈ 1.7).
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
                random_state=config.RANDOM_STATE,
            ),
            "neural_network": KerasMLPClassifier(
                epochs=nn_epochs, batch_size=nn_batch_size, random_state=config.RANDOM_STATE
            ),
        }

    def train_all_models(
        self,
        X_train,
        y_train,
        nn_epochs=config.NN_EPOCHS,
        nn_batch_size=config.NN_BATCH_SIZE,
        tune_xgboost=False,
    ):
        """Entrena los 5 modelos requeridos por el enunciado.

        Todos se envuelven en un Pipeline sklearn (preprocessor + clasificador),
        garantizando evaluación homógenea y ausencia de fuga de datos.
        Cada pipeline recibe su propio clone del preprocessor para evitar
        que el fit de un modelo sobreescriba el estado del objeto compartido.
        """
        algorithms = self._build_algorithms(nn_epochs, nn_batch_size)

        for name, clf in algorithms.items():
            logger.info("Entrenando modelo: %s", name)

            # Para XGBoost: opción de realizar búsqueda aleatoria de hiperparámetros.
            if name == "xgboost" and tune_xgboost:
                best_pipeline = self._tune_xgboost(clf, X_train, y_train)
                self.models[name] = best_pipeline
                joblib.dump(best_pipeline, MODELS_DIR / f"{name}.pkl")
                logger.info("Guardado en %s", MODELS_DIR / f"{name}.pkl")
                continue

            # clone() crea una copia sin estado del preprocessor para que cada
            # pipeline sea independiente y el fit no mute el objeto compartido.
            pipeline = Pipeline([("preprocessor", clone(self.preprocessor)), ("classifier", clf)])
            pipeline.fit(X_train, y_train)
            self.models[name] = pipeline
            joblib.dump(pipeline, MODELS_DIR / f"{name}.pkl")
            logger.info("Guardado en %s", MODELS_DIR / f"{name}.pkl")
