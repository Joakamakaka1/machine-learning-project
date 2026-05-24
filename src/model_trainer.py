"""
- Archivo que gestionará:
- El entrenamiento de los modelos tradicionales y de la red neuronal de Keras
- Encapsulará todo en Pipelines de scikit-learn
"""

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras
from xgboost import XGBClassifier

from src.config import MODELS_DIR


# Funcion para construir el modelo de red neuronal con Keras
def build_keras_model(input_dim):
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# Clase para gestionar:
# - El entrenamiento de modelos tradicionales
# - La red neuronal de Keras
class ModelTrainer:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.models = {}

    def train_classical_models(self, X_train, y_train):
        algorithms = {
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
            "decision_tree": DecisionTreeClassifier(max_depth=10, random_state=42),
            "random_forest": RandomForestClassifier(
                n_estimators=100, max_depth=12, random_state=42
            ),
            "xgboost": XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
            ),
        }

        for name, clf in algorithms.items():
            print(f"Entrenando modelo: {name}...")
            # Creamos un pipeline que une el preprocesamiento y el modelo
            pipeline = Pipeline([("preprocessor", self.preprocessor), ("classifier", clf)])
            pipeline.fit(X_train, y_train)
            self.models[name] = pipeline
            # Guardamos el pipeline entrenado
            joblib.dump(pipeline, MODELS_DIR / f"{name}.pkl")

    def train_neural_network(self, X_train, y_train, epochs=10, batch_size=64):
        print("Entrenando Red Neuronal de Keras...")
        # Transformamos los datos de entrenamiento usando el preprocesador
        # El preprocessor ya fue ajustado en train_classical_models (main.py)
        X_train_processed = self.preprocessor.transform(X_train)
        input_dim = X_train_processed.shape[1]

        keras_clf = build_keras_model(input_dim)

        # Entrenamiento con .fit de Keras
        keras_clf.fit(
            X_train_processed,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1,
        )

        # Guardamos preprocesador y modelo Keras
        self.models["neural_network"] = {
            "preprocessor": self.preprocessor,
            "keras_model": keras_clf,
        }

        # Serializar componentes
        joblib.dump(self.preprocessor, MODELS_DIR / "nn_preprocessor.pkl")
        keras_clf.save(MODELS_DIR / "neural_network.keras")
