"""
- Archivo encargado de:
- Cargar y procesar el dataset
- Separar variables independientes de la variable objetivo
- Crear pipeline de Scikit-learn para preprocesamiento
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from src.config import (
    CAT_COLS,
    COLS_TO_DROP,
    HIGH_CARD_CAT_COLS,
    NA_TOKENS,
    NUM_COLS,
    RAW_DATA_PATH,
    TARGET_COL,
)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Transformer sklearn que añade features derivadas y normaliza columnas especiales.

    Al ser un transformer estándar (fit/transform), se integra como primer paso
    en el Pipeline de preprocesamiento, garantizando que las features derivadas
    se calculen correctamente dentro de cada fold de la validación cruzada.

    Features creadas:
    - total_nights: noches totales de estancia.
    - total_guests: total de huéspedes (mínimo 1).
    - adr_per_night: precio efectivo por noche.
    - room_type_changed: si la habitación asignada difiere de la reservada.
    - previous_cancel_rate: tasa histórica de cancelación del cliente.
    - is_new_guest: cliente sin ningún historial previo.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()

        df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
        df["total_guests"] = (df["adults"] + df["children"].fillna(0) + df["babies"]).clip(lower=1)
        df["adr_per_night"] = np.where(
            df["total_nights"] > 0, df["adr"] / df["total_nights"], df["adr"]
        )
        df["room_type_changed"] = (
            df["reserved_room_type"] != df["assigned_room_type"]
        ).astype(int)
        df["previous_cancel_rate"] = df["previous_cancellations"] / (
            df["previous_cancellations"] + df["previous_bookings_not_canceled"] + 1
        )
        df["is_new_guest"] = (
            (df["is_repeated_guest"] == 0)
            & (df["previous_cancellations"] == 0)
            & (df["previous_bookings_not_canceled"] == 0)
        ).astype(int)

        # agent es un ID de agencia sin significado ordinal: se trata como categoría.
        # Los NaN se convierten a "Unknown" antes de que lleguen al TargetEncoder.
        if "agent" in df.columns:
            agent_series = df["agent"].astype("Int64").astype("object")
            df["agent"] = agent_series.where(agent_series.notna(), other="Unknown").astype(str)

        return df


def load_data(file_path=RAW_DATA_PATH):
    """Carga el dataset desde disco normalizando tokens nulos al estándar NaN de pandas."""
    return pd.read_csv(file_path, na_values=NA_TOKENS, keep_default_na=True)


def preprocess_and_split(df):
    """Limpia el dataframe, lo divide en train/test y construye el preprocessor sklearn.

    El preprocessor es un Pipeline completo:
      FeatureEngineer → ColumnTransformer (num | cat_low | cat_high)

    Encapsular todo en un Pipeline garantiza que las transformaciones se apliquen
    correctamente dentro de cada fold de la validación cruzada, evitando data leakage.

    Returns:
        X_train, X_test, y_train, y_test, preprocessor
    """
    # 1: Eliminar columnas con fuga de datos o irrelevantes
    df_cleaned = df.drop(columns=[col for col in COLS_TO_DROP if col in df.columns])

    # 2: Separar variables independientes (X) de la variable objetivo (y)
    X = df_cleaned.drop(columns=[TARGET_COL])
    y = df_cleaned[TARGET_COL]

    # 3: Dividir en train/test con estratificación para mantener la proporción de clases
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4: Filtrar solo las columnas presentes en el dataset para cada grupo
    num_cols = [c for c in NUM_COLS if c in X.columns]
    cat_low_cols = [c for c in CAT_COLS if c in X.columns]
    cat_high_cols = [c for c in HIGH_CARD_CAT_COLS if c in X.columns]

    # 5: Sub-pipelines por tipo de columna
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Baja cardinalidad (hotel, meal, deposit_type, etc.) → OneHotEncoder
    cat_low_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Alta cardinalidad (country ~180, agent ~300) → TargetEncoder.
    # Sustituye cada categoría por su probabilidad media de cancelación,
    # reduciendo >400 columnas OHE a solo 2 columnas densas y más informativas.
    cat_high_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("target_enc", TargetEncoder(random_state=42)),
    ])

    # 6: Combinar los tres sub-pipelines en un ColumnTransformer
    col_transformer = ColumnTransformer(
        [
            ("num", num_pipeline, num_cols),
            ("cat_low", cat_low_pipeline, cat_low_cols),
            ("cat_high", cat_high_pipeline, cat_high_cols),
        ],
        remainder="drop",
    )

    # 7: Pipeline completo: ingeniería de features → transformación de columnas
    preprocessor = Pipeline([
        ("features", FeatureEngineer()),
        ("col_transform", col_transformer),
    ])

    return X_train, X_test, y_train, y_test, preprocessor

