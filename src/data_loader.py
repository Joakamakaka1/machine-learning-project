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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CAT_COLS,
    COLS_TO_DROP,
    NA_TOKENS,
    NUM_COLS,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COL,
    TEST_SIZE,
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
    - previous_cancel_rate: tasa histórica de cancelación del cliente.
    - is_new_guest: cliente sin ningún historial previo.
    - is_domestic: 1 si el país es Portugal (mercado doméstico con ~57% cancelación).
    - is_agent_booking: 1 si la reserva fue gestionada por agencia (~39% vs 25%).
    - guest_composition: perfil del huésped (solo/couple/family/group/other).
    - stay_category: tipo de estancia (weekend_only/week_only/mixed/no_stay).
    - log_lead_time: log1p(lead_time) para suavizar la cola de outliers.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()

        # --- Cálculo de las 10 features derivadas ---
        df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
        df["total_guests"] = (df["adults"] + df["children"].fillna(0) + df["babies"]).clip(lower=1)
        df["adr_per_night"] = np.where(
            df["total_nights"] > 0, df["adr"] / df["total_nights"], df["adr"]
        )
        df["previous_cancel_rate"] = df["previous_cancellations"] / (
            df["previous_cancellations"] + df["previous_bookings_not_canceled"] + 1
        )
        df["is_new_guest"] = (
            (df["is_repeated_guest"] == 0)
            & (df["previous_cancellations"] == 0)
            & (df["previous_bookings_not_canceled"] == 0)
        ).astype(int)

        # is_domestic: huéspedes portugueses cancelan mucho más (~57% vs ~24%)
        df["is_domestic"] = (df["country"] == "PRT").astype(int)

        # is_agent_booking: reservas con agencia cancelan más (~39% vs ~25%)
        df["is_agent_booking"] = df["agent"].notna().astype(int)

        # guest_composition: perfil socio-demográfico de la reserva
        df["guest_composition"] = "other"
        children_filled = df["children"].fillna(0)
        mask_solo = (df["adults"] == 1) & (children_filled == 0) & (df["babies"] == 0)
        mask_couple = (df["adults"] == 2) & (children_filled == 0) & (df["babies"] == 0)
        mask_group = (df["adults"] > 2) & (children_filled == 0) & (df["babies"] == 0)
        mask_family = (children_filled > 0) | (df["babies"] > 0)
        df.loc[mask_solo, "guest_composition"] = "solo"
        df.loc[mask_couple, "guest_composition"] = "couple"
        df.loc[mask_group, "guest_composition"] = "group"
        df.loc[mask_family, "guest_composition"] = "family"

        # stay_category: tipo de estancia (ocio fin de semana vs negocios)
        df["stay_category"] = "no_stay"
        wknd = df["stays_in_weekend_nights"] > 0
        week = df["stays_in_week_nights"] > 0
        df.loc[wknd & ~week, "stay_category"] = "weekend_only"
        df.loc[~wknd & week, "stay_category"] = "week_only"
        df.loc[wknd & week, "stay_category"] = "mixed"

        # log_lead_time: suaviza la distribución sesgada de lead_time
        df["log_lead_time"] = np.log1p(df["lead_time"])

        return df


def load_data(file_path=RAW_DATA_PATH):
    """Carga el dataset desde disco normalizando tokens nulos al estándar NaN de pandas."""
    return pd.read_csv(file_path, na_values=NA_TOKENS, keep_default_na=True)


class CastToString(BaseEstimator, TransformerMixin):
    """Convierte todas las columnas a str para uniformar los tipos antes del OneHotEncoder.

    Necesario cuando una columna categórica (e.g. agent) contiene valores numéricos
    (float/int) mezclados con la cadena 'Unknown' introducida por el SimpleImputer.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "astype"):
            return X.astype(str)
        return np.array(X, dtype=str)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features, dtype=object)
        return np.array([], dtype=object)


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
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 4: Definir las columnas numéricas y categóricas para el ColumnTransformer.
    # Dado que FeatureEngineer se ejecuta primero, todas las columnas de NUM_COLS y CAT_COLS
    # estarán presentes en el flujo antes del ColumnTransformer.
    num_cols = NUM_COLS
    cat_cols = CAT_COLS

    # 5: Sub-pipelines por tipo de columna
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categóricas (hotel, meal, country, agent, etc.) → OneHotEncoder
    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("cast_str", CastToString()),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # 6: Combinar los sub-pipelines en un ColumnTransformer
    col_transformer = ColumnTransformer(
        [
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
    )

    # 7: Pipeline completo: ingeniería de features → transformación de columnas
    preprocessor = Pipeline(
        [
            ("features", FeatureEngineer()),
            ("col_transform", col_transformer),
        ]
    )

    return X_train, X_test, y_train, y_test, preprocessor
