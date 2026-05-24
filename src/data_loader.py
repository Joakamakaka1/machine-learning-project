"""
- Archivo encargado de:
- Cargar y procesar el dataset
- Separar variables independientes de la variable objetivo
- Crear pipeline de Scikit-learn para preprocesamiento
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import COLS_TO_DROP, RAW_DATA_PATH, TARGET_COL


# Función para cargar los datos del dataset desde la ruta especificada
def load_data(file_path=RAW_DATA_PATH):
    df = pd.read_csv(file_path)
    return df


# Función para:
# - Limpiar los datos eliminando columnas irrelevantes o que pueden causar fuga de datos
# - Preprocesar los datos dividiéndolos en conjuntos de entrenamiento y prueba
# - Definir el procesador
def preprocess_and_split(df):
    # 1: Eliminar columnas irrelevantes o con fuga de datos
    df_cleaned = df.drop(columns=[col for col in COLS_TO_DROP if col in df.columns])

    # 2: Separar variables independientes (X) de la variable objetivo (y)
    X = df_cleaned.drop(columns=[TARGET_COL])
    y = df_cleaned[TARGET_COL]

    # 3: Dividir el dataset en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4: Identificar columnas
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # 5: Pipeline para variables numéricas
    num_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    # Pipeline para variables categóricas
    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # 6: Combinar pipelines en un ColumnTransformer
    preprocessor = ColumnTransformer(
        [("num", num_pipeline, num_cols), ("cat", cat_pipeline, cat_cols)]
    )

    return X_train, X_test, y_train, y_test, preprocessor
