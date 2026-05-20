# Guía técnica para la entrega final — Clasificación Binaria

> **Referencia al enunciado:** `enunciado-pontIA.md`
> **Fuentes de teoría:** `apuntes_clase_1.md` a `apuntes_clase_5.md` + `apuntes_extras.md`

---

## Índice

0. [Dataset: Hotel Bookings](#dataset-hotel-bookings)
1. [Mapa de la entrega](#1-mapa-de-la-entrega)
2. [Estructura de carpetas](#2-estructura-de-carpetas)
3. [EDA — Análisis exploratorio](#3-eda--análisis-exploratorio)
4. [Módulo: `data_loader.py`](#4-módulo-data_loaderpy)
5. [Módulo: `model_trainer.py`](#5-módulo-model_trainerpy)
6. [Red Neuronal con Keras](#6-red-neuronal-con-keras-requisito-obligatorio)
7. [Módulo: `evaluator.py`](#7-módulo-evaluatorpy)
8. [Módulo: `predictor.py`](#8-módulo-predictorpy)
9. [Módulo: `config.py`](#9-módulo-configpy)
10. [Pipeline sklearn](#10-pipeline-sklearn)
11. [Bonus técnicos](#11-bonus-técnicos)
12. [Checklist de entregables](#12-checklist-de-entregables)

---

## Dataset: Hotel Bookings

**Fichero:** `dataset_practica_final.csv` | **Target:** `is_canceled` — 1 = reserva cancelada, 0 = reserva completada

> ~37% de cancelaciones → desbalanceo moderado. Usar `stratify=y` en el split y **ROC-AUC** como métrica principal.

---

### ⚠️ DATA LEAKAGE — columnas que DEBEN eliminarse ANTES de entrenar

| Columna                   | Por qué es leakage                                                                                                                                                                       |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `reservation_status`      | Encoda directamente el target: `Canceled` = cancelada, `Check-Out` = completada, `No-Show` = no presentado. Si se incluye, cualquier modelo obtiene ~100% de AUC sin aprender nada real. |
| `reservation_status_date` | Fecha en que se actualizó el estado → también codifica el outcome.                                                                                                                       |

```python
# ¡Primero de todo al cargar el CSV!
LEAKAGE_COLS = ['reservation_status', 'reservation_status_date']
df = df.drop(columns=LEAKAGE_COLS)
```

> Si no se eliminan, los modelos tendrán un AUC artificialmente perfecto y la práctica no tendrá validez.

---

### ⚠️ NULL como string

Los valores nulos en `country`, `agent` y `company` están escritos literalmente como `"NULL"` en el CSV, no como NaN real de pandas. **Hay que convertirlos al leer el fichero:**

```python
df = pd.read_csv(path, na_values=['NULL'])   # convierte "NULL" → NaN automáticamente
```

---

### Clasificación de columnas

**Numéricas** (→ `SimpleImputer(strategy='median')` + `StandardScaler`):

```python
NUMERIC_COLS = [
    'lead_time', 'arrival_date_year', 'arrival_date_week_number',
    'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights',
    'adults', 'children', 'babies', 'is_repeated_guest',
    'previous_cancellations', 'previous_bookings_not_canceled',
    'booking_changes', 'days_in_waiting_list', 'adr',
    'required_car_parking_spaces', 'total_of_special_requests',
    # Columnas de feature engineering (añadir después de la función de ingeniería)
    'total_nights', 'has_children', 'room_was_changed', 'has_agent', 'has_company'
]
```

**Categóricas de baja/media cardinalidad** (→ `OneHotEncoder(drop='first', handle_unknown='ignore')`):

```python
CAT_LOW_COLS = [
    'hotel',                  # 2 valores: City Hotel, Resort Hotel
    'arrival_date_month',     # 12 meses (nombres → OHE los convierte a dummies)
    'meal',                   # 5 valores: BB, HB, FB, SC, Undefined
    'market_segment',         # ~7 valores: Online TA, Offline TA/TO, Direct, Corporate...
    'distribution_channel',   # ~5 valores: TA/TO, Direct, Corporate, GDS...
    'deposit_type',           # 3 valores: No Deposit, Refundable, Non Refund
    'customer_type',          # 4 valores: Transient, Contract, Transient-Party, Group
    'reserved_room_type',     # ~8 letras (A–H)
    'assigned_room_type',     # ~9 letras (A–I)
    'country_grouped',        # resultado del agrupamiento de 'country' (ver abajo)
]
```

**Alta cardinalidad — NO incluir directamente en OHE:**

| Columna   | Problema                | Solución recomendada                                                             |
| --------- | ----------------------- | -------------------------------------------------------------------------------- |
| `country` | ~170 países únicos      | Agrupar top-20 más frecuentes + resto como `'Other'` → columna `country_grouped` |
| `agent`   | ~300+ IDs + muchos NaN  | Binarizar: `has_agent = agent.notna().astype(int)`                               |
| `company` | ~150+ IDs + mayoría NaN | Binarizar: `has_company = company.notna().astype(int)`                           |

---

### Feature engineering recomendado

Crear estas columnas **antes** de construir el `ColumnTransformer`:

```python
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Total de noches de estancia
    df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']

    # ¿La reserva incluye niños o bebés?
    df['has_children'] = ((df['children'].fillna(0) + df['babies'].fillna(0)) > 0).astype(int)

    # ¿Se asignó una habitación diferente a la reservada?
    df['room_was_changed'] = (df['reserved_room_type'] != df['assigned_room_type']).astype(int)

    # Flags binarios para agent y company (en lugar de IDs de alta cardinalidad)
    df['has_agent']   = df['agent'].notna().astype(int)
    df['has_company'] = df['company'].notna().astype(int)

    # Agrupación de country: top-20 más frecuentes + 'Other'
    top_countries = df['country'].value_counts().nlargest(20).index
    df['country_grouped'] = df['country'].where(df['country'].isin(top_countries), other='Other')

    # Eliminar columnas originales de alta cardinalidad
    df = df.drop(columns=['agent', 'company', 'country'])

    return df
```

> Llamar a `feature_engineering(df)` justo después de eliminar las columnas de leakage y antes del split.

---

### Justificación de la métrica principal

| Métrica     | ¿Válida?          | Por qué                                                                                              |
| ----------- | ----------------- | ---------------------------------------------------------------------------------------------------- |
| Accuracy    | ⚠️ Cuidado        | Con 63% de clase 0, predecir siempre "no cancelado" ya da 63% de accuracy sin aprender nada          |
| F1-Score    | ✅ Buena          | Balance entre precision y recall; útil con umbral fijo en 0.5                                        |
| **ROC-AUC** | ✅✅ **Mejor**    | Mide la capacidad discriminativa independientemente del umbral; robusta ante el desbalanceo moderado |
| Recall      | ✅ Complementaria | Detectar todas las cancelaciones (evitar FN) es crítico para el hotel                                |

**Argumento para el informe:** En el sector hotelero, no detectar una cancelación (falso negativo) supone perder ingresos que podrían recuperarse con overbooking controlado, mientras que predecir falsas cancelaciones (falsos positivos) conlleva un coste moderado. Con ~37% de cancelaciones, ROC-AUC es la métrica más objetiva para comparar modelos y F1-score sirve como métrica de operación final a umbral fijo.

---

## 1. Mapa de la entrega

### Mínimos obligatorios (100%)

| Requisito                         | Qué hacer                                                                |
| --------------------------------- | ------------------------------------------------------------------------ |
| Dataset binario                   | Usar el CSV proporcionado (variable target 0/1)                          |
| 5 modelos                         | LogReg, DecisionTree, RandomForest, XGBoost/LightGBM, Red Neuronal Keras |
| Métrica principal + justificación | Elegir 1 de: accuracy, precision, recall, F1, AUC-ROC                    |
| Visualizaciones                   | Matriz de confusión + Curva ROC de **todos** los modelos                 |
| Pipeline estructurado             | Carga → Preprocesado → Entrenamiento → Evaluación → Selección            |
| Código modularizado               | `src/` con scripts separados, no solo notebook                           |
| README completo                   | Autores, descripción, instrucciones de entorno, resultados               |
| `requirements.txt`                | Todas las dependencias con versiones                                     |
| `.gitignore`                      | No subir modelos .pkl, datos pesados, `.env`, `__pycache__`              |
| Definición de roles               | Quién hace qué (en el PDF o README)                                      |

### Bonus opcionales (+20% máx)

| Bonus                                          | Complejidad         | Recomendado                    |
| ---------------------------------------------- | ------------------- | ------------------------------ |
| GridSearchCV / RandomizedSearchCV              | Baja                | ✅ Sí (fácil de añadir)        |
| MLflow experiment tracking                     | Media               | ✅ Sí (vista en clase)         |
| FastAPI (`/train`, `/predict`, `/evaluate`)    | Media-Alta          | Si hay tiempo                  |
| Streamlit / Gradio UI                          | Media               | Si hay tiempo                  |
| Embeddings (TF-IDF, Word2Vec, cat. embeddings) | Depende del dataset | Solo si el dataset tiene texto |

---

## 2. Estructura de carpetas

```
proyecto-final-ml/
├── .gitignore
├── README.md
├── data/
│   ├── raw/           ← CSV original sin tocar
│   └── processed/     ← datos limpios listos para modelar (opcional)
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── neural_network.pkl   ← o .h5 / SavedModel
│   └── best_model.pkl
├── notebooks/
│   ├── exploracion/
│   │   └── eda_inicial.ipynb
│   └── finales/
│       ├── eda_final.ipynb
│       └── comparativa_modelos.ipynb
├── outputs/               ← gráficos PNG, HTML
│   ├── confusion_matrix.png
│   └── roc_curves.png
└── src/
    ├── __init__.py
    ├── config.py
    ├── data_loader.py
    ├── model_trainer.py
    ├── evaluator.py
    └── predictor.py
```

---

## 3. EDA — Análisis exploratorio

El EDA va en `notebooks/exploracion/eda_inicial.ipynb` o `notebooks/finales/eda_final.ipynb`.
El informe exige un apartado de EDA; aquí está qué hacer:

### Inspección básica

```python
df.head()
df.tail()
df.shape                    # (filas, columnas)
df.info()                   # tipos y nulos
df.describe().transpose()   # estadísticos descriptivos
df.isna().sum()             # nulos por columna
df.duplicated().sum()       # filas duplicadas
df['target'].value_counts(normalize=True)   # desbalanceo de clases
```

### Visualizaciones recomendadas

```python
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Distribución del target
sns.countplot(x='target', data=df)

# Distribución de cada variable numérica
df.hist(figsize=(14, 10), bins=30)

# Correlaciones (solo numéricas)
corr = df.select_dtypes('number').corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
# O con plotly
px.imshow(corr, text_auto=True, title='Mapa de correlaciones')

# Pairplot (con pocas features)
sns.pairplot(df, hue='target')

# Histograma de cada feature separado por clase
for col in numeric_cols:
    sns.histplot(data=df, x=col, hue='target', kde=True)
    plt.title(col)
    plt.show()
```

### Decisiones de diseño basadas en EDA

| Lo que ves            | Qué haces                                                         |
| --------------------- | ----------------------------------------------------------------- |
| Nulos                 | `SimpleImputer(strategy='mean')` o `'median'` o `'most_frequent'` |
| Variables categóricas | `OneHotEncoder` o `OrdinalEncoder` según si hay orden             |
| Escalas muy distintas | `StandardScaler` o `MinMaxScaler`                                 |
| Desbalanceo de clases | `stratify=y` en el split + elegir F1/Recall/AUC como métrica      |
| Columnas irrelevantes | `.drop(columns=[...])` antes de modelar                           |

---

### EDA específico para el dataset de Hotel Bookings

```python
# 0. Cargar con na_values=['NULL'] para tratar los "NULL" como NaN desde el inicio
df = pd.read_csv(PATH_DATA_RAW / "dataset_practica_final.csv", na_values=['NULL'])

# 1. Desbalanceo de clases (paso obligatorio antes de elegir métrica)
print(df['is_canceled'].value_counts())
print(df['is_canceled'].value_counts(normalize=True))
# Esperado: ~63% no cancelado (0), ~37% cancelado (1)

# 2. Nulos por columna (company y agent tendrán muchos)
df.isna().sum().sort_values(ascending=False).head(10)

# 3. Distribución de lead_time según cancelación
# Las reservas con mucha antelación tienden a cancelarse más
sns.histplot(data=df, x='lead_time', hue='is_canceled', kde=True, bins=50)
plt.title('Días de antelación según si la reserva fue cancelada')

# 4. Tasa de cancelación por tipo de hotel
df.groupby('hotel')['is_canceled'].mean().plot(kind='bar', title='Tasa de cancelación por tipo de hotel')

# 5. deposit_type es una de las columnas más informativas
df.groupby('deposit_type')['is_canceled'].mean().sort_values().plot(
    kind='bar', title='Tasa de cancelación por tipo de depósito'
)
# 'Non Refund' suele tener una tasa de cancelación artificialmente alta

# 6. Correlación de variables numéricas con is_canceled
corr_target = df[NUMERIC_COLS + ['is_canceled']].corr()['is_canceled'].drop('is_canceled')
corr_target.abs().sort_values(ascending=False).plot(kind='bar')
plt.title('Correlación absoluta con is_canceled')

# 7. Visualizar proporción de cancelaciones por market_segment
df.groupby('market_segment')['is_canceled'].mean().sort_values(ascending=False).plot(
    kind='bar', title='Tasa de cancelación por canal de marketing'
)
```

---

## 4. Módulo: `data_loader.py`

Responsabilidad: cargar el CSV, limpiar, codificar y devolver `X_train, X_test, y_train, y_test` (ya preprocesados y escalados si aplica).

### Funciones a usar

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data(path: str) -> pd.DataFrame:
    # na_values=['NULL'] convierte los "NULL" literales de agent/company/country a NaN real
    df = pd.read_csv(path, na_values=['NULL'])
    df.columns = df.columns.str.lower().str.strip()  # normalizar nombres

    # 1. Eliminar leakage ANTES de cualquier otra cosa
    leakage_cols = ['reservation_status', 'reservation_status_date']
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])

    # 2. Feature engineering
    df['total_nights']    = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df['has_children']    = ((df['children'].fillna(0) + df['babies'].fillna(0)) > 0).astype(int)
    df['room_was_changed']= (df['reserved_room_type'] != df['assigned_room_type']).astype(int)
    df['has_agent']       = df['agent'].notna().astype(int)
    df['has_company']     = df['company'].notna().astype(int)

    top_countries = df['country'].value_counts().nlargest(20).index
    df['country_grouped'] = df['country'].where(df['country'].isin(top_countries), other='Other')

    # Eliminar columnas de alta cardinalidad que ya se procesaron
    df = df.drop(columns=['agent', 'company', 'country'], errors='ignore')

    return df


def split_data(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    # stratify=y → mantener proporción de clases en train y test


def build_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    Construye un ColumnTransformer que aplica:
    - Imputer + StandardScaler a columnas numéricas
    - Imputer + OneHotEncoder a columnas categóricas
    """
    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_pipe, numeric_cols),
        ('cat', categorical_pipe, categorical_cols)
    ])
    return preprocessor
```

> **Por qué `ColumnTransformer`:** permite aplicar transformaciones distintas a numéricos y categóricos en un solo paso, integrable en un `Pipeline` de sklearn.
> **Por qué `stratify=y`:** evita que el test tenga una distribución de clases diferente al train (crítico con datos desbalanceados).

---

## 5. Módulo: `model_trainer.py`

Responsabilidad: entrenar cada uno de los 5 modelos requeridos y guardarlos.

### Imports necesarios

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib
from pathlib import Path
```

### Los 5 modelos requeridos

```python
# 1. Regresión Logística
lr = LogisticRegression(max_iter=1000, random_state=42)

# 2. Árbol de Decisión
dt = DecisionTreeClassifier(max_depth=5, random_state=42)

# 3. Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)

# 4. XGBoost (Gradient Boosting)
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    eval_metric='logloss',
    random_state=42
)
# ⚠️ XGBoost no acepta columnas con dtype object → usar pd.get_dummies o OneHotEncoder

# 5. Red neuronal → ver sección 6
```

### Integrar el preprocesador en un Pipeline

```python
# Patrón recomendado: Pipeline = preprocesador + modelo
# Así el modelo guarda las transformaciones y las aplica automáticamente en inferencia

def build_pipeline(preprocessor, modelo):
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', modelo)
    ])

pipe_lr  = build_pipeline(preprocessor, lr)
pipe_dt  = build_pipeline(preprocessor, dt)
pipe_rf  = build_pipeline(preprocessor, rf)
pipe_xgb = build_pipeline(preprocessor, xgb)

# Entrenar
pipe_lr.fit(X_train, y_train)   # aplica preprocesado + entrena en una sola llamada
```

### Guardar modelos

```python
PATH_MODELS = Path("../models/")
PATH_MODELS.mkdir(parents=True, exist_ok=True)

def save_model(model, filename: str):
    with open(PATH_MODELS / filename, "wb") as f:
        joblib.dump(model, f)

save_model(pipe_lr,  "logistic_regression.pkl")
save_model(pipe_dt,  "decision_tree.pkl")
save_model(pipe_rf,  "random_forest.pkl")
save_model(pipe_xgb, "xgboost.pkl")
```

### Hiperparámetros (bonus — GridSearchCV)

```python
# En un Pipeline, los parámetros del modelo se acceden con nombre_paso__parametro
param_grid_rf = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [3, 5, 8, None],
    'model__min_samples_split': [2, 5, 10]
}

grid_rf = GridSearchCV(
    estimator=pipe_rf,
    param_grid=param_grid_rf,
    cv=5,
    scoring='roc_auc',    # o 'f1', 'recall', 'precision' según tu métrica principal
    n_jobs=-1,
    verbose=1
)
grid_rf.fit(X_train, y_train)

print(grid_rf.best_params_)
best_rf = grid_rf.best_estimator_   # Pipeline ya entrenado con los mejores params
```

> **`n_jobs=-1`** usa todos los cores disponibles → acelera la búsqueda considerablemente.
> **`scoring='roc_auc'`** → si hay desbalanceo, AUC es más informativo que accuracy.

---

## 6. Red Neuronal con Keras (requisito obligatorio)

La red neuronal **no puede meterse en un Pipeline de sklearn** fácilmente. Requiere su propio flujo.

### Preprocesado previo

```python
# El preprocesador se entrena sobre X_train y se aplica a X_test
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled  = preprocessor.transform(X_test)

n_features = X_train_scaled.shape[1]
```

### Arquitectura para clasificación binaria

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

modelo_nn = models.Sequential([
    layers.Input(shape=(n_features,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),               # evitar overfitting
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')   # clasificación binaria
])

modelo_nn.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

modelo_nn.summary()
```

### Entrenamiento con EarlyStopping

```python
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True   # recupera los pesos de la mejor época
)

history = modelo_nn.fit(
    X_train_scaled, y_train,
    epochs=200,
    validation_split=0.2,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)
```

### Predicción con umbral

```python
y_pred_prob_nn = modelo_nn.predict(X_test_scaled).flatten()
y_pred_nn = (y_pred_prob_nn >= 0.5).astype(int)
```

### Visualizar curvas de aprendizaje (obligatorio)

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy'); plt.legend()

plt.tight_layout()
plt.savefig('../outputs/learning_curves_nn.png')
plt.show()
```

### Guardar la red neuronal

```python
# Opción 1: formato nativo TF (recomendado)
modelo_nn.save("../models/neural_network.keras")

# Carga posterior
from tensorflow.keras.models import load_model
modelo_cargado = load_model("../models/neural_network.keras")

# Opción 2: guardar con joblib si usas un wrapper sklearn (SciKeras)
# pip install scikeras
from scikeras.wrappers import KerasClassifier
```

---

## 7. Módulo: `evaluator.py`

Responsabilidad: calcular métricas de todos los modelos, generar tabla comparativa, dibujar curvas ROC y matrices de confusión.

### Imports

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
```

### Función genérica de evaluación

```python
def evaluate_model(name: str, model, X_test, y_test, is_nn: bool = False) -> dict:
    """
    Devuelve un dict con todas las métricas.
    is_nn=True si el modelo es Keras (predict devuelve probabilidades directamente).
    """
    if is_nn:
        y_proba = model.predict(X_test).flatten()
    else:
        y_proba = model.predict_proba(X_test)[:, 1]

    y_pred = (y_proba >= 0.5).astype(int)

    return {
        'Modelo':    name,
        'Accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'Recall':    round(recall_score(y_test, y_pred, zero_division=0), 4),
        'F1-Score':  round(f1_score(y_test, y_pred, zero_division=0), 4),
        'ROC-AUC':   round(roc_auc_score(y_test, y_proba), 4)
    }
```

### Tabla comparativa (obligatoria en el informe)

```python
modelos = {
    'Logistic Regression': pipe_lr,
    'Decision Tree':       pipe_dt,
    'Random Forest':       pipe_rf,
    'XGBoost':             pipe_xgb,
}

results = []
for name, model in modelos.items():
    results.append(evaluate_model(name, model, X_test, y_test))

# Red neuronal aparte (X_test ya escalado)
results.append(evaluate_model('Neural Network (Keras)', modelo_nn, X_test_scaled, y_test, is_nn=True))

df_results = pd.DataFrame(results).set_index('Modelo')
print(df_results.sort_values('ROC-AUC', ascending=False))
```

### Curvas ROC comparativas (obligatorio)

```python
plt.figure(figsize=(10, 7))

for name, model in modelos.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')

# Red neuronal
y_proba_nn = modelo_nn.predict(X_test_scaled).flatten()
fpr_nn, tpr_nn, _ = roc_curve(y_test, y_proba_nn)
auc_nn = roc_auc_score(y_test, y_proba_nn)
plt.plot(fpr_nn, tpr_nn, label=f'Neural Network (AUC={auc_nn:.3f})', linestyle='--')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('Curvas ROC — Comparativa de modelos')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('../outputs/roc_curves.png', dpi=150)
plt.show()
```

### Matrices de confusión (obligatorio)

```python
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

all_models = {**modelos, 'Neural Network': (modelo_nn, X_test_scaled, True)}

for ax, (name, model) in zip(axes, modelos.items()):
    y_pred = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=False)
    ax.set_title(name)

# NN aparte
y_pred_nn = (modelo_nn.predict(X_test_scaled).flatten() >= 0.5).astype(int)
cm_nn = confusion_matrix(y_test, y_pred_nn)
ConfusionMatrixDisplay(cm_nn).plot(ax=axes[4], colorbar=False)
axes[4].set_title('Neural Network')

fig.tight_layout()
plt.savefig('../outputs/confusion_matrices.png', dpi=150)
plt.show()
```

### Feature importance (recomendado para el informe)

```python
# RandomForest y XGBoost tienen feature_importances_ directamente en el pipeline
importances = pipe_rf.named_steps['model'].feature_importances_

# Si el preprocesador es un ColumnTransformer, conseguir los nombres de features es complejo.
# Alternativa simple: usar el preprocesador ya entrenado para obtener nombres
feature_names = preprocessor.get_feature_names_out()

df_imp = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

sns.barplot(data=df_imp.head(15), x='importance', y='feature')
plt.title('Top 15 variables más importantes — Random Forest')
plt.tight_layout()
plt.savefig('../outputs/feature_importance.png')
```

---

## 8. Módulo: `predictor.py`

Responsabilidad: cargar el mejor modelo y realizar inferencia sobre nuevos datos.

```python
import joblib
import pandas as pd
from pathlib import Path

PATH_MODELS = Path("../models/")


def load_model(filename: str):
    with open(PATH_MODELS / filename, "rb") as f:
        return joblib.load(f)


def predict(model, raw_data: dict) -> dict:
    """
    raw_data: dict con los valores de las features sin procesar.
    El Pipeline incluye el preprocesador, así que acepta datos crudos.
    """
    X = pd.DataFrame([raw_data])
    y_pred  = model.predict(X)[0]
    y_proba = model.predict_proba(X)[0, 1]
    return {
        'prediction': int(y_pred),
        'probability': round(float(y_proba), 4)
    }


# Ejemplo de uso — datos de hotel (sin escalar, el Pipeline lo gestiona)
best_model = load_model("best_model.pkl")
resultado = predict(best_model, {
    "hotel": "City Hotel",
    "lead_time": 45,
    "arrival_date_year": 2016,
    "arrival_date_month": "July",
    "arrival_date_week_number": 27,
    "arrival_date_day_of_month": 1,
    "stays_in_weekend_nights": 1,
    "stays_in_week_nights": 2,
    "adults": 2, "children": 0.0, "babies": 0,
    "meal": "BB",
    "market_segment": "Online TA",
    "distribution_channel": "TA/TO",
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0,
    "reserved_room_type": "A",
    "assigned_room_type": "A",
    "booking_changes": 0,
    "deposit_type": "No Deposit",
    "days_in_waiting_list": 0,
    "customer_type": "Transient",
    "adr": 82.0,
    "required_car_parking_spaces": 0,
    "total_of_special_requests": 1,
    # feature engineering ya aplicado en load_data:
    "total_nights": 3, "has_children": 0, "room_was_changed": 0,
    "has_agent": 1, "has_company": 0, "country_grouped": "PRT"
})
print(resultado)
# {'prediction': 0, 'probability': 0.2341}
# prediction=0 → la reserva NO se cancela
# prediction=1 → la reserva SE cancela
```

> Si el mejor modelo es la red neuronal (Keras), usar `load_model` de TF en su lugar y gestionar el escalado por separado (o usar un Pipeline con SciKeras).

---

## 9. Módulo: `config.py`

Centraliza todas las constantes para no repetir paths ni parámetros en cada módulo.

```python
from pathlib import Path

# Paths
BASE_DIR          = Path(__file__).resolve().parent.parent
PATH_DATA_RAW     = BASE_DIR / "data" / "raw"
PATH_DATA_PROC    = BASE_DIR / "data" / "processed"
PATH_MODELS       = BASE_DIR / "models"
PATH_OUTPUTS      = BASE_DIR / "outputs"

# Dataset — Hotel Bookings
DATASET_FILENAME  = "dataset_practica_final.csv"
TARGET_COLUMN     = "is_canceled"
TEST_SIZE         = 0.2
RANDOM_STATE      = 42

# Columnas a eliminar por DATA LEAKAGE (¡siempre antes de entrenar!)
LEAKAGE_COLS = ['reservation_status', 'reservation_status_date']

# Columnas numéricas (aplicar StandardScaler)
NUMERIC_COLS = [
    'lead_time', 'arrival_date_year', 'arrival_date_week_number',
    'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights',
    'adults', 'children', 'babies', 'is_repeated_guest',
    'previous_cancellations', 'previous_bookings_not_canceled',
    'booking_changes', 'days_in_waiting_list', 'adr',
    'required_car_parking_spaces', 'total_of_special_requests',
    # Columnas de feature engineering
    'total_nights', 'has_children', 'room_was_changed', 'has_agent', 'has_company'
]

# Columnas categóricas de baja/media cardinalidad (aplicar OneHotEncoder)
CAT_LOW_COLS = [
    'hotel', 'arrival_date_month', 'meal', 'market_segment',
    'distribution_channel', 'deposit_type', 'customer_type',
    'reserved_room_type', 'assigned_room_type',
    'country_grouped'   # resultado del agrupamiento de 'country'
]

# Modelos
N_ESTIMATORS_RF   = 200
MAX_DEPTH_RF      = 6
N_ESTIMATORS_XGB  = 200
LEARNING_RATE_XGB = 0.1
CV_FOLDS          = 5
SCORING_METRIC    = "roc_auc"    # métrica principal para GridSearchCV y comparativas

# Red neuronal
NN_EPOCHS         = 200
NN_BATCH_SIZE     = 32
NN_PATIENCE       = 15
```

---

## 10. Pipeline sklearn

El `Pipeline` de sklearn es el núcleo de la automatización. Encadena preprocesado + modelo en un solo objeto que:

- Aplica `fit_transform` en train y solo `transform` en test automáticamente.
- Se puede guardar con joblib completo (scaler incluido).
- Acepta `GridSearchCV` con parámetros del modelo como `'model__n_estimators'`.
- Permite hacer `pipeline.predict(X_raw)` directamente sin preprocesar a mano.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Flujo mínimo completo:
pipe = Pipeline([
    ('preprocessor', columntransformer_object),
    ('model', RandomForestClassifier(random_state=42))
])

pipe.fit(X_train, y_train)           # preprocesa + entrena
y_pred = pipe.predict(X_test)        # preprocesa + predice
y_proba = pipe.predict_proba(X_test)[:, 1]

# Guardar todo en uno
with open("../models/best_model.pkl", "wb") as f:
    joblib.dump(pipe, f)

# Cargar y usar sin importar sklearn (ya incluye los transformadores)
with open("../models/best_model.pkl", "rb") as f:
    pipe_loaded = joblib.load(f)
pipe_loaded.predict(X_raw)
```

---

## 11. Bonus técnicos

### Bonus 1 — GridSearchCV (el más fácil, recomendado siempre)

Ya detallado en sección 5. Resumen rápido:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 5, None]
}
grid = GridSearchCV(pipe_rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)
best = grid.best_estimator_   # ya es un Pipeline entrenado
```

### Bonus 2 — MLflow (tracking de experimentos)

```python
import mlflow
import mlflow.sklearn

mlflow.set_experiment("Proyecto_Final_ML")

# Opción 1: autolog (registra todo automáticamente para sklearn)
mlflow.sklearn.autolog()
pipe_rf.fit(X_train, y_train)   # se registra automáticamente

# Opción 2: manual (más control)
with mlflow.start_run(run_name="RandomForest_v1"):
    pipe_rf.fit(X_train, y_train)
    y_proba = pipe_rf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    mlflow.log_params({"n_estimators": 200, "max_depth": 6})
    mlflow.log_metric("roc_auc", auc)
    mlflow.sklearn.log_model(
        sk_model=pipe_rf,
        name="random_forest",
        registered_model_name="BestClassifier"
    )
```

### Bonus 3 — FastAPI REST API

```python
# src/api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("../models/best_model.pkl")

class BookingInput(BaseModel):
    hotel: str
    lead_time: int
    arrival_date_year: int
    arrival_date_month: str
    arrival_date_week_number: int
    arrival_date_day_of_month: int
    stays_in_weekend_nights: int
    stays_in_week_nights: int
    adults: int
    children: float = 0
    babies: int = 0
    meal: str
    market_segment: str
    distribution_channel: str
    is_repeated_guest: int = 0
    previous_cancellations: int = 0
    previous_bookings_not_canceled: int = 0
    reserved_room_type: str
    assigned_room_type: str
    booking_changes: int = 0
    deposit_type: str
    days_in_waiting_list: int = 0
    customer_type: str
    adr: float
    required_car_parking_spaces: int = 0
    total_of_special_requests: int = 0
    # campos de feature engineering (calculados en el cliente o en la API)
    country_grouped: str = 'Other'
    total_nights: int = 0
    has_children: int = 0
    room_was_changed: int = 0
    has_agent: int = 0
    has_company: int = 0

@app.post("/predict")
def predict_booking(data: BookingInput):
    X = pd.DataFrame([data.dict()])
    y_pred  = int(model.predict(X)[0])
    y_proba = float(model.predict_proba(X)[0, 1])
    return {
        "is_canceled": y_pred,
        "cancel_probability": round(y_proba, 4),
        "interpretation": "Se cancelará" if y_pred == 1 else "No se cancelará"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

# Arrancar: uvicorn src.api:app --reload
```

### Bonus 4 — Streamlit UI

```python
# src/app_streamlit.py
import streamlit as st
import joblib
import pandas as pd

model = joblib.load("../models/best_model.pkl")

st.title("Predictor de Clasificación")

# Inputs — campos del dataset de hotel
hotel         = st.selectbox("Tipo de hotel", ["City Hotel", "Resort Hotel"])
lead_time     = st.slider("Días de antelación (lead_time)", 0, 500, 30)
deposit_type  = st.selectbox("Tipo de depósito", ["No Deposit", "Refundable", "Non Refund"])
market_seg    = st.selectbox("Canal de marketing", ["Online TA", "Offline TA/TO", "Direct", "Corporate", "Groups", "Complementary", "Aviation"])
customer_type = st.selectbox("Tipo de cliente", ["Transient", "Transient-Party", "Contract", "Group"])
deposit_type  = st.selectbox("Tipo de depósito", ["No Deposit", "Refundable", "Non Refund"])
stays_weekend = st.slider("Noches de fin de semana", 0, 10, 1)
stays_week    = st.slider("Noches entre semana", 0, 14, 2)
adr           = st.number_input("ADR (precio promedio/noche €)", min_value=0.0, max_value=1000.0, value=100.0)
prev_cancel   = st.slider("Cancelaciones previas", 0, 20, 0)
special_req   = st.slider("Peticiones especiales", 0, 5, 0)
room_res      = st.selectbox("Habitación reservada", list("ABCDEFGH"))
room_asg      = st.selectbox("Habitación asignada", list("ABCDEFGHI"))

if st.button("Predecir"):
    input_data = {
        "hotel": hotel, "lead_time": lead_time, "deposit_type": deposit_type,
        "market_segment": market_seg, "customer_type": customer_type,
        "stays_in_weekend_nights": stays_weekend, "stays_in_week_nights": stays_week,
        "adr": adr, "previous_cancellations": prev_cancel,
        "total_of_special_requests": special_req,
        "reserved_room_type": room_res, "assigned_room_type": room_asg,
        # feature engineering
        "total_nights": stays_weekend + stays_week,
        "room_was_changed": int(room_res != room_asg),
        # rellenar el resto con valores por defecto
        "arrival_date_year": 2016, "arrival_date_month": "July",
        "arrival_date_week_number": 27, "arrival_date_day_of_month": 1,
        "adults": 2, "children": 0.0, "babies": 0, "meal": "BB",
        "distribution_channel": "TA/TO", "is_repeated_guest": 0,
        "previous_bookings_not_canceled": 0, "booking_changes": 0,
        "days_in_waiting_list": 0, "required_car_parking_spaces": 0,
        "has_children": 0, "has_agent": 1, "has_company": 0,
        "country_grouped": "Other"
    }
    X = pd.DataFrame([input_data])
    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0, 1]
    if pred == 1:
        st.error(f"❌ Se cancelará — Probabilidad: {proba:.2%}")
    else:
        st.success(f"✅ No se cancelará — Probabilidad de cancelación: {proba:.2%}")

# Arrancar: streamlit run src/app_streamlit.py
```

---

## 12. Checklist de entregables

### Repositorio GitHub

- [ ] `README.md` con: autores, descripción del problema, instrucciones de entorno, resultados
- [ ] `.gitignore` correcto (no subir `.pkl` pesados, datos, `__pycache__`, `.env`)
- [ ] `requirements.txt` con todas las dependencias y versiones
- [ ] Commits de ambas personas en el historial
- [ ] Roles definidos explícitamente

### Código

- [ ] `src/__init__.py`
- [ ] `src/config.py` con paths, `LEAKAGE_COLS`, `NUMERIC_COLS`, `CAT_LOW_COLS` definidos
- [ ] `src/data_loader.py` con `na_values=['NULL']`, eliminación de leakage y feature engineering
- [ ] `src/model_trainer.py` con los 5 modelos entrenados
- [ ] `src/evaluator.py` con métricas, ROC y matrices de confusión
- [ ] `src/predictor.py` con función de inferencia sobre nuevos datos
- [ ] Pipeline de sklearn integrando preprocesador + modelo
- [ ] Modelo guardado con joblib (o SavedModel para Keras)
- [ ] ⚠️ `reservation_status` y `reservation_status_date` eliminados **antes** de entrenar
- [ ] Feature engineering aplicado: `total_nights`, `has_children`, `room_was_changed`, `has_agent`, `has_company`, `country_grouped`
- [ ] `country`, `agent`, `company` originales no entran en el modelo

### Visualizaciones obligatorias

- [ ] Matriz de confusión de cada modelo
- [ ] Curvas ROC comparativas (todos los modelos en el mismo gráfico)
- [ ] Tabla comparativa de métricas (Accuracy, Precision, Recall, F1, AUC)
- [ ] Curvas de aprendizaje de la red neuronal (loss y accuracy por época)

### Informe PDF/Markdown

- [ ] Roles de la pareja
- [ ] Justificación del problema y del dataset
- [ ] EDA (con gráficos)
- [ ] Diseño del sistema y pipeline
- [ ] Tabla comparativa de modelos + justificación del mejor
- [ ] Justificación de la métrica principal elegida
- [ ] Reflexión crítica: limitaciones y mejoras

---

## Referencia rápida — dónde encontrar cada cosa en los apuntes

| Tema                                               | Dónde está                             |
| -------------------------------------------------- | -------------------------------------- |
| EDA completo, pandas, visualizaciones              | `apuntes_clase_1.md` §1                |
| `train_test_split`, `stratify`, validación cruzada | `apuntes_clase_1.md` §6-7              |
| `GridSearchCV`, `best_params_`                     | `apuntes_clase_1.md` §9                |
| `StandardScaler`, `MinMaxScaler`, data leakage     | `apuntes_clase_2.md` §escalado         |
| LogReg, DTree, RF, XGBoost, KNN, SVM               | `apuntes_clase_2.md` §1-7              |
| Métricas clasificación: F1, AUC, confusion matrix  | `apuntes_clase_2.md` §métricas         |
| `ColumnTransformer`, `Pipeline`, `OrdinalEncoder`  | `apuntes_clase_4.md` §preprocesamiento |
| Red neuronal Keras: Sequential, compile, fit       | `apuntes_clase_4.md` §1-5              |
| Dropout, EarlyStopping, curvas de aprendizaje      | `apuntes_clase_4.md` §8                |
| Serialización con joblib, `pathlib.Path`           | `apuntes_extras.md` §4.1               |
| MLflow: start_run, log_params, log_model           | `apuntes_extras.md` §3                 |
| FastAPI / Streamlit (bonus)                        | `apuntes_extras.md` §4.2               |
