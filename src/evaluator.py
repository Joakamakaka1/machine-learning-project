"""
- Archivo encargado de:
- Evaluar todos los modelos
- Generar la tabla comparativa exigida
- Guardar los gráficos necesarios
"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve

from src.config import MODELS_DIR, OUTPUTS_DIR


# Función encargada de:
# - Evaluar los modelos tradicionales o la red neuronal
# - Devolver métricas
def evaluate_model(model_name, X_test, y_test):
    if model_name == "neural_network":
        # Cargar componentes
        preprocessor = joblib.load(MODELS_DIR / "nn_preprocessor.pkl")
        keras_model = tf.keras.models.load_model(MODELS_DIR / "neural_network.keras")

        X_test_processed = preprocessor.transform(X_test)
        y_probs = keras_model.predict(X_test_processed).flatten()
        y_preds = (y_probs >= 0.5).astype(int)
    else:
        # Cargar pipeline
        pipeline = joblib.load(MODELS_DIR / f"{model_name}.pkl")
        y_preds = pipeline.predict(X_test)

        if hasattr(pipeline, "predict_proba"):
            y_probs = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_probs = y_preds

    # Calcular métricas
    report = classification_report(y_test, y_preds, output_dict=True)
    accuracy = report["accuracy"]
    f1 = report["weighted avg"]["f1-score"]

    # Calcular ROC AUC
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "fpr": fpr,
        "tpr": tpr,
        "y_preds": y_preds,
        "y_probs": y_probs,
    }


# Función encargada de:
# - Generar la tabla comparativa de los modelos
# - Guardarla en un CSV
def generate_comparison_report(models_results):
    records = []
    for name, metrics in models_results.items():
        records.append(
            {
                "Modelo": name.replace("_", " ").title(),
                "Accuracy": round(metrics["accuracy"], 4),
                "F1-Score": round(metrics["f1_score"], 4),
                "ROC-AUC": round(metrics["roc_auc"], 4),
            }
        )
    df = pd.DataFrame(records)
    df.to_csv(OUTPUTS_DIR / "comparativa_modelos.csv", index=False)
    print("\nTabla Comparativa de Modelos:")
    print(df.to_string(index=False))
    return df


# Función encargada de:
# - Dibujar la matriz de confusión
# - Guardarla como imagen
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Cancela", "Cancela"],
        yticklabels=["No Cancela", "Cancela"],
    )
    plt.title(f"Matriz de Confusión - {model_name.replace('_', ' ').title()}")
    plt.ylabel("Realidad")
    plt.xlabel("Predicción")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / f"confusion_matrix_{model_name}.png")
    plt.close()


# Función encargada de:
# - Dibujar las curvas ROC de todos los modelos en un mismo gráfico
# - Compararlas
def plot_roc_curves(models_results, y_test):
    plt.figure(figsize=(8, 6))
    for name, res in models_results.items():
        plt.plot(res["fpr"], res["tpr"], label=f"{name} (AUC = {res['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Al Azar")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Tasa de Falsos Positivos (1 - Especificidad)")
    plt.ylabel("Tasa de Verdaderos Positivos (Sensibilidad)")
    plt.title("Curva ROC Comparativa")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "roc_curve_comparison.png")
    plt.close()
