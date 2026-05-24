"""
- Archivo encargado de:
- Evaluar todos los modelos
- Generar la tabla comparativa exigida
- Guardar los gráficos necesarios
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.config import MODELS_DIR, OUTPUTS_DIR


# Función encargada de:
# - Evaluar los modelos tradicionales o la red neuronal
# - Devolver métricas
def evaluate_model(model_name, X_test, y_test):
    # 1. Evaluamos primero el modelo de red neuronal
    if model_name == "neural_network":
        # Cargar componentes
        preprocessor = joblib.load(MODELS_DIR / "nn_preprocessor.pkl")
        keras_model = tf.keras.models.load_model(MODELS_DIR / "neural_network.keras")

        X_test_processed = preprocessor.transform(X_test)
        y_probs = keras_model.predict(X_test_processed).flatten()
        y_preds = (y_probs >= 0.5).astype(int)
    # 2. Evaluamos los modelos tradicionales basados en sklearn
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


# --------- FUNCIONES QUE DIBUJAN LOS RESULTADOS DE LA EVALUACION DE LOS MODELOS ---------


# Función encargada de:
# - Dibujar la matriz de confusión
# - Guardarla como imagen
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    # Anotaciones con conteo absoluto y porcentaje por fila
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]:.1%})"

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=["No Cancela", "Cancela"],
        yticklabels=["No Cancela", "Cancela"],
        annot_kws={"fontsize": 11},
    )
    plt.title(
        f"Matriz de Confusión - {model_name.replace('_', ' ').title()}",
        fontsize=13,
        pad=15,
    )
    plt.ylabel("Realidad", fontsize=12)
    plt.xlabel("Predicción", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / f"confusion_matrix_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close()


# Función encargada de:
# - Extraer la importancia de variables del pipeline de un modelo basado en árboles
# - Dibujar y guardar el gráfico
def plot_feature_importances(model_name="xgboost", top_n=15):
    pipeline = joblib.load(MODELS_DIR / f"{model_name}.pkl")
    preprocessor_step = pipeline.named_steps["preprocessor"]
    classifier_step = pipeline.named_steps["classifier"]

    feature_names = preprocessor_step.get_feature_names_out()
    importances = classifier_step.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    plt.figure(figsize=(10, 7))
    plt.barh(range(len(indices)), importances[indices], color="#3498db", alpha=0.85)
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices], fontsize=11)
    plt.xlabel("Importancia (Ganancia)", fontsize=12)
    plt.title(
        f"Top {top_n} Variables más Importantes — {model_name.upper()}",
        fontsize=14,
        pad=15,
    )
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "feature_importances.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Feature importances guardadas en: {OUTPUTS_DIR / 'feature_importances.png'}")


# Función encargada de:
# - Dibujar un barplot agrupado (Accuracy / F1 / AUC) por modelo
# - Comparar visualmente los resultados de la tabla comparativa
def plot_model_comparison_chart(models_results):
    records = []
    for name, metrics in models_results.items():
        label = name.replace("_", " ").title()
        records.extend(
            [
                {"Modelo": label, "Métrica": "Accuracy", "Valor": metrics["accuracy"]},
                {"Modelo": label, "Métrica": "F1-Score", "Valor": metrics["f1_score"]},
                {"Modelo": label, "Métrica": "ROC-AUC", "Valor": metrics["roc_auc"]},
            ]
        )
    df_plot = pd.DataFrame(records)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_plot, x="Modelo", y="Valor", hue="Métrica", palette="muted")
    plt.title("Comparativa de Métricas por Modelo", fontsize=14, pad=15)
    plt.xlabel("Modelo", fontsize=12)
    plt.ylabel("Puntuación", fontsize=12)
    plt.ylim(0.5, 1.0)
    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "comparativa_modelos_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico comparativo guardado en: {OUTPUTS_DIR / 'comparativa_modelos_chart.png'}")


# Función encargada de:
# - Dibujar las curvas Precision-Recall de todos los modelos
# - Complementa la curva ROC. Útil con desbalanceo de clases
def plot_precision_recall_curves(models_results, y_test):
    no_skill = float(y_test.mean())

    plt.figure(figsize=(8, 6))
    for name, res in models_results.items():
        precision, recall, _ = precision_recall_curve(y_test, res["y_probs"])
        ap = average_precision_score(y_test, res["y_probs"])
        label_name = name.replace("_", " ").title()
        plt.plot(recall, precision, label=f"{label_name} (AP = {ap:.3f})", linewidth=2)

    plt.axhline(
        y=no_skill,
        color="k",
        linestyle="--",
        label=f"Sin Habilidad ({no_skill:.2f})",
        alpha=0.7,
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall (Sensibilidad)", fontsize=12)
    plt.ylabel("Precisión", fontsize=12)
    plt.title("Curvas Precision-Recall Comparativas", fontsize=14, pad=15)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "precision_recall_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Curvas PR guardadas en: {OUTPUTS_DIR / 'precision_recall_curves.png'}")


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
