import os
import shutil

import mlflow

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.data_loader import load_data, preprocess_and_split
from src.evaluator import (
    evaluate_model,
    generate_comparison_report,
    plot_confusion_matrix,
    plot_feature_importances,
    plot_model_comparison_chart,
    plot_precision_recall_curves,
    plot_roc_curves,
)
from src.model_trainer import ModelTrainer


def main():
    print("--- INICIANDO PIPELINE DE MACHINE LEARNING ---")

    # 1. Cargar y separar los datos
    df = load_data()
    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(df)
    print(f"Datos cargados y preprocesados. Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # 2. Guardar los conjuntos de datos en data/processed
    print("\nGuardando splits de datos en la carpeta de procesados...")
    train_split = X_train.assign(is_canceled=y_train)
    test_split = X_test.assign(is_canceled=y_test)
    train_split.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    test_split.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
    print(f"Archivos train.csv y test.csv guardados con exito en: {PROCESSED_DATA_DIR}")

    # 3. Entrenar los modelos
    nn_epochs = 5
    nn_batch_size = 128

    trainer = ModelTrainer(preprocessor)
    trainer.train_classical_models(X_train, y_train)
    trainer.train_neural_network(X_train, y_train, epochs=nn_epochs, batch_size=nn_batch_size)

    # 4. Evaluar los modelos
    print("\nEvaluando modelos...")
    results = {}
    model_names = [
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "xgboost",
        "neural_network",
    ]

    for name in model_names:
        metrics = evaluate_model(name, X_test, y_test)
        results[name] = metrics
        plot_confusion_matrix(y_test, metrics["y_preds"], name)
        print(f"Modelo {name} evaluado. F1-Score: {metrics['f1_score']:.4f}")

    # Generar los graficos comparativos
    plot_roc_curves(results, y_test)
    plot_feature_importances("xgboost")
    plot_model_comparison_chart(results)
    plot_precision_recall_curves(results, y_test)

    # Generar tabla comparativa en consola y guardarla en la carpeta de outputs/
    generate_comparison_report(results)

    # 5. Se selecciona el mejor modelo sklearn por F1-Score
    sklearn_model_names = ["logistic_regression", "decision_tree", "random_forest", "xgboost"]
    best_model_key = max(sklearn_model_names, key=lambda name: results[name]["f1_score"])
    print(
        f"\n>>> Mejor modelo sklearn: {best_model_key.replace('_', ' ').title()} (F1-Score maximo)"
    )

    shutil.copy(MODELS_DIR / f"{best_model_key}.pkl", MODELS_DIR / "best_model.pkl")
    print("El mejor modelo ha sido guardado en models/best_model.pkl")

    # 6. Registrar el experimento en MLflow
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow_experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "hotel-booking-cancellation")
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(mlflow_experiment)
        with mlflow.start_run(run_name="Hotel Booking Pipeline Run"):
            mlflow.log_params(
                {
                    "nn_epochs": nn_epochs,
                    "nn_batch_size": nn_batch_size,
                    "train_rows": X_train.shape[0],
                    "test_rows": X_test.shape[0],
                    "best_model": best_model_key,
                }
            )
            for name in model_names:
                mlflow.log_metric(f"{name}_f1_score", results[name]["f1_score"])
                mlflow.log_metric(f"{name}_roc_auc", results[name]["roc_auc"])
                mlflow.log_metric(f"{name}_accuracy", results[name]["accuracy"])
            mlflow.log_artifact("outputs/comparativa_modelos.csv")
            mlflow.log_artifact("outputs/roc_curve_comparison.png")
            mlflow.log_artifact(str(MODELS_DIR / "best_model.pkl"))
        print("Experimento registrado en MLflow.")
    except Exception as e:
        print(f"MLflow no disponible, se omite el tracking. Detalle: {e}")

    print("--- PIPELINE FINALIZADO CON EXITO ---")


if __name__ == "__main__":
    main()
