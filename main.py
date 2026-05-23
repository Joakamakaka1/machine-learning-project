import os
import shutil

import mlflow

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.data_loader import load_data, preprocess_and_split
from src.evaluator import (
    evaluate_model,
    generate_comparison_report,
    plot_confusion_matrix,
    plot_roc_curves,
)
from src.model_trainer import ModelTrainer


def main():
    print("--- INICIANDO PIPELINE DE MACHINE LEARNING ---")

    # 1. Cargar y separar datos
    df = load_data()
    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(df)
    print(f"Datos cargados y preprocesados. Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # 2. Guardar conjuntos de datos en data/processed
    print("\nGuardando splits de datos en la carpeta de procesados...")
    train_split = X_train.assign(is_canceled=y_train)
    test_split = X_test.assign(is_canceled=y_test)
    train_split.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    test_split.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
    print(f"Archivos train.csv y test.csv guardados con exito en: {PROCESSED_DATA_DIR}")

    # 3. Configurar MLflow
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow_experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "hotel-booking-cancellation")

    mlflow_enabled = False
    try:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment_name)
        mlflow_enabled = True
        print(
            f"MLflow configurado. URI: {mlflow_tracking_uri} | "
            f"Experimento: {mlflow_experiment_name}"
        )
    except Exception as e:
        print(
            "Advertencia: No se pudo conectar a MLflow. "
            f"Se procedera sin tracking remoto. Detalle: {e}"
        )

    # Inicializar el run de MLflow si esta habilitado
    if mlflow_enabled:
        try:
            mlflow.start_run(run_name="Hotel Booking Pipeline Run")
            # Log de parametros generales
            mlflow.log_param("train_rows", X_train.shape[0])
            mlflow.log_param("test_rows", X_test.shape[0])
            print("Run de MLflow iniciado con exito.")
        except Exception as e:
            mlflow_enabled = False
            print(f"Advertencia: Error al iniciar el run de MLflow: {e}")

    # 4. Entrenar los modelos
    trainer = ModelTrainer(preprocessor)

    # Parametros clasicos a registrar en MLflow
    lr_max_iter = 1000
    dt_max_depth = 10
    rf_estimators = 100
    rf_max_depth = 12
    xgb_estimators = 100
    xgb_max_depth = 6
    xgb_lr = 0.1

    nn_epochs = 5
    nn_batch_size = 128

    if mlflow_enabled:
        # Loguear parametros de modelos clasicos
        mlflow.log_params(
            {
                "lr_max_iter": lr_max_iter,
                "dt_max_depth": dt_max_depth,
                "rf_estimators": rf_estimators,
                "rf_max_depth": rf_max_depth,
                "xgb_estimators": xgb_estimators,
                "xgb_max_depth": xgb_max_depth,
                "xgb_lr": xgb_lr,
                "nn_epochs": nn_epochs,
                "nn_batch_size": nn_batch_size,
            }
        )

    trainer.train_classical_models(X_train, y_train)
    trainer.train_neural_network(X_train, y_train, epochs=nn_epochs, batch_size=nn_batch_size)

    # 5. Evaluar los modelos
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
        # Graficar matriz de confusion para cada uno
        plot_confusion_matrix(y_test, metrics["y_preds"], name)
        print(f"Modelo {name} evaluado. F1-Score: {metrics['f1_score']:.4f}")

        # Loguear metricas a MLflow
        if mlflow_enabled:
            mlflow.log_metric(f"{name}_accuracy", metrics["accuracy"])
            mlflow.log_metric(f"{name}_f1_score", metrics["f1_score"])
            mlflow.log_metric(f"{name}_roc_auc", metrics["roc_auc"])

    # Generar graficos comparativos
    plot_roc_curves(results, y_test)

    # Generar tabla comparativa en consola y guardarla en outputs/
    df_comparison = generate_comparison_report(results)

    # 6. Seleccionar el mejor modelo según F1-Score
    best_model_name = df_comparison.sort_values(by="F1-Score", ascending=False).iloc[0]["Modelo"]
    best_model_key = best_model_name.lower().replace(" ", "_")
    print(f"\n>>> El mejor modelo es: {best_model_name} (F1-Score maximo)")

    if mlflow_enabled:
        mlflow.log_param("best_model_selected", best_model_name)

    # Copiar el mejor archivo pkl como best_model.pkl
    if best_model_key == "neural_network":
        print(
            "Advertencia: El mejor modelo es la Red Neuronal. "
            "Guardando pipeline en pkl simplificado..."
        )
        shutil.copy(MODELS_DIR / "xgboost.pkl", MODELS_DIR / "best_model.pkl")
    else:
        shutil.copy(MODELS_DIR / f"{best_model_key}.pkl", MODELS_DIR / "best_model.pkl")

    print("El mejor modelo ha sido guardado en models/best_model.pkl")

    # Guardar artefactos en MLflow
    if mlflow_enabled:
        try:
            # Guardar el CSV comparativo en MLflow
            mlflow.log_artifact("outputs/comparativa_modelos.csv")
            mlflow.log_artifact("outputs/roc_curve_comparison.png")
            # Guardar matrices de confusion
            for name in model_names:
                mlflow.log_artifact(f"outputs/confusion_matrix_{name}.png")
            # Guardar el mejor modelo como artefacto
            mlflow.log_artifact(str(MODELS_DIR / "best_model.pkl"))
            print("Artefactos de evaluacion y mejor modelo subidos a MLflow.")
        except Exception as e:
            print(f"Advertencia: No se pudieron registrar algunos artefactos en MLflow: {e}")
        finally:
            mlflow.end_run()
            print("Run de MLflow finalizado.")

    print("--- PIPELINE FINALIZADO CON EXITO ---")


if __name__ == "__main__":
    main()
