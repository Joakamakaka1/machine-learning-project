import logging
import os
import shutil

import mlflow

from src.config import (
    MODEL_NAMES,
    MODELS_DIR,
    NN_BATCH_SIZE,
    NN_EPOCHS,
    PROCESSED_DATA_DIR,
    setup_dirs,
)
from src.data_loader import load_data, preprocess_and_split
from src.evaluator import (
    evaluate_model,
    generate_comparison_report,
    plot_confusion_matrix,
    plot_feature_importances,
    plot_model_comparison_chart,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_shap_summary,
)
from src.model_trainer import train_all_models

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    setup_dirs()
    logger.info("--- INICIANDO PIPELINE DE MACHINE LEARNING ---")

    # 1. Cargar y separar los datos
    df = load_data()
    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(df)
    logger.info(
        "Datos cargados y preprocesados. Train: %d, Test: %d",
        X_train.shape[0],
        X_test.shape[0],
    )

    # 2. Guardar los conjuntos de datos en data/processed
    logger.info("Guardando splits de datos en la carpeta de procesados...")
    train_split = X_train.assign(is_canceled=y_train)
    test_split = X_test.assign(is_canceled=y_test)
    train_split.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    test_split.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
    logger.info("train.csv y test.csv guardados en: %s", PROCESSED_DATA_DIR)

    # 3. Entrenar los modelos
    # Todos los modelos se entrenan dentro de un Pipeline sklearn
    # (preprocesado + clasificador), lo que garantiza:
    #   - Una única serialización por modelo (.pkl).
    #   - Evaluación y selección homogéneas (la red neuronal puede ganar).
    #   - Ausencia de fuga de datos entre preprocesado y test.
    trained_models = train_all_models(
        preprocessor,
        X_train,
        y_train,
        nn_epochs=NN_EPOCHS,
        nn_batch_size=NN_BATCH_SIZE,
        tune_xgboost=True,
    )

    # 4. Evaluar los modelos (usando los pipelines en memoria, sin recargar de disco)
    logger.info("Evaluando modelos...")
    results = {}

    for name in MODEL_NAMES:
        metrics = evaluate_model(trained_models[name], X_test, y_test)
        results[name] = metrics
        plot_confusion_matrix(y_test, metrics["y_preds"], name)
        logger.info("Modelo %s evaluado. F1-Score: %.4f", name, metrics["f1_score"])

    # Generar los gráficos comparativos
    plot_roc_curves(results, y_test)
    plot_feature_importances(trained_models["xgboost"], "xgboost")
    plot_shap_summary(trained_models["xgboost"], "xgboost", X_test)
    plot_model_comparison_chart(results)
    plot_precision_recall_curves(results, y_test)

    # Generar tabla comparativa en consola y guardarla en outputs/
    generate_comparison_report(results)

    # 5. Selección del mejor modelo por ROC-AUC.
    # Se usa ROC-AUC (y no accuracy o F1) porque es independiente del umbral de
    # decisión y robusto al desbalance de clases (~37 % de cancelaciones).
    best_model_key = max(MODEL_NAMES, key=lambda name: results[name]["roc_auc"])
    best_name = best_model_key.replace("_", " ").title()
    best_roc = results[best_model_key]["roc_auc"]
    logger.info(">>> Mejor modelo: %s (ROC-AUC máximo: %.4f)", best_name, best_roc)

    shutil.copy(MODELS_DIR / f"{best_model_key}.pkl", MODELS_DIR / "best_model.pkl")
    logger.info("El mejor modelo ha sido guardado en models/best_model.pkl")

    # 6. Registrar el experimento en MLflow
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow_experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "hotel-booking-cancellation")
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(mlflow_experiment)
        with mlflow.start_run(run_name="Hotel Booking Pipeline Run"):
            mlflow.log_params(
                {
                    "nn_epochs": NN_EPOCHS,
                    "nn_batch_size": NN_BATCH_SIZE,
                    "train_rows": X_train.shape[0],
                    "test_rows": X_test.shape[0],
                    "best_model": best_model_key,
                }
            )
            for name in MODEL_NAMES:
                mlflow.log_metric(f"{name}_f1_score", results[name]["f1_score"])
                mlflow.log_metric(f"{name}_roc_auc", results[name]["roc_auc"])
                mlflow.log_metric(f"{name}_accuracy", results[name]["accuracy"])
            mlflow.log_artifact("outputs/comparativa_modelos.csv")
            mlflow.log_artifact("outputs/roc_curve_comparison.png")
            mlflow.log_artifact(str(MODELS_DIR / "best_model.pkl"))
        logger.info("Experimento registrado en MLflow.")
    except Exception as e:
        logger.warning("MLflow no disponible, se omite el tracking. Detalle: %s", e)

    logger.info("--- PIPELINE FINALIZADO CON EXITO ---")


if __name__ == "__main__":
    main()
