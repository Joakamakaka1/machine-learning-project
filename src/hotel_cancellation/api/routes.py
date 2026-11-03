"""Endpoints of the API. There is no training endpoint: it is heavy and unsafe in a service."""

from dataclasses import asdict

from fastapi import APIRouter, HTTPException, status

from hotel_cancellation.api.deps import EvaluationDep, PredictorDep
from hotel_cancellation.api.schemas import (
    BatchRequest,
    BatchResponse,
    Booking,
    CostAssumptions,
    EvaluationResponse,
    HealthResponse,
    ModelInfoResponse,
    ModelSummary,
    PredictionResponse,
    ScenarioResponse,
    TestSummary,
    ValidationSummary,
)

API_PREFIX = "/api/v1"

router = APIRouter(prefix=API_PREFIX)


@router.get("/health")
def health(predictor: PredictorDep) -> HealthResponse:
    """Say that the service is up with its model loaded."""
    return HealthResponse(status="ok", model_loaded=True, model_version=predictor.model_version)


@router.get("/model-info")
def model_info(predictor: PredictorDep, evaluation: EvaluationDep) -> ModelInfoResponse:
    """Identify the served model, with its assumed costs and its saved metrics."""
    manifest = predictor.manifest
    decision = manifest["decision"]
    return ModelInfoResponse(
        model=ModelSummary(
            name=manifest["model"]["name"],
            calibration=manifest["model"]["calibration"],
            version=predictor.model_version,
            sha256=predictor.sha256,
            created_at=manifest["created_at"],
            params=manifest["model"]["params"],
        ),
        threshold=predictor.threshold,
        costs=CostAssumptions(
            are_assumptions=decision["costs_are_assumptions"],
            false_positive=decision["false_positive_cost"],
            false_negative=decision["false_negative_cost"],
            scenarios=decision["cost_scenarios"],
        ),
        input_features=manifest["input_features"],
        validation=ValidationSummary(
            mean=manifest["validation"]["mean"], std=manifest["validation"]["std"]
        ),
        test=None if evaluation is None else TestSummary(**evaluation.champion_summary()),
    )


@router.post("/predict")
def predict(booking: Booking, predictor: PredictorDep) -> PredictionResponse:
    """Predict one booking."""
    prediction = predictor.predict([booking.model_dump()])[0]
    return PredictionResponse(**asdict(prediction))


@router.post("/predict/batch")
def predict_batch(request: BatchRequest, predictor: PredictorDep) -> BatchResponse:
    """Predict several bookings, up to the limit of the request."""
    predictions = predictor.predict([booking.model_dump() for booking in request.bookings])
    return BatchResponse(
        predictions=[PredictionResponse(**asdict(prediction)) for prediction in predictions]
    )


@router.get("/evaluate")
def evaluate(predictor: PredictorDep, evaluation: EvaluationDep) -> EvaluationResponse:
    """Return the evaluation saved by the single execution. It never computes on the test."""
    if evaluation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No saved evaluation is available.")
    return EvaluationResponse(
        record=evaluation.record,
        threshold=predictor.threshold,
        costs_are_assumptions=True,
        metrics=evaluation.metrics,
        segments=evaluation.segments,
        bootstrap=evaluation.bootstrap,
        monthly=evaluation.monthly,
        psi=evaluation.psi,
        special_requests_scenario=ScenarioResponse(
            informative_only=True, rows=evaluation.special_requests_scenario
        ),
    )
