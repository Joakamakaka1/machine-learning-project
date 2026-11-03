"""What the routes receive: everything was prepared once, during startup."""

from typing import Annotated

from fastapi import Depends, Request

from hotel_cancellation.evaluation.stored_evaluation import StoredEvaluation
from hotel_cancellation.predictor import BookingPredictor


def get_predictor(request: Request) -> BookingPredictor:
    """Give the predictor loaded at startup."""
    predictor: BookingPredictor = request.app.state.predictor
    return predictor


def get_evaluation(request: Request) -> StoredEvaluation | None:
    """Give the evaluation read at startup, if there is one."""
    evaluation: StoredEvaluation | None = request.app.state.evaluation
    return evaluation


PredictorDep = Annotated[BookingPredictor, Depends(get_predictor)]
EvaluationDep = Annotated[StoredEvaluation | None, Depends(get_evaluation)]
