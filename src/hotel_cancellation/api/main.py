"""Application of the API. Importing this module loads nothing: the lifespan does."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib import metadata
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from hotel_cancellation import config
from hotel_cancellation.api.errors import install_error_handlers
from hotel_cancellation.api.routes import router
from hotel_cancellation.predictor import BookingPredictor
from hotel_cancellation.stored_evaluation import StoredEvaluation, load_stored_evaluation

logger = logging.getLogger(__name__)

# The interface runs on this machine: any port of localhost, and nothing else.
_LOCAL_ORIGINS = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"


def _evaluation_of(predictor: BookingPredictor, folder: Path) -> StoredEvaluation | None:
    """Read the saved evaluation, unless it measured another model: it says nothing of this one."""
    evaluation = load_stored_evaluation(folder)
    if evaluation is not None and evaluation.execution["champion_sha256"] != predictor.sha256:
        logger.warning("The saved evaluation measured another model, so it is not published.")
        return None
    return evaluation


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once. A model that fails its checks stops the service from starting."""
    settings = config.get_settings()
    predictor = BookingPredictor.from_folder(settings.champion_dir)
    app.state.predictor = predictor
    app.state.evaluation = _evaluation_of(predictor, settings.final_evaluation_dir)
    logger.info("Serving %s", predictor.model_version)
    yield


def create_app() -> FastAPI:
    """Build the application."""
    app = FastAPI(
        title="Cancelación de reservas de hotel",
        description=(
            "Predice si se cancelará una reserva al confirmarla. Los costes que fijan el umbral "
            "son supuestos, no datos de los hoteles."
        ),
        version=metadata.version("hotel-cancellation"),
        lifespan=_lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=_LOCAL_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    install_error_handlers(app)
    app.include_router(router)
    return app


app = create_app()
