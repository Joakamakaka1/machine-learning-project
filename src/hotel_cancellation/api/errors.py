"""Answers for failures: the log keeps the detail and the client never sees it."""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

_VALIDATION_FIELDS = ("type", "loc", "msg")


async def _invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
    # The input stays out: a `NaN` in it could not be written back as JSON, and it can be large.
    errors = [{field: error[field] for field in _VALIDATION_FIELDS} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


async def _internal_error(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled error on %s", request.url.path, exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def install_error_handlers(app: FastAPI) -> None:
    """Register the handlers of invalid requests and of unexpected errors."""
    app.exception_handler(RequestValidationError)(_invalid_request)
    app.add_exception_handler(Exception, _internal_error)
