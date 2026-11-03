"""Answers for failures: the log keeps the detail and the client never sees it."""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


async def _internal_error(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled error on %s", request.url.path, exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def install_error_handlers(app: FastAPI) -> None:
    """Register the handler of unexpected errors."""
    app.add_exception_handler(Exception, _internal_error)
