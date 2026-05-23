from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.config import settings
from api.endpoints.v1 import router as v1_router
from api.exceptions import APIException

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API REST modular para servir el predictor de cancelaciones hoteleras.",
    version=settings.VERSION,
)


# Registrar manejador global para excepciones de la API
@app.exception_handler(APIException)
def api_exception_handler(request: Request, exc: APIException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "status_code": exc.status_code,
        },
    )


# Incluimos los endpoints de la versión 1 (incluye health y predict)
app.include_router(v1_router, prefix=settings.API_V1_STR)
