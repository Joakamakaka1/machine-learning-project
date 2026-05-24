from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.config import API_V1_STR, PROJECT_NAME, VERSION
from api.exceptions import APIException
from api.router import router

app = FastAPI(
    title=PROJECT_NAME,
    description="API REST para servir el predictor de cancelaciones hoteleras.",
    version=VERSION,
)


# Manejador global para excepciones de la API
@app.exception_handler(APIException)
def api_exception_handler(request: Request, exc: APIException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "status_code": exc.status_code,
        },
    )


app.include_router(router, prefix=API_V1_STR)
