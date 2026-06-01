from fastapi import FastAPI

from api.config import settings
from api.router import router

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API REST para servir el predictor de cancelaciones hoteleras.",
    version=settings.VERSION,
)

app.include_router(router, prefix=settings.API_V1_STR)
