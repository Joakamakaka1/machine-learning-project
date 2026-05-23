from fastapi import FastAPI

from api.config import settings
from api.endpoints.predict import router as predict_router
from api.services.predictor import prediction_service

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API REST para predecior cancelaciones hoteleras.",
    version=settings.VERSION,
)


# Endpoint de salud del sistema
@app.get("/")
def health_check():
    return {
        "status": "OK",
        "project_name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "model_loaded": prediction_service is not None,
    }


app.include_router(predict_router, prefix=settings.API_V1_STR, tags=["predictions"])
