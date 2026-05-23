from fastapi import APIRouter

from api.endpoints.v1.health import router as health_router
from api.endpoints.v1.predict import router as predict_router

router = APIRouter()

# Incluir los enrutadores individuales de la versión 1
router.include_router(health_router)
router.include_router(predict_router)
