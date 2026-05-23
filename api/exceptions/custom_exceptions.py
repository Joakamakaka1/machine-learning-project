class APIException(Exception):
    """Excepción base para toda la API REST."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class ModelNotLoadedError(APIException):
    """Excepción lanzada cuando el modelo de ML no está disponible o no se ha cargado."""

    def __init__(
        self, detail: str = "El modelo no se ha entrenado o no está disponible en el servidor."
    ):
        super().__init__(status_code=503, detail=detail)


class PredictionError(APIException):
    """Excepción lanzada cuando ocurre un fallo durante la predicción/inferencia."""

    def __init__(
        self,
        detail: str = "Error al realizar la inferencia sobre los datos de entrada.",
    ):
        super().__init__(status_code=400, detail=detail)
