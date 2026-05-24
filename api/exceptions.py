class APIException(Exception):
    """Excepcion base para toda la API REST."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class ModelNotLoadedError(APIException):
    """Excepcion lanzada cuando el modelo de ML no esta disponible."""

    def __init__(
        self, detail: str = "El modelo no se ha entrenado o no esta disponible en el servidor."
    ):
        super().__init__(status_code=503, detail=detail)


class PredictionError(APIException):
    """Excepcion lanzada cuando ocurre un fallo durante la prediccion."""

    def __init__(
        self,
        detail: str = "Error al realizar la inferencia sobre los datos de entrada.",
    ):
        super().__init__(status_code=400, detail=detail)
