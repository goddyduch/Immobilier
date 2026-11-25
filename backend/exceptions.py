"""Exceptions métier et handlers HTTP uniformes pour FastAPI."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class ApiError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400, details: dict | None = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class ResourceNotFoundError(ApiError):
    def __init__(self, message: str = "Ressource introuvable.", details: dict | None = None):
        super().__init__("NOT_FOUND", message, status_code=404, details=details)


class ModelNotReadyError(ApiError):
    def __init__(self, message: str = "Modèle ML non chargé.", details: dict | None = None):
        super().__init__("MODEL_NOT_READY", message, status_code=503, details=details)


class ExternalServiceError(ApiError):
    def __init__(self, message: str = "Erreur lors de l'appel externe.", details: dict | None = None):
        super().__init__("EXTERNAL_ERROR", message, status_code=502, details=details)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error_handler(_req: Request, exc: ApiError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(_req: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Paramètres d'entrée invalides.",
                    "details": {"errors": exc.errors()},
                }
            },
        )
