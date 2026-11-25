"""Injection de dépendances FastAPI."""

from __future__ import annotations

from fastapi import Header, Request

from backend import config
from backend.exceptions import ApiError, ModelNotReadyError
from backend.services.model_store import ModelSnapshot, ModelStore


def get_model_store(request: Request) -> ModelStore:
    return request.app.state.model_store


def get_model_snapshot(request: Request) -> ModelSnapshot:
    snap = request.app.state.model_store.get_snapshot()
    if snap is None:
        raise ModelNotReadyError("Le modèle ML n'est pas encore chargé.")
    return snap


def verify_admin_key(x_api_key: str | None = Header(default=None)) -> None:
    if config.ADMIN_API_KEY and x_api_key != config.ADMIN_API_KEY:
        raise ApiError("UNAUTHORIZED", "Clé API admin invalide.", status_code=401)
