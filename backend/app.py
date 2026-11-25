"""Point d'entrée FastAPI V1 — lifespan, watcher modèle, CORS, routers."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend import config
from backend.exceptions import register_exception_handlers
from backend.routers.v1_all import router as v1_router
from backend.services.model_store import ModelStore


async def _watcher_modele(store: ModelStore, stop_event: asyncio.Event):
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            try:
                store.reload_if_changed()
            except Exception:
                pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = ModelStore()
    store.load()
    app.state.model_store = store
    stop_event = asyncio.Event()
    task = asyncio.create_task(_watcher_modele(store, stop_event))
    try:
        yield
    finally:
        stop_event.set()
        await task


def create_app() -> FastAPI:
    app = FastAPI(
        title="IA-Immobilier API V1",
        description="Estimation immobilière, détection de décotes, comparables DVF, simulateur locatif et monitoring.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.get_allowed_origins(),
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key"],
    )
    register_exception_handlers(app)
    app.include_router(v1_router)
    return app


app = create_app()
