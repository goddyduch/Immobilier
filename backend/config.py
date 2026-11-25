"""Configuration centralisée (Mac local vs Docker / Serveur distant)."""

from __future__ import annotations

import os


def get_allowed_origins() -> list[str]:
    raw = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000",
    )
    return [o.strip() for o in raw.split(",") if o.strip()]


MODEL_PATH = os.getenv("MODEL_PATH", "models/modele_poitiers.joblib")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")
SCORE_CONFIANCE_MIN_AFFICHAGE = int(os.getenv("SCORE_CONFIANCE_MIN", "45"))
SCORE_CONFIANCE_OPPORTUNITE_FORTE = int(os.getenv("SCORE_CONFIANCE_FORTE", "60"))
