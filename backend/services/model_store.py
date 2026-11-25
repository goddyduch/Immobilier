"""ModelStore avec ModelSnapshot immuable et rechargement atomique à chaud."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from src.ml import avm as AVMComparable
from src.ingestion import db as DB
from src.ml import features as fe
from backend import config


@dataclass(frozen=True)
class ModelSnapshot:
    version: str
    modele: object
    modele_q10: object | None
    modele_q90: object | None
    colonnes_X: list[str]
    shap_explainer: object | None
    df_dvf_raw: pd.DataFrame
    df_dvf_comp: pd.DataFrame
    dvf_trees_by_type: dict[str, tuple[BallTree, pd.DataFrame, float]]
    df_marche_comp: pd.DataFrame
    iris_features: list[dict]
    loaded_at: str
    mtime_ns: int


def _construire_iris_cache() -> list[dict]:
    if not os.path.exists(fe.CHEMIN_IRIS_GEOJSON):
        return []
    try:
        data = json.load(open(fe.CHEMIN_IRIS_GEOJSON, "r", encoding="utf-8"))
        return data.get("features", [])
    except Exception:
        return []


def _construire_dvf_trees(df_dvf: pd.DataFrame) -> dict[str, tuple[BallTree, pd.DataFrame, float]]:
    out: dict[str, tuple[BallTree, pd.DataFrame, float]] = {}
    if df_dvf is None or df_dvf.empty:
        return out
    d = df_dvf.copy()
    d["latitude"] = pd.to_numeric(d.get("latitude"), errors="coerce")
    d["longitude"] = pd.to_numeric(d.get("longitude"), errors="coerce")
    d["surface_reelle_bati"] = pd.to_numeric(d.get("surface_reelle_bati"), errors="coerce")
    d["valeur_fonciere"] = pd.to_numeric(d.get("valeur_fonciere"), errors="coerce")
    d = d[d["latitude"].notna() & d["longitude"].notna() & (d["surface_reelle_bati"] >= 9) & (d["valeur_fonciere"] >= 10000)].copy()
    d["pm2"] = d["valeur_fonciere"] / d["surface_reelle_bati"]
    d = d[(d["pm2"] >= 500) & (d["pm2"] <= 6000)].copy()
    d["type_bien"] = np.where(d.get("type_local", "").astype(str).str.lower().str.contains("maison"), "house", "flat")
    for tb in ("flat", "house"):
        sub = d[d["type_bien"] == tb].reset_index(drop=True)
        if not sub.empty:
            tree = BallTree(np.radians(sub[["latitude", "longitude"]].to_numpy()), metric="haversine")
            out[tb] = (tree, sub, float(sub["pm2"].median()))
    return out


class ModelStore:
    """Gère le snapshot immuable en RAM et le reload atomique quand le .joblib change."""

    def __init__(self, model_path: str = config.MODEL_PATH):
        self.model_path = model_path
        self._lock = threading.RLock()
        self._snapshot: ModelSnapshot | None = None

    def load(self) -> ModelSnapshot | None:
        if not os.path.exists(self.model_path):
            return None
        mtime_ns = os.stat(self.model_path).st_mtime_ns
        artefact = joblib.load(self.model_path)
        modele = artefact["modele"]

        shap_explainer = None
        try:
            import shap
            pipe = modele.regressor_
            gbm = pipe.named_steps["gbm"]
            hgb_est = gbm.named_estimators_["hgb"] if hasattr(gbm, "named_estimators_") and "hgb" in gbm.named_estimators_ else gbm
            shap_explainer = shap.TreeExplainer(hgb_est)
        except Exception:
            shap_explainer = None

        df_dvf_raw = fe.telecharger_ou_charger_dvf(fe.CHEMIN_DVF_CSV, verbose=False)
        df_dvf_comp = AVMComparable._preparer_dvf_comp(df_dvf_raw)
        df_hist = DB.lire_table_ou_csv(DB.TABLE_HIST_FEAT, DB.CSV_HIST_FEAT)
        df_marche_comp = AVMComparable._preparer_marche_comp(df_hist)

        version = "v1"
        for mp in ("models/manifest.json", "manifest.json"):
            if os.path.exists(mp):
                try:
                    version = json.load(open(mp, "r", encoding="utf-8")).get("version", "v1")
                    break
                except Exception:
                    pass

        snap = ModelSnapshot(
            version=version,
            modele=modele,
            modele_q10=artefact.get("modele_q10"),
            modele_q90=artefact.get("modele_q90"),
            colonnes_X=list(artefact.get("colonnes_X", [])),
            shap_explainer=shap_explainer,
            df_dvf_raw=df_dvf_raw,
            df_dvf_comp=df_dvf_comp,
            dvf_trees_by_type=_construire_dvf_trees(df_dvf_raw),
            df_marche_comp=df_marche_comp,
            iris_features=_construire_iris_cache(),
            loaded_at=datetime.now(timezone.utc).isoformat(),
            mtime_ns=mtime_ns,
        )
        with self._lock:
            self._snapshot = snap
        return snap

    def get_snapshot(self) -> ModelSnapshot | None:
        return self._snapshot

    def reload_if_changed(self) -> bool:
        if not os.path.exists(self.model_path):
            return False
        cur = self._snapshot
        mtime_ns = os.stat(self.model_path).st_mtime_ns
        if cur is not None and cur.mtime_ns == mtime_ns:
            return False
        self.load()
        return True
