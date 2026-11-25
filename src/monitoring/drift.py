"""
monitoring.py — Calcul du Data Drift (PSI), Prediction Drift, Complétude & Qualité SLA.
"""

from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
from sqlalchemy import text

from src.ingestion import db as DB

VARS_DRIFT = [
    "surface_m2",
    "dpe_num",
    "valeur_dvf_nette_travaux",
    "pm2_dvf_pondere_dist",
    "dist_centre_km",
    "dist_arret_bus_m",
    "dist_commerce_alim_m",
    "surface_par_piece",
    "loyer_m2_estime",
    "cout_travaux_estime_euro",
]


def calculer_psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float | None:
    """Calcule le Population Stability Index (PSI) entre la baseline d'entraînement et les données récentes."""
    r = pd.to_numeric(ref, errors="coerce").dropna().to_numpy()
    c = pd.to_numeric(cur, errors="coerce").dropna().to_numpy()
    if len(r) < 20 or len(c) < 10:
        return None
    quantiles = np.unique(np.quantile(r, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        return 0.0
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf
    r_cnt = np.histogram(r, bins=quantiles)[0]
    c_cnt = np.histogram(c, bins=quantiles)[0]
    r_pct = np.maximum(r_cnt / max(len(r), 1), 1e-4)
    c_pct = np.maximum(c_cnt / max(len(c), 1), 1e-4)
    psi = float(np.sum((c_pct - r_pct) * np.log(c_pct / r_pct)))
    return round(psi, 4)


def calculer_snapshot_monitoring(sauvegarder_bdd: bool = True) -> dict:
    """Calcule l'état complet de monitoring (complétude, PSI Data Drift, Prediction Drift, alertes)."""
    df_hist = DB.lire_table_ou_csv(DB.TABLE_HIST_FEAT, DB.CSV_HIST_FEAT)
    df_pred = DB.lire_table_ou_csv(DB.TABLE_PREDICTIONS, DB.CSV_PREDICTIONS)

    completude = {}
    for col in ["latitude", "dpe", "ges", "annee_construction", "etage", "clip_etat_score", "avm_comp_blended_euro"]:
        if col in df_pred.columns and len(df_pred) > 0:
            completude[col] = round(float(df_pred[col].notna().mean() * 100.0), 1)
    if "source" in df_pred.columns and not df_pred.empty:
        par_src = {}
        for src, grp in df_pred.groupby("source"):
            par_src[str(src)] = {
                "n": int(len(grp)),
                "gps_pct": round(float(grp["latitude"].notna().mean() * 100.0), 1) if "latitude" in grp else 0.0,
                "dpe_pct": round(float(grp["dpe"].notna().mean() * 100.0), 1) if "dpe" in grp else 0.0,
                "clip_pct": round(float(grp["clip_etat_score"].notna().mean() * 100.0), 1) if "clip_etat_score" in grp else 0.0,
            }
        completude["par_source"] = par_src

    psi_map = {}
    for col in VARS_DRIFT:
        if col in df_hist.columns and col in df_pred.columns:
            val = calculer_psi(df_hist[col], df_pred[col])
            if val is not None:
                psi_map[col] = val

    pred_drift = {}
    if not df_pred.empty:
        ec = pd.to_numeric(df_pred.get("ecart_hybride_pct"), errors="coerce")
        conf = pd.to_numeric(df_pred.get("score_confiance"), errors="coerce")
        larg = pd.to_numeric(df_pred.get("intervalle_largeur_pct"), errors="coerce")
        avm_nb = pd.to_numeric(df_pred.get("avm_comp_nb"), errors="coerce").fillna(0)
        pred_drift = {
            "ecart_hybride_median_pct": round(float(ec.median()), 2) if ec.notna().any() else None,
            "largeur_intervalle_median_pct": round(float(larg.median()), 1) if larg.notna().any() else None,
            "score_confiance_median": round(float(conf.median()), 1) if conf.notna().any() else None,
            "taux_opportunites_fortes_pct": round(float(((ec <= -12) & (conf >= 60)).mean() * 100.0), 2),
            "taux_trop_incertain_pct": round(float((conf < 45).mean() * 100.0), 1),
            "taux_sans_avm_pct": round(float((avm_nb == 0).mean() * 100.0), 1),
        }

    alertes = []
    statut = "ok"
    gps_cov = completude.get("latitude", 100.0)
    dpe_cov = completude.get("dpe", 100.0)
    if gps_cov < 80.0:
        alertes.append({"niveau": "critical", "message": f"Couverture GPS faible ({gps_cov}% < 80%)"})
        statut = "critical"
    elif gps_cov < 90.0:
        alertes.append({"niveau": "warning", "message": f"Couverture GPS en baisse ({gps_cov}%)"})
        statut = "warning"

    if dpe_cov < 75.0:
        alertes.append({"niveau": "warning", "message": f"Couverture DPE faible ({dpe_cov}% < 75%)"})
        if statut == "ok":
            statut = "warning"

    psi_crit = [k for k, v in psi_map.items() if v >= 0.25]
    psi_warn = [k for k, v in psi_map.items() if 0.10 <= v < 0.25]
    if len(psi_crit) >= 2:
        alertes.append({"niveau": "critical", "message": f"Data Drift critique (PSI >= 0.25) sur {', '.join(psi_crit)}"})
        statut = "critical"
    elif psi_crit or len(psi_warn) >= 2:
        alertes.append({"niveau": "warning", "message": f"Data Drift modéré détecté ({', '.join(psi_crit + psi_warn)})"})
        if statut == "ok":
            statut = "warning"

    opp_rate = pred_drift.get("taux_opportunites_fortes_pct", 0.0)
    if opp_rate > 12.0:
        alertes.append({"niveau": "warning", "message": f"Taux d'opportunités fortes anormalement élevé ({opp_rate}%)"})
        if statut == "ok":
            statut = "warning"

    manifest = {}
    for mp in ("models/manifest.json", "manifest.json"):
        if os.path.exists(mp):
            try:
                manifest = json.load(open(mp, "r", encoding="utf-8"))
                break
            except Exception:
                pass

    nb_quar = 0
    if DB.db_disponible():
        try:
            with DB.engine.connect() as conn:
                nb_quar = int(conn.execute(text("SELECT COUNT(*) FROM offres_quarantaine")).scalar() or 0)
        except Exception:
            nb_quar = 0

    snapshot = {
        "model_version": manifest.get("version", "v1"),
        "model_metrics": manifest.get("metrics", {}),
        "nb_historique": int(len(df_hist)),
        "nb_actives": int(len(df_pred)),
        "nb_quarantaine": nb_quar,
        "completude": completude,
        "data_drift_psi": psi_map,
        "prediction_drift": pred_drift,
        "statut_global": statut,
        "alertes": alertes,
    }

    if sauvegarder_bdd and DB.db_disponible():
        try:
            with DB.engine.begin() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS model_monitoring_snapshots (
                        snapshot_id BIGSERIAL PRIMARY KEY,
                        model_version VARCHAR(64),
                        nb_historique INTEGER,
                        nb_actives INTEGER,
                        nb_quarantaine INTEGER,
                        completude_json JSONB,
                        data_drift_psi_json JSONB,
                        prediction_drift_json JSONB,
                        perf_dvf_json JSONB,
                        latences_json JSONB,
                        statut_global VARCHAR(24) NOT NULL DEFAULT 'ok',
                        alertes_json JSONB,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                """))
                conn.execute(
                    text("""
                        INSERT INTO model_monitoring_snapshots
                        (model_version, nb_historique, nb_actives, nb_quarantaine,
                         completude_json, data_drift_psi_json, prediction_drift_json,
                         statut_global, alertes_json)
                        VALUES (:mv, :nh, :na, :nq, CAST(:comp AS JSONB), CAST(:psi AS JSONB),
                                CAST(:pd AS JSONB), :st, CAST(:al AS JSONB))
                    """),
                    {
                        "mv": snapshot["model_version"],
                        "nh": snapshot["nb_historique"],
                        "na": snapshot["nb_actives"],
                        "nq": snapshot["nb_quarantaine"],
                        "comp": json.dumps(completude, ensure_ascii=False),
                        "psi": json.dumps(psi_map, ensure_ascii=False),
                        "pd": json.dumps(pred_drift, ensure_ascii=False),
                        "st": statut,
                        "al": json.dumps(alertes, ensure_ascii=False),
                    },
                )
        except Exception as e:
            print(f"[Monitoring] Sauvegarde snapshot ignorée : {e}")

    return snapshot
