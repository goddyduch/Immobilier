"""Tests multi-villes (12 villes-centres), circuit-breaker 3h, double évaluation et API V1."""

from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from backend.app import create_app
from src.ingestion import db as DB
from src.ingestion import territoires as T
from src.ml import evaluation as ev


def test_territoires_12_villes_centres_sans_la_rochelle():
    assert len(T.VILLES_CENTRES) == 12
    assert "la_rochelle" not in T.VILLES_CENTRES
    assert T.est_ville_centre("86000", "Poitiers") is True
    assert T.est_ville_centre("37000", "Tours") is True
    assert T.est_ville_centre("49000", "Angers") is True
    # Communes périphériques exclues en V1
    assert T.est_ville_centre("86180", "Buxerolles") is False
    assert T.est_ville_centre("86280", "Saint-Benoît") is False
    assert T.est_ville_centre("17000", "La Rochelle") is False


def test_synchronisation_non_expiration_croisee(monkeypatch, tmp_path):
    """Un scan de Tours ne doit pas expirer les annonces actives de Poitiers."""
    csv_hist = tmp_path / "hist.csv"
    csv_rec = tmp_path / "rec.csv"
    df_init = pd.DataFrame([
        {"id": "bienici-poitiers-1", "source": "bienici", "prix": 180000, "surface_m2": 60, "code_postal": "86000", "ville": "Poitiers", "code_territoire": "poitiers", "est_active": True},
        {"id": "bienici-tours-1", "source": "bienici", "prix": 220000, "surface_m2": 65, "code_postal": "37000", "ville": "Tours", "code_territoire": "tours", "est_active": True},
    ])
    df_init.to_csv(csv_hist, index=False)

    monkeypatch.setattr(DB, "db_disponible", lambda: False)
    monkeypatch.setattr(DB, "CSV_HIST_BRUT", str(csv_hist))
    monkeypatch.setattr(DB, "CSV_RECENT_BRUT", str(csv_rec))
    monkeypatch.setattr(DB, "ecrire_table_sql", lambda df, t: True)

    df_scan_tours = pd.DataFrame([
        {"id": "bienici-tours-2", "source": "bienici", "prix": 210000, "surface_m2": 62, "code_postal": "37000", "ville": "Tours"},
    ])
    df_h, _, stats = DB.synchroniser_offres_brutes(
        df_scan_tours,
        sources_scannees=["bienici"],
        marquer_absentes_expirees=True,
        code_territoire="tours",
        scrape_complet=True,
    )
    row_poitiers = df_h[df_h["id"] == "bienici-poitiers-1"].iloc[0]
    row_tours_old = df_h[df_h["id"] == "bienici-tours-1"].iloc[0]
    assert bool(row_poitiers["est_active"]) is True
    assert bool(row_tours_old["est_active"]) is False
    assert stats["expirees"] == 1


def test_metriques_evaluation_completes():
    import numpy as np
    y_true = np.array([150000, 200000, 250000, 180000, 300000, 160000, 220000, 275000, 195000, 240000, 170000, 260000, 210000, 290000, 185000], dtype=float)
    y_pred = y_true * 1.05
    surf = np.full(len(y_true), 70.0)
    m = ev.calculer_metriques_cv(y_true, y_pred, surf, q10=y_true * 0.85, q90=y_true * 1.15)
    for k in ("r2", "chi2_stat", "chi2_pvalue", "rmse", "mae", "med_ae", "mape", "mdape", "within_20pct", "couverture_q10_q90_pct"):
        assert k in m
    assert m["within_20pct"] == 100.0
    assert m["couverture_q10_q90_pct"] == 100.0


def test_api_evaluation_et_shap_pct():
    app = create_app()
    with TestClient(app) as client:
        r_ev = client.get("/v1/evaluation")
        assert r_ev.status_code == 200
        ev_data = r_ev.json()
        assert "global_metrics" in ev_data
        assert "par_ville" in ev_data
        assert "par_categorie_taille" in ev_data
        assert any(c["categorie"] == "< 20 m²" for c in ev_data["par_categorie_taille"])
        assert "comparatif_modeles" in ev_data

        r_est = client.post("/v1/estimate", json={
            "type_bien": "flat",
            "surface_m2": 55,
            "pieces": 2,
            "ville": "Buxerolles",
            "code_postal": "86180",
            "dpe": "C",
        })
        assert r_est.status_code == 200
        est = r_est.json()
        assert est["ville_couverte"] is False
        assert est["avertissement_couverture"] is not None
        if est["shap_facteurs"]:
            assert "impact_pct" in est["shap_facteurs"][0]
            assert "description" in est["shap_facteurs"][0]
