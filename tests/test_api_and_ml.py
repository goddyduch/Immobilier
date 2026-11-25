import time
from fastapi.testclient import TestClient

from backend.app import app
from backend.schemas.all import SimulateInvestmentRequest
from backend.services.investment import simuler_investissement


def test_simulateur_investissement_cashflow():
    req = SimulateInvestmentRequest(
        prix_achat_fai=47000,
        frais_notaire_euro=3995,
        travaux_total_euro=5000,
        loyer_hc_mensuel=450,
        assurance_pno_annuel=200,
        taxe_fonciere_annuel=431,
        vacance_mois_an=0.5,
        charges_copro_proprio_annuel=291,
        frais_comptable_annuel=200,
        cfe_annuel=200,
        apport_euro=20000,
        duree_credit_ans=10,
        taux_credit_pct=1.5,
        taux_assurance_pct=0.0,
    )
    res = simuler_investissement(req)
    assert res.total_projet_euro == 55995.0
    assert res.emprunt_total_euro == 35995.0
    assert res.rendement_brut_pct > 9.0
    assert 315.0 <= res.mensualite_credit_euro <= 330.0


def test_simulateur_comptant_sans_division_zero_et_alerte_dpe():
    req = SimulateInvestmentRequest(
        prix_achat_fai=100000,
        loyer_hc_mensuel=600,
        apport_euro=120000,
        dpe_avant="G",
        dpe_apres="G",
    )
    res = simuler_investissement(req)
    assert res.emprunt_total_euro == 0.0
    assert res.mensualite_credit_euro == 0.0
    assert res.taux_couverture_pct is None
    assert res.alerte_dpe is not None and "DPE G" in res.alerte_dpe


def test_api_endpoints_et_monotonie_ml():
    with TestClient(app) as client:
        r_h = client.get("/v1/health")
        assert r_h.status_code == 200
        assert r_h.json()["model_loaded"] is True

        r_opp = client.get("/v1/opportunities?limit=5")
        assert r_opp.status_code == 200
        data_opp = r_opp.json()
        assert "kpis" in data_opp and "items" in data_opp

        # Monotonie surface : 90 m² > 45 m²
        base = {
            "type_bien": "flat",
            "surface_m2": 45,
            "pieces": 2,
            "ville": "Poitiers",
            "code_postal": "86000",
            "latitude": 46.5802,
            "longitude": 0.3404,
            "dpe": "C",
        }
        r_small = client.post("/v1/estimate", json=base)
        r_large = client.post("/v1/estimate", json={**base, "surface_m2": 90, "pieces": 4})
        assert r_small.status_code == 200 and r_large.status_code == 200
        assert r_large.json()["prix_ml"] > r_small.json()["prix_ml"]

        # Monotonie DPE : B >= F
        r_b = client.post("/v1/estimate", json={**base, "dpe": "B"})
        r_f = client.post("/v1/estimate", json={**base, "dpe": "F"})
        assert r_b.json()["prix_ml"] >= r_f.json()["prix_ml"]

        # Latence P95 unitaire
        t0 = time.perf_counter()
        r_perf = client.post("/v1/estimate", json=base)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        assert r_perf.status_code == 200
        assert dt_ms < 1000.0  # marge large en test CI, typiquement < 150 ms

        r_mon = client.get("/v1/monitoring/overview")
        assert r_mon.status_code == 200
        assert "statut_global" in r_mon.json()
