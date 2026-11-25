"""Endpoints REST V1 : /v1/health, /v1/opportunities, /v1/estimate, /v1/check-url, /v1/simulate-investment, /v1/dvf-*, /v1/monitoring."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from src.ingestion import db as DB
from src.monitoring import drift as monitoring
from backend.dependencies import get_model_snapshot, get_model_store, verify_admin_key
from backend.schemas.all import (
    CheckUrlRequest,
    CheckUrlResponse,
    DvfAnalysisResponse,
    EstimateRequest,
    EstimateResponse,
    HealthResponse,
    NegotiationRequest,
    NegotiationResponse,
    OpportunitiesResponse,
    OpportunityDetail,
    SimulateInvestmentRequest,
    SimulateInvestmentResponse,
)
from backend.services.dvf_analysis import analyser_dvf, simuler_negociation
from backend.services.inference import estimate_single
from backend.services.investment import simuler_investissement
from backend.services.model_store import ModelSnapshot, ModelStore
from backend.services.opportunities import detail_opportunite, lister_opportunites
from backend.services.url_checker import verifier_et_analyser_url

router = APIRouter(prefix="/v1")


@router.get("/health", response_model=HealthResponse, tags=["Système"])
def health(store: ModelStore = Depends(get_model_store)):
    snap = store.get_snapshot()
    df_pred = DB.lire_table_ou_csv(DB.TABLE_PREDICTIONS, DB.CSV_PREDICTIONS)
    return HealthResponse(
        status="ok" if snap is not None else "degraded",
        model_loaded=snap is not None,
        model_version=snap.version if snap else "none",
        db_available=DB.db_disponible(),
        nb_offres_predictions=int(len(df_pred)),
        nb_dvf_reference=int(len(snap.df_dvf_raw)) if snap else 0,
    )


@router.get("/opportunities", response_model=OpportunitiesResponse, tags=["Radar"])
def get_opportunities(
    ville: str | None = Query(default=None),
    type_bien: str | None = Query(default=None),
    tranche_m2: str | None = Query(default=None),
    quartier: str | None = Query(default=None),
    verdict: str | None = Query(default=None),
    min_confiance: int = Query(default=45, ge=0, le=100),
    mode: str = Query(default="avec_travaux"),
    sort_by: str = Query(default="auto"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    return lister_opportunites(
        ville=ville, type_bien=type_bien, tranche_m2=tranche_m2, quartier=quartier,
        verdict=verdict, min_confiance=min_confiance, mode=mode, sort_by=sort_by,
        limit=limit, offset=offset,
    )


@router.get("/opportunities/{ad_id}", response_model=OpportunityDetail, tags=["Radar"])
def get_opportunity_detail(ad_id: str):
    return detail_opportunite(ad_id)


@router.post("/estimate", response_model=EstimateResponse, tags=["Estimation"])
def post_estimate(req: EstimateRequest, snap: ModelSnapshot = Depends(get_model_snapshot)):
    return estimate_single(req, snap)


@router.post("/check-url", response_model=CheckUrlResponse, tags=["Vérificateur URL"])
async def post_check_url(req: CheckUrlRequest, store: ModelStore = Depends(get_model_store)):
    return await verifier_et_analyser_url(req.url, store.get_snapshot())


@router.post("/simulate-investment", response_model=SimulateInvestmentResponse, tags=["Simulateur Locatif"])
def post_simulate_investment(req: SimulateInvestmentRequest):
    return simuler_investissement(req)


@router.get("/dvf-analysis", response_model=DvfAnalysisResponse, tags=["DVF & Négociation"])
def get_dvf_analysis(ville: str | None = None, type_bien: str | None = None):
    return analyser_dvf(ville=ville, type_bien=type_bien)


@router.post("/dvf-negotiation", response_model=NegotiationResponse, tags=["DVF & Négociation"])
def post_dvf_negotiation(req: NegotiationRequest):
    return simuler_negociation(req)


@router.get("/monitoring/overview", tags=["Monitoring"])
def get_monitoring_overview():
    snap = monitoring.calculer_snapshot_monitoring(sauvegarder_bdd=False)
    quarantaine = []
    runs = []
    if DB.db_disponible():
        try:
            q_df = DB.lire("SELECT quarantine_id, id, source, motif_rejet, detected_at FROM offres_quarantaine ORDER BY detected_at DESC LIMIT 20")
            quarantaine = q_df.to_dict(orient="records")
        except Exception:
            pass
        try:
            r_df = DB.lire("SELECT run_id, step_name, status, started_at, finished_at, stats_json FROM pipeline_runs ORDER BY started_at DESC LIMIT 15")
            runs = r_df.to_dict(orient="records")
        except Exception:
            pass
    snap["quarantaine_recente"] = quarantaine
    snap["pipeline_runs_recents"] = runs
    return snap


@router.post("/admin/reload-model", dependencies=[Depends(verify_admin_key)], tags=["Admin"])
def post_reload_model(store: ModelStore = Depends(get_model_store)):
    snap = store.load()
    return {"reloaded": snap is not None, "version": snap.version if snap else None}
