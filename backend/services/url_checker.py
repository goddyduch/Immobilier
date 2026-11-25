"""Vérificateur d'annonce par URL (lookup DB sinon analyse live asynchrone)."""

from __future__ import annotations

import json
import re
from urllib.parse import urlparse, urlunparse
import httpx

from src.ingestion import db as DB
from backend.schemas.all import CheckUrlResponse, EstimateRequest, OpportunityDetail
from backend.services.inference import estimate_single
from backend.services.model_store import ModelSnapshot
from backend.services.opportunities import detail_opportunite


def normaliser_url(url: str) -> str:
    p = urlparse(url.strip())
    return urlunparse((p.scheme.lower(), p.netloc.lower(), p.path.rstrip("/"), "", "", ""))


async def verifier_et_analyser_url(url: str, snap: ModelSnapshot | None) -> CheckUrlResponse:
    norm = normaliser_url(url)

    # 1. Lookup rapide dans offres_predictions
    df_pred = DB.lire_table_ou_csv(DB.TABLE_PREDICTIONS, DB.CSV_PREDICTIONS)
    if not df_pred.empty and "url" in df_pred.columns:
        urls_norm = df_pred["url"].fillna("").astype(str).apply(normaliser_url)
        match = df_pred[urls_norm == norm]
        if not match.empty:
            ad_id = str(match.iloc[0]["id"])
            return CheckUrlResponse(
                status="found_in_db",
                message="Annonce trouvée dans notre base d'analyses.",
                en_ligne=True,
                detail=detail_opportunite(ad_id),
            )

    # 2. Tentative d'extraction légère (BienIci API JSON ou JSON-LD public)
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    try:
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
            if "bienici.com" in norm:
                m_id = re.search(r"/annonce/[^/]+/([a-zA-Z0-9_-]+)$", norm) or re.search(r"([a-zA-Z0-9_-]+)$", norm)
                if m_id:
                    r_api = await client.get(
                        "https://www.bienici.com/realEstateAd.json",
                        params={"id": m_id.group(1)},
                        headers=headers,
                    )
                    if r_api.status_code == 200:
                        ad = r_api.json()
                        if ad and ad.get("surfaceArea") and ad.get("price"):
                            req = EstimateRequest(
                                type_bien="house" if ad.get("propertyType") == "house" else "flat",
                                surface_m2=float(ad["surfaceArea"]),
                                pieces=int(ad.get("roomsQuantity") or 3),
                                ville=str(ad.get("city") or "Poitiers"),
                                code_postal=str(ad.get("postalCode") or "86000"),
                                quartier=str(ad.get("district", {}).get("name") if isinstance(ad.get("district"), dict) else ad.get("city") or "Poitiers"),
                                latitude=(ad.get("blurInfo", {}) or {}).get("position", {}).get("lat"),
                                longitude=(ad.get("blurInfo", {}) or {}).get("position", {}).get("lon"),
                                dpe=ad.get("energyClassification") if ad.get("energyClassification") in list("ABCDEFG") else "D",
                                annee_construction=ad.get("yearOfConstruction"),
                                prix_affiche=float(ad["price"]),
                                description=str(ad.get("description") or ""),
                            )
                            if snap is not None:
                                est = estimate_single(req, snap)
                                photos = [p.get("url") for p in (ad.get("photos") or []) if isinstance(p, dict) and p.get("url")][:6]
                                detail = OpportunityDetail(
                                    id=str(ad.get("id", "live")),
                                    titre=str(ad.get("title") or "Annonce analysée en direct"),
                                    ville=req.ville,
                                    quartier=req.quartier,
                                    type_bien=req.type_bien,
                                    surface_m2=req.surface_m2,
                                    pieces=req.pieces,
                                    dpe=req.dpe,
                                    prix_affiche=float(req.prix_affiche or 0),
                                    prix_estime_ml=est.prix_ml,
                                    prix_hybride=est.prix_hybride,
                                    prix_q10=est.prix_q10,
                                    prix_q90=est.prix_q90,
                                    intervalle_largeur_pct=est.intervalle_largeur_pct,
                                    valeur_estimee_renovee=est.valeur_estimee_renovee,
                                    cout_travaux_estime=est.cout_travaux_estime,
                                    niveau_travaux_cat=est.niveau_travaux_cat,
                                    ecart_brut_pct=est.ecart_brut_pct or 0.0,
                                    decote_nette_pct=est.decote_nette_pct or 0.0,
                                    score_confiance=est.score_confiance,
                                    verdict=est.verdict,
                                    loyer_mensuel_estime=est.loyer_mensuel_estime,
                                    loyer_bas=est.loyer_bas,
                                    loyer_haut=est.loyer_haut,
                                    rendement_brut_pct=est.rendement_brut_pct,
                                    latitude=req.latitude,
                                    longitude=req.longitude,
                                    url=url,
                                    photos_urls=photos,
                                    description=req.description,
                                    taux_nego_estime_pct=est.taux_nego_estime_pct,
                                    prix_cible_nego=est.prix_cible_nego,
                                    shap_facteurs=est.shap_facteurs,
                                    comparables=est.comparables,
                                )
                                return CheckUrlResponse(
                                    status="analyzed_live",
                                    message="Annonce extraite et analysée en direct.",
                                    en_ligne=True,
                                    detail=detail,
                                )

            # Fallback lecture HTML JSON-LD générique
            r_html = await client.get(url, headers=headers)
            if r_html.status_code in (403, 429):
                return CheckUrlResponse(
                    status="blocked_or_unavailable",
                    message="Le site protège l'accès automatique. Utilisez le formulaire rapide ci-dessous.",
                    en_ligne=None,
                )
            if r_html.status_code in (404, 410):
                return CheckUrlResponse(
                    status="blocked_or_unavailable",
                    message="Cette annonce semble expirée ou supprimée (404).",
                    en_ligne=False,
                )
            # Extraction JSON-LD basique si présent
            m_ld = re.search(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', r_html.text, re.S)
            if m_ld:
                try:
                    ld = json.loads(m_ld.group(1))
                    if isinstance(ld, list) and ld:
                        ld = ld[0]
                    offers = ld.get("offers", {}) if isinstance(ld, dict) else {}
                    price = offers.get("price") if isinstance(offers, dict) else None
                    if price:
                        return CheckUrlResponse(
                            status="blocked_or_unavailable",
                            message="Annonce détectée mais surface/DPE incomplets — complétez le formulaire rapide.",
                            en_ligne=True,
                        )
                except Exception:
                    pass
    except Exception:
        pass

    return CheckUrlResponse(
        status="blocked_or_unavailable",
        message="Impossible d'extraire automatiquement cette URL. Utilisez le formulaire d'estimation rapide.",
        en_ligne=None,
    )
