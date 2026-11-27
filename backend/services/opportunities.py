"""Service de lecture/filtrage des opportunités pré-calculées (offres_predictions)."""

from __future__ import annotations

import ast
import json
import numpy as np
import pandas as pd

from src.ingestion import db as DB
from backend.exceptions import ResourceNotFoundError
from backend.schemas.all import (
    ComparableItem,
    OpportunitiesFacets,
    OpportunitiesKpis,
    OpportunitiesResponse,
    OpportunityDetail,
    OpportunityItem,
    ShapFactor,
)
from backend.services.inference import determiner_verdict

TRANCHES_M2 = [
    ("10-20 m²", 10, 20),
    ("20-35 m²", 20, 35),
    ("35-50 m²", 35, 50),
    ("50-70 m²", 50, 70),
    ("70-95 m²", 70, 95),
    ("95-120 m²", 95, 120),
    ("120+ m²", 120, 10000),
]


def _tranche_label(surf: float) -> str:
    for lbl, lo, hi in TRANCHES_M2:
        if lo <= surf < hi:
            return lbl
    return "Autre"


def _parse_photos(raw) -> list[str]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if isinstance(raw, list):
        return [str(u) for u in raw if str(u).startswith("http")]
    s = str(raw).strip()
    if not s or s in ("nan", "None", "[]"):
        return []
    for loader in (json.loads, ast.literal_eval):
        try:
            res = loader(s)
            if isinstance(res, list):
                return [str(u) for u in res if str(u).startswith("http")]
        except Exception:
            pass
    if "|" in s:
        return [p.strip() for p in s.split("|") if p.strip().startswith("http")]
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip().startswith("http")]
    return [s] if s.startswith("http") else []


def _sf(val, default: float = 0.0) -> float:
    try:
        f = float(val)
        return default if np.isnan(f) or np.isinf(f) else f
    except (TypeError, ValueError):
        return default


def _sfn(val) -> float | None:
    try:
        f = float(val)
        return None if np.isnan(f) or np.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _row_to_item(r: pd.Series) -> OpportunityItem:
    prix_aff = _sf(r.get("prix_reel", r.get("prix")))
    prix_ml = _sf(r.get("prix_predit"))
    prix_hyb = _sf(r.get("prix_hybride"), prix_ml)
    q10 = _sf(r.get("prix_predit_q10"), round(prix_ml * 0.88, 0))
    q90 = _sf(r.get("prix_predit_q90"), round(prix_ml * 1.12, 0))
    larg = _sf(r.get("intervalle_largeur_pct"))
    trav = _sf(r.get("cout_travaux_estime_euro"))
    val_ren = _sf(r.get("valeur_estimee_renovee"), prix_ml)
    ec_brut = _sf(r.get("ecart_hybride_pct"), round((prix_aff - prix_hyb) / max(prix_hyb, 1.0) * 100.0, 1))
    dec_nette = _sf(r.get("decote_nette_pct"), round((prix_aff + trav - val_ren) / max(val_ren, 1.0) * 100.0, 1))
    conf = int(_sf(r.get("score_confiance"), 50))
    verdict = determiner_verdict(ec_brut, conf, larg)
    return OpportunityItem(
        id=str(r.get("id", "")),
        titre=None if pd.isna(r.get("titre")) else str(r.get("titre")),
        ville=None if pd.isna(r.get("ville")) else str(r.get("ville")),
        quartier=None if pd.isna(r.get("quartier_clean", r.get("quartier"))) else str(r.get("quartier_clean", r.get("quartier"))),
        type_bien=None if pd.isna(r.get("type_bien")) else str(r.get("type_bien")),
        type_label=None if pd.isna(r.get("type_label")) else str(r.get("type_label")),
        surface_m2=_sf(r.get("surface_m2")),
        pieces=_sfn(r.get("pieces")),
        dpe=None if pd.isna(r.get("dpe")) or str(r.get("dpe")) == "inconnu" else str(r.get("dpe")),
        prix_affiche=prix_aff,
        prix_estime_ml=prix_ml,
        prix_hybride=prix_hyb,
        prix_q10=q10,
        prix_q90=q90,
        intervalle_largeur_pct=larg,
        valeur_estimee_renovee=val_ren,
        cout_travaux_estime=trav,
        niveau_travaux_cat=int(_sf(r.get("niveau_travaux_cat"), 0)),
        ecart_brut_pct=ec_brut,
        decote_nette_pct=dec_nette,
        score_confiance=conf,
        verdict=verdict,
        loyer_mensuel_estime=_sfn(r.get("loyer_mensuel_estime")),
        rendement_brut_pct=_sfn(r.get("rendement_brut_pct")),
        latitude=_sfn(r.get("latitude")),
        longitude=_sfn(r.get("longitude")),
        url=None if pd.isna(r.get("url")) else str(r.get("url")),
        photos_urls=_parse_photos(r.get("photos_urls")),
        source=None if pd.isna(r.get("source")) else str(r.get("source")),
    )


def _charger_df_predictions() -> pd.DataFrame:
    df = DB.lire_table_ou_csv(DB.TABLE_PREDICTIONS, DB.CSV_PREDICTIONS)
    if df.empty:
        return df
    if "est_ville_centre" in df.columns:
        vc = df[df["est_ville_centre"] == True]  # noqa: E712
        if not vc.empty:
            df = vc
    return df.assign(
        tranche_m2=pd.to_numeric(df["surface_m2"], errors="coerce").fillna(0).apply(_tranche_label)
    )


def lister_opportunites(
    ville: str | None = None,
    type_bien: str | None = None,
    tranche_m2: str | None = None,
    quartier: str | None = None,
    verdict: str | None = None,
    min_confiance: int = 45,
    mode: str = "avec_travaux",
    sort_by: str = "auto",
    limit: int = 50,
    offset: int = 0,
) -> OpportunitiesResponse:
    df = _charger_df_predictions()
    fraicheur_villes = DB.get_etat_scraping()
    if df.empty:
        return OpportunitiesResponse(
            kpis=OpportunitiesKpis(total_biens=0, nb_opportunites_fortes=0, nb_sous_evalues=0),
            facets=OpportunitiesFacets(villes=[], quartiers=[], tranches_m2=[t[0] for t in TRANCHES_M2]),
            total_filtered=0,
            items=[],
            fraicheur_villes=fraicheur_villes,
        )

    villes_dispo = sorted([str(v) for v in df["ville"].dropna().unique() if str(v).strip()]) if "ville" in df else []
    q_col_fac = "quartier_clean" if "quartier_clean" in df.columns else "quartier"
    df_fac = df[df["ville"].astype(str) == ville] if (ville and ville != "Toutes" and "ville" in df.columns) else df
    quartiers_dispo = sorted([str(q) for q in df_fac.get(q_col_fac, pd.Series()).dropna().unique() if str(q).strip()])

    sub = df.copy()
    if ville and ville != "Toutes":
        sub = sub[sub["ville"].astype(str) == ville]
    if type_bien and type_bien != "Tous":
        sub = sub[sub["type_bien"].astype(str) == type_bien]
    if tranche_m2 and tranche_m2 != "Toutes":
        sub = sub[sub["tranche_m2"] == tranche_m2]
    if quartier and quartier != "Tous":
        q_col = "quartier_clean" if "quartier_clean" in sub.columns else "quartier"
        sub = sub[sub[q_col].astype(str) == quartier]
    if min_confiance > 0 and "score_confiance" in sub.columns:
        sub = sub[pd.to_numeric(sub["score_confiance"], errors="coerce").fillna(0) >= min_confiance]

    items_all = [_row_to_item(r) for _, r in sub.iterrows()]
    if verdict and verdict != "Tous":
        items_all = [it for it in items_all if it.verdict == verdict]

    if sort_by == "auto":
        items_all.sort(key=lambda it: it.decote_nette_pct if mode == "avec_travaux" else it.ecart_brut_pct)
    elif sort_by == "rendement":
        items_all.sort(key=lambda it: it.rendement_brut_pct or -99.0, reverse=True)
    elif sort_by == "prix_asc":
        items_all.sort(key=lambda it: it.prix_affiche)
    elif sort_by == "confiance":
        items_all.sort(key=lambda it: it.score_confiance, reverse=True)

    ecarts = [it.decote_nette_pct if mode == "avec_travaux" else it.ecart_brut_pct for it in items_all]
    confs = [it.score_confiance for it in items_all]
    rdmts = [it.rendement_brut_pct for it in items_all if it.rendement_brut_pct is not None]

    kpis = OpportunitiesKpis(
        total_biens=len(items_all),
        nb_opportunites_fortes=sum(1 for it in items_all if it.verdict == "Opportunité forte"),
        nb_sous_evalues=sum(1 for it in items_all if it.verdict in ("Opportunité forte", "Sous-évalué")),
        decote_mediane_pct=round(float(np.median(ecarts)), 1) if ecarts else None,
        confiance_mediane=round(float(np.median(confs)), 0) if confs else None,
        rendement_brut_median_pct=round(float(np.median(rdmts)), 2) if rdmts else None,
    )

    page = items_all[offset : offset + limit]
    return OpportunitiesResponse(
        kpis=kpis,
        facets=OpportunitiesFacets(villes=villes_dispo, quartiers=quartiers_dispo, tranches_m2=[t[0] for t in TRANCHES_M2]),
        total_filtered=len(items_all),
        items=page,
        fraicheur_villes=fraicheur_villes,
    )


def detail_opportunite(ad_id: str) -> OpportunityDetail:
    df = _charger_df_predictions()
    if df.empty:
        raise ResourceNotFoundError(f"Offre {ad_id} introuvable.")
    sub = df[df["id"].astype(str) == str(ad_id)]
    if sub.empty:
        raise ResourceNotFoundError(f"Offre {ad_id} introuvable.")
    r = sub.iloc[0]
    base = _row_to_item(r)

    # Compléter photos / description depuis l'historique brut si nécessaire
    desc = "" if pd.isna(r.get("description")) else str(r.get("description"))
    photos = base.photos_urls
    if (not desc or not photos):
        df_brut = DB.lire_table_ou_csv(DB.TABLE_HIST_BRUT, DB.CSV_HIST_BRUT)
        if not df_brut.empty:
            b = df_brut[df_brut["id"].astype(str) == str(ad_id)]
            if not b.empty:
                rb = b.iloc[0]
                if not desc and pd.notna(rb.get("description")):
                    desc = str(rb.get("description"))
                if not photos:
                    photos = _parse_photos(rb.get("photos_urls"))

    shap_facteurs: list[ShapFactor] = []
    try:
        from src.ml import features as fe
        sd = json.loads(str(r.get("shap_top_facteurs_json", "{}") or "{}"))
        prix_base = max(base.prix_estime_ml, 1.0)
        for f in sd.get("facteurs", []):
            feat = str(f.get("feature", ""))
            imp_e = float(f.get("impact_euro", 0.0) or 0.0)
            imp_p = f.get("impact_pct")
            if imp_p is None or (float(imp_p) == 0.0 and abs(imp_e) > 0):
                imp_p = round(imp_e / prix_base * 100.0, 1)
            shap_facteurs.append(
                ShapFactor(
                    feature=feat,
                    label=fe.LABELS_FEATURES.get(feat, str(f.get("label", feat))),
                    impact_euro=round(imp_e, 0),
                    impact_pct=float(imp_p),
                    description=str(f.get("description") or fe.DESCRIPTIONS_FEATURES.get(feat, "")),
                )
            )
    except Exception:
        shap_facteurs = []

    comparables: list[ComparableItem] = []
    try:
        comps = json.loads(str(r.get("avm_comparables_json", "[]") or "[]"))
        comparables = [ComparableItem(**c) for c in comps]
    except Exception:
        comparables = []

    loyer_m = base.loyer_mensuel_estime or 0.0
    return OpportunityDetail(
        **{**base.model_dump(), "photos_urls": photos},
        description=desc,
        ges=None if pd.isna(r.get("ges")) or str(r.get("ges")) == "inconnu" else str(r.get("ges")),
        etage=_sfn(r.get("etage")),
        annee_construction=_sfn(r.get("annee_construction")),
        prix_notaire_dvf=_sfn(r.get("prix_notaire_dvf")),
        taux_nego_estime_pct=_sfn(r.get("taux_nego_estime_pct")),
        prix_cible_nego=_sfn(r.get("prix_cible_nego")),
        loyer_m2_estime=_sfn(r.get("loyer_m2_estime")),
        loyer_bas=round(loyer_m * 0.85, 0) if loyer_m > 0 else None,
        loyer_haut=round(loyer_m * 1.15, 0) if loyer_m > 0 else None,
        avm_comp_blended_euro=_sfn(r.get("avm_comp_blended_euro")),
        avm_comp_ecart_pct=_sfn(r.get("avm_comp_ecart_pct")),
        clip_etat_score=_sfn(r.get("clip_etat_score")),
        clip_score_travaux=_sfn(r.get("clip_score_travaux")),
        shap_facteurs=shap_facteurs,
        shap_resume="" if pd.isna(r.get("shap_resume")) else str(r.get("shap_resume")),
        comparables=comparables,
    )
