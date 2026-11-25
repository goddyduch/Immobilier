"""Analyse Notaires DVF & Simulateur de négociation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ingestion import db as DB
from backend.schemas.all import (
    DvfAnalysisResponse,
    DvfTrancheStat,
    NegotiationRequest,
    NegotiationResponse,
)
from backend.services.opportunities import TRANCHES_M2, _tranche_label


def analyser_dvf(ville: str | None = None, type_bien: str | None = None) -> DvfAnalysisResponse:
    df = DB.lire_table_ou_csv(DB.TABLE_PREDICTIONS, DB.CSV_PREDICTIONS)
    if df.empty:
        return DvfAnalysisResponse(
            nb_offres=0, nb_dvf_total=0, pm2_affiche_median=0.0, pm2_notaire_median=0.0,
            ecart_global_pct=0.0, taux_nego_median_pct=0.0, economie_mediane_euro=0.0,
            par_tranche=[], par_dpe=[],
        )

    sub = df.copy()
    if ville and ville != "Toutes" and "ville" in sub.columns:
        sub = sub[sub["ville"].astype(str) == ville]
    if type_bien and type_bien != "Tous" and "type_bien" in sub.columns:
        sub = sub[sub["type_bien"].astype(str) == type_bien]

    if sub.empty:
        sub = df.copy()

    surf = pd.to_numeric(sub["surface_m2"], errors="coerce").replace(0, np.nan)
    prix_aff = pd.to_numeric(sub.get("prix_reel", sub.get("prix")), errors="coerce")
    pm2_aff = (prix_aff / surf).dropna()
    pm2_not = pd.to_numeric(sub.get("pm2_voisinage"), errors="coerce").dropna()
    nego = pd.to_numeric(sub.get("taux_nego_estime_pct"), errors="coerce").dropna()
    eco = pd.to_numeric(sub.get("economie_nego_euro"), errors="coerce").dropna()

    med_aff = float(pm2_aff.median()) if not pm2_aff.empty else 0.0
    med_not = float(pm2_not.median()) if not pm2_not.empty else 0.0
    ecart_g = round((med_aff - med_not) / max(med_not, 1.0) * 100.0, 1) if med_not > 0 else 0.0

    sub["tranche_m2"] = surf.fillna(0).apply(_tranche_label)
    sub["pm2_aff"] = prix_aff / surf
    sub["pm2_not"] = pd.to_numeric(sub.get("pm2_voisinage"), errors="coerce")

    par_tranche: list[DvfTrancheStat] = []
    for lbl, _, _ in TRANCHES_M2:
        g = sub[sub["tranche_m2"] == lbl]
        if len(g) == 0:
            continue
        ma = float(g["pm2_aff"].median()) if g["pm2_aff"].notna().any() else 0.0
        mn = float(g["pm2_not"].median()) if g["pm2_not"].notna().any() else 0.0
        ng = float(pd.to_numeric(g.get("taux_nego_estime_pct"), errors="coerce").median()) if "taux_nego_estime_pct" in g else 0.0
        par_tranche.append(
            DvfTrancheStat(
                tranche_m2=lbl,
                nb_biens=int(len(g)),
                pm2_affiche_median=round(ma, 0),
                pm2_notaire_median=round(mn, 0),
                ecart_affiche_vs_dvf_pct=round((ma - mn) / max(mn, 1.0) * 100.0, 1) if mn > 0 else 0.0,
                taux_nego_conseille_pct=round(ng, 1) if not np.isnan(ng) else 0.0,
            )
        )

    par_dpe = []
    if "dpe" in sub.columns:
        for dpe in list("ABCDEFG"):
            g = sub[sub["dpe"].astype(str).str.upper() == dpe]
            if len(g) > 0:
                ng = float(pd.to_numeric(g.get("taux_nego_estime_pct"), errors="coerce").median())
                par_dpe.append({
                    "dpe": dpe,
                    "nb_biens": int(len(g)),
                    "taux_nego_median_pct": round(ng, 1) if not np.isnan(ng) else 0.0,
                })

    return DvfAnalysisResponse(
        nb_offres=int(len(sub)),
        nb_dvf_total=int(pd.to_numeric(sub.get("nb_ventes_dvf"), errors="coerce").fillna(0).sum()),
        pm2_affiche_median=round(med_aff, 0),
        pm2_notaire_median=round(med_not, 0),
        ecart_global_pct=ecart_g,
        taux_nego_median_pct=round(float(nego.median()), 1) if not nego.empty else 0.0,
        economie_mediane_euro=round(float(eco.median()), 0) if not eco.empty else 0.0,
        par_tranche=par_tranche,
        par_dpe=par_dpe,
    )


def simuler_negociation(req: NegotiationRequest) -> NegotiationResponse:
    marge = 4.0
    args = []

    dpe = req.dpe.upper().strip()
    if dpe in ("F", "G"):
        marge += 5.0
        args.append(f"Passoire énergétique (DPE {dpe}) : décote travaux / interdiction locative (+5 % de levier).")
    elif dpe == "E":
        marge += 2.5
        args.append("DPE E : travaux d'isolation à anticiper (+2,5 % de levier).")

    if req.travaux_euro > 0:
        ratio_trav = req.travaux_euro / req.prix_affiche * 100.0
        bonus = min(6.0, ratio_trav * 0.45)
        marge += bonus
        args.append(f"Enveloppe travaux estimée à {req.travaux_euro:,.0f} € (+{bonus:.1f} % d'argument de négo).".replace(",", " "))

    prix_ref = None
    if req.prix_notaire_ref_m2 and req.prix_notaire_ref_m2 > 0:
        prix_ref = round(req.prix_notaire_ref_m2 * req.surface_m2 * 1.08, 0)
        surcote = (req.prix_affiche - prix_ref) / prix_ref * 100.0
        if surcote > 3.0:
            marge += min(6.0, surcote * 0.5)
            args.append(f"Prix affiché {surcote:+.1f} % au-dessus du référentiel notarial DVF recalé ({prix_ref:,.0f} €).".replace(",", " "))

    marge = round(min(18.0, max(1.5, marge)), 1)
    offre_cons = round(req.prix_affiche * (1.0 - marge / 100.0), 0)
    offre_prud = round(req.prix_affiche * (1.0 - max(1.5, marge - 2.5) / 100.0), 0)
    offre_off = round(req.prix_affiche * (1.0 - min(22.0, marge + 3.0) / 100.0), 0)

    if not args:
        args.append("Bien dans les prix du marché : négociation standard de 3 à 4 % recommandée.")

    return NegotiationResponse(
        prix_ref_dvf_total=prix_ref,
        offre_prudente=offre_prud,
        offre_conseillee=offre_cons,
        offre_offensive=offre_off,
        marge_conseillee_pct=marge,
        economie_estimee_euro=round(req.prix_affiche - offre_cons, 0),
        arguments=args,
    )
