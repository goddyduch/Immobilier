"""Extraction unitaire synchrone (< 30 ms) alignée sur les colonnes_X du modèle."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml import features as fe
from backend.schemas.all import EstimateRequest
from backend.services.model_store import ModelSnapshot


def build_single_feature_row(req: EstimateRequest, snap: ModelSnapshot) -> pd.DataFrame:
    """Construit un DataFrame 1-ligne prêt pour inference.py sans recharger de gros CSV."""
    desc = req.description or ""
    equipements = []
    if req.has_ascenseur:
        equipements.append("ascenseur")
    if req.has_terrasse:
        equipements.append("terrasse")
    if req.has_jardin:
        equipements.append("jardin")
    if req.has_balcon:
        equipements.append("balcon")
    if req.has_parking:
        equipements.append("parking")
    if req.has_cave:
        equipements.append("cave")
    if equipements:
        desc = (desc + " " + " ".join(equipements)).strip()

    row_dict = {
        "id": "live_estimate",
        "type_bien": req.type_bien,
        "surface_m2": float(req.surface_m2),
        "pieces": int(req.pieces),
        "nbr_chambres": int(req.nbr_chambres) if req.nbr_chambres is not None else max(0, req.pieces - 1),
        "ville": req.ville,
        "code_postal": req.code_postal,
        "quartier": req.quartier or req.ville,
        "latitude": req.latitude,
        "longitude": req.longitude,
        "etage": req.etage,
        "dpe": req.dpe or "D",
        "ges": req.ges or "D",
        "annee_construction": req.annee_construction,
        "prix": float(req.prix_affiche) if req.prix_affiche else 180000.0,
        "description": desc,
        "has_ascenseur": int(req.has_ascenseur),
        "has_terrasse": int(req.has_terrasse),
        "has_jardin": int(req.has_jardin),
        "has_balcon": int(req.has_balcon),
        "has_parking": int(req.has_parking),
        "has_cave": int(req.has_cave),
    }
    df = pd.DataFrame([row_dict])
    df["dpe_num"] = df["dpe"].map(fe.ORDRE_DPE)
    df["ges_num"] = df["ges"].map(fe.ORDRE_DPE)

    df = fe.features_texte(df)
    df = fe.features_geo(df)
    df = fe.features_open_data_grand_poitiers(df)
    df = fe.features_surface_et_structure(df)

    # Heuristique LLM instantanée (0 appel réseau)
    heur = fe.calculer_heuristique_llm(df.iloc[0])
    df["llm_etat_score"] = heur["llm_etat_score"]
    df["llm_travaux_score"] = heur["llm_travaux_score"]
    df["llm_note_offre"] = heur["llm_note_offre"]

    # Si l'utilisateur a précisé manuellement un niveau de travaux (0..4), on le force proprement
    if req.niveau_travaux_manuel is not None:
        tarifs = {0: 0.0, 1: 400.0, 2: 750.0, 3: 1200.0, 4: 2000.0}
        cat = int(req.niveau_travaux_manuel)
        df["niveau_travaux_cat"] = cat
        df["cout_travaux_estime_euro"] = round(float(req.surface_m2) * tarifs.get(cat, 0.0), 0)
    else:
        df = fe.estimer_cout_travaux(df)

    # DVF local ultra-rapide via BallTree en RAM
    pm2_ref = 2200.0
    nb_dvf = 0
    if req.type_bien in snap.dvf_trees_by_type:
        tree, sub_dvf, med_glob = snap.dvf_trees_by_type[req.type_bien]
        pm2_ref = med_glob
        if req.latitude is not None and req.longitude is not None:
            pts = np.radians([[req.latitude, req.longitude]])
            dist, idx = tree.query(pts, k=min(40, len(sub_dvf)))
            dist_m = dist[0] * 6_371_000.0
            mask = dist_m <= 800.0
            if mask.any():
                nb_dvf = int(mask.sum())
                pm2_arr = sub_dvf["pm2"].to_numpy()[idx[0]][mask]
                poids = 1.0 / (dist_m[mask] + 50.0)
                pm2_ref = float(np.sum(pm2_arr * poids) / np.sum(poids))
    df["pm2_voisinage"] = round(pm2_ref, 0)
    df["pm2_dvf_pondere_dist"] = round(pm2_ref, 0)
    df["pm2_dvf_tranche_surface"] = round(pm2_ref, 0)
    df["nb_ventes_dvf"] = nb_dvf
    df["valeur_dvf_ajustee_taille"] = round(pm2_ref * float(req.surface_m2), 0)
    df["valeur_dvf_nette_travaux"] = round(
        max(10000.0, df["valeur_dvf_ajustee_taille"].iloc[0] - 0.70 * float(df["cout_travaux_estime_euro"].iloc[0])), 0
    )

    if "loyer_mensuel_estime" not in df.columns or pd.isna(df["loyer_mensuel_estime"].iloc[0]):
        loyer_m2 = float(df["loyer_m2_estime"].iloc[0]) if "loyer_m2_estime" in df.columns and pd.notna(df["loyer_m2_estime"].iloc[0]) else 12.5
        df["loyer_m2_estime"] = loyer_m2
        df["loyer_mensuel_estime"] = round(loyer_m2 * float(req.surface_m2), 0)

    # Alignement strict sur colonnes_X
    for col in snap.colonnes_X:
        if col not in df.columns:
            df[col] = "inconnu" if col in fe.CATEGORIELLES else np.nan
    for c in fe.CATEGORIELLES:
        if c in df.columns:
            df[c] = df[c].fillna("inconnu").astype(str)
    return df
