"""Service d'inférence unitaire (< 250 ms) : ML + Q10/Q90 + AVM + Confiance + Verdict + SHAP."""

from __future__ import annotations

import json
import time
import numpy as np
import pandas as pd

from src.ml import avm as AVMComparable
from src.ml import train as tr
from src.ml import features as fe
from backend import config
from backend.schemas.all import ComparableItem, EstimateRequest, EstimateResponse, ShapFactor
from backend.services.model_store import ModelSnapshot
from backend.services.single_features import build_single_feature_row


def determiner_verdict(ecart_pct: float | None, score_confiance: int, intervalle_largeur_pct: float = 0.0) -> str:
    if score_confiance < config.SCORE_CONFIANCE_MIN_AFFICHAGE or intervalle_largeur_pct > 55.0:
        return "Trop incertain"
    if ecart_pct is None:
        return "Estimation"
    if ecart_pct <= -12.0 and score_confiance >= config.SCORE_CONFIANCE_OPPORTUNITE_FORTE:
        return "Opportunité forte"
    if ecart_pct <= -5.0:
        return "Sous-évalué"
    if ecart_pct < 7.0:
        return "Aligné"
    return "Surévalué"


def estimate_single(req: EstimateRequest, snap: ModelSnapshot) -> EstimateResponse:
    t0 = time.perf_counter()
    df_feat = build_single_feature_row(req, snap)
    X = df_feat[snap.colonnes_X].copy()

    prix_ml = float(np.round(snap.modele.predict(X)[0], 0))

    # Valeur rénovée équivalente
    X_renov = X.copy()
    for col, val in [("txt_a_renover", 0.0), ("llm_travaux_score", 0.0), ("clip_score_travaux", 0.0)]:
        if col in X_renov.columns:
            X_renov[col] = val
    if "llm_etat_score" in X_renov.columns:
        X_renov["llm_etat_score"] = max(float(X_renov["llm_etat_score"].iloc[0] or 3.0), 4.0)
    if "valeur_dvf_nette_travaux" in X_renov.columns and "valeur_dvf_ajustee_taille" in X_renov.columns:
        X_renov["valeur_dvf_nette_travaux"] = X_renov["valeur_dvf_ajustee_taille"]
    val_renov = float(max(prix_ml, np.round(snap.modele.predict(X_renov)[0], 0)))

    # Q10 / Q90
    if snap.modele_q10 is not None and snap.modele_q90 is not None:
        q10 = float(min(np.round(snap.modele_q10.predict(X)[0], 0), prix_ml * 0.97))
        q90 = float(max(np.round(snap.modele_q90.predict(X)[0], 0), prix_ml * 1.03))
    else:
        q10 = round(prix_ml * 0.88, 0)
        q90 = round(prix_ml * 1.12, 0)
    larg_pct = round((q90 - q10) / max(prix_ml, 1.0) * 100.0, 1)

    # AVM Comparables (sur les données pré-préparées en RAM)
    avm_euro = None
    comp_nb = 0
    comp_sim = 0.0
    comparables: list[ComparableItem] = []
    if req.latitude is not None and req.longitude is not None:
        # Pré-filtrage spatial ultra-rapide (< 2 ms) pour ne passer que ~60 voisins à l'AVM
        df_dvf_sub = snap.df_dvf_raw
        if req.type_bien in snap.dvf_trees_by_type:
            tree, sub_tree_df, _ = snap.dvf_trees_by_type[req.type_bien]
            k_q = min(60, len(sub_tree_df))
            if k_q > 0:
                _, idxs = tree.query(np.radians([[float(req.latitude), float(req.longitude)]]), k=k_q)
                df_dvf_sub = sub_tree_df.iloc[idxs[0]]
        elif snap.df_dvf_raw is not None and not snap.df_dvf_raw.empty:
            lat_s = pd.to_numeric(snap.df_dvf_raw.get("latitude"), errors="coerce")
            lon_s = pd.to_numeric(snap.df_dvf_raw.get("longitude"), errors="coerce")
            df_dvf_sub = snap.df_dvf_raw[
                (lat_s - float(req.latitude)).abs().le(0.03)
                & (lon_s - float(req.longitude)).abs().le(0.04)
            ]

        df_marche_sub = snap.df_marche_comp
        if df_marche_sub is not None and not df_marche_sub.empty:
            lat_m = pd.to_numeric(df_marche_sub.get("latitude"), errors="coerce")
            lon_m = pd.to_numeric(df_marche_sub.get("longitude"), errors="coerce")
            df_marche_sub = df_marche_sub[
                (lat_m - float(req.latitude)).abs().le(0.03)
                & (lon_m - float(req.longitude)).abs().le(0.04)
            ]

        df_avm = AVMComparable.calculer_avm_comparables(
            df_feat, df_marche=df_marche_sub, df_dvf=df_dvf_sub, top_k=8
        )
        v_avm = df_avm["avm_comp_blended_euro"].iloc[0]
        if pd.notna(v_avm) and float(v_avm) > 0:
            avm_euro = float(round(float(v_avm), 0))
        comp_nb = int(df_avm["avm_comp_nb"].iloc[0] or 0)
        comp_sim = float(df_avm["avm_comp_score_moyen"].iloc[0] or 0.0)
        try:
            raw_comps = json.loads(str(df_avm["avm_comparables_json"].iloc[0] or "[]"))
            comparables = [ComparableItem(**c) for c in raw_comps[:5]]
        except Exception:
            comparables = []

    prix_hybride = round(0.75 * prix_ml + 0.25 * avm_euro, 0) if avm_euro else prix_ml

    # Resserrement de la fourchette autour de prix_hybride (~14-27 % au lieu de 30-70 %)
    max_half_pct = 0.11 if comp_nb >= 4 else 0.135
    if snap.modele_q10 is not None and snap.modele_q90 is not None:
        raw_q10 = float(np.round(snap.modele_q10.predict(X)[0], 0))
        raw_q90 = float(np.round(snap.modele_q90.predict(X)[0], 0))
        raw_half = max(0.06 * prix_hybride, (raw_q90 - raw_q10) / 2.0)
        half = min(raw_half, max_half_pct * prix_hybride)
    else:
        half = 0.11 * prix_hybride
    q10 = round(prix_hybride - half, 0)
    q90 = round(prix_hybride + half, 0)
    larg_pct = round((q90 - q10) / max(prix_hybride, 1.0) * 100.0, 1)

    # Score de confiance (0-100) aligné sur Prediction.py
    larg_rel = (q90 - q10) / max(prix_ml, 1.0)
    sc_int = max(0.0, min(1.0, 1.0 - larg_rel / 0.55)) * 40.0
    sc_comp = min(1.0, comp_nb / 6.0) * 20.0 + (comp_sim / 100.0) * 15.0
    sc_data = (
        (8.0 if req.dpe in list("ABCDEFG") else 0.0)
        + (6.0 if req.annee_construction is not None else 0.0)
        + (6.0 if (req.latitude is not None and req.longitude is not None) else 0.0)
        + 5.0
    )
    score_confiance = int(round(max(0.0, min(100.0, sc_int + sc_comp + sc_data))))

    # Négociation
    df_nego = df_feat.copy()
    df_nego["type_label"] = "Maison" if req.type_bien == "house" else "Appartement"
    df_nego["prix_predit"] = prix_ml
    df_nego["pm2_predit"] = round(prix_ml / float(req.surface_m2), 0)
    df_nego["prix_reel"] = float(req.prix_affiche) if req.prix_affiche else prix_hybride
    df_nego["pm2_reel"] = round(df_nego["prix_reel"] / float(req.surface_m2), 0)
    df_nego["ecart_pct"] = round((float(df_nego["prix_reel"].iloc[0]) - prix_ml) / max(prix_ml, 1.0) * 100.0, 1)
    df_nego["duree_en_ligne_jours"] = 0.0
    df_nego["evolution_prix_euro"] = 0.0
    df_nego["a_baisse_prix"] = 0
    df_nego = tr.estimer_negociation(df_nego)
    taux_nego = float(df_nego["taux_nego_estime_pct"].iloc[0] or 4.0)
    prix_cible_nego = float(df_nego["prix_cible_nego"].iloc[0] or round(prix_hybride * (1 - taux_nego / 100.0), 0))

    cout_trav = float(df_feat["cout_travaux_estime_euro"].iloc[0] or 0.0)
    niv_trav = int(df_feat["niveau_travaux_cat"].iloc[0] or 0)
    loyer_mens = float(df_feat["loyer_mensuel_estime"].iloc[0] or 0.0)
    loyer_bas = round(loyer_mens * 0.85, 0)
    loyer_haut = round(loyer_mens * 1.15, 0)

    ecart_brut = None
    decote_nette = None
    rendement_brut = None
    if req.prix_affiche:
        ecart_brut = round((req.prix_affiche - prix_hybride) / max(prix_hybride, 1.0) * 100.0, 1)
        decote_nette = round((req.prix_affiche + cout_trav - val_renov) / max(val_renov, 1.0) * 100.0, 1)
        rendement_brut = round((loyer_mens * 12.0) / req.prix_affiche * 100.0, 2)

    verdict = determiner_verdict(ecart_brut, score_confiance, larg_pct)

    # SHAP (impact en % du prix + euros + plus de facteurs + descriptions métier)
    shap_facteurs: list[ShapFactor] = []
    if snap.shap_explainer is not None:
        try:
            pipe = snap.modele.regressor_
            prepa = pipe.named_steps["prepa"]
            X_trans = prepa.transform(X)
            feat_names = [n.split("__", 1)[-1] for n in prepa.get_feature_names_out()]
            sv = snap.shap_explainer.shap_values(X_trans)[0]
            impacts_pct = (np.exp(sv) - 1.0) * 100.0
            impacts = prix_ml * (np.exp(sv) - 1.0)
            paires = [
                ShapFactor(
                    feature=feat_names[j],
                    label=fe.LABELS_FEATURES.get(feat_names[j], feat_names[j]),
                    impact_euro=round(float(impacts[j]), 0),
                    impact_pct=round(float(impacts_pct[j]), 1),
                    description=fe.DESCRIPTIONS_FEATURES.get(feat_names[j], ""),
                )
                for j in range(len(feat_names))
                if abs(impacts_pct[j]) >= 0.2 or abs(impacts[j]) >= 100.0
            ]
            paires.sort(key=lambda x: abs(x.impact_pct), reverse=True)
            shap_facteurs = paires[:35]
        except Exception:
            shap_facteurs = []

    from src.ingestion.territoires import est_ville_centre, lister_noms_villes
    ville_couverte = est_ville_centre(req.code_postal, req.ville)
    avertissement_couv = None
    if not ville_couverte and (req.ville or req.code_postal):
        avertissement_couv = (
            f"Hors périmètre V1 : Justeprix couvre actuellement les 12 villes-centres uniquement "
            f"({', '.join(lister_noms_villes())}). Estimation donnée à titre indicatif."
        )

    latence_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    return EstimateResponse(
        prix_ml=prix_ml,
        prix_q10=q10,
        prix_q90=q90,
        intervalle_largeur_pct=larg_pct,
        prix_avm=avm_euro,
        prix_hybride=prix_hybride,
        valeur_estimee_renovee=val_renov,
        cout_travaux_estime=cout_trav,
        niveau_travaux_cat=niv_trav,
        ecart_brut_pct=ecart_brut,
        decote_nette_pct=decote_nette,
        score_confiance=score_confiance,
        verdict=verdict,
        taux_nego_estime_pct=taux_nego,
        prix_cible_nego=prix_cible_nego,
        loyer_mensuel_estime=loyer_mens,
        loyer_bas=loyer_bas,
        loyer_haut=loyer_haut,
        rendement_brut_pct=rendement_brut,
        ville_couverte=ville_couverte,
        avertissement_couverture=avertissement_couv,
        shap_facteurs=shap_facteurs,
        comparables=comparables,
        latence_ms=latence_ms,
    )
