"""
AVMComparable.py — Estimation AVM par biens comparables (« à côté ») intégrant les travaux.

Combine deux viviers de comparables autour de chaque bien cible :
  1. Les ventes réelles enregistrées par les notaires (DVF Grand Poitiers)
  2. Les annonces immobilières locales (offres_historique_brut / features)

Prend en compte :
  - Distance géographique (Haversine)
  - Proximité de surface et type de bien (Maison / Appartement)
  - Écart de DPE
  - Différentiel de coût des travaux estimés (`cout_travaux_estime_euro`)
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd

ORDRE_DPE = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
COEF_DECODAGE_TRAVAUX = 0.70  # 1 € de travaux impacte en moyenne ~0.70 € la valeur de marché immédiate
COEF_DVF_VERS_AFFICHE = 1.08  # Les actes DVF sont nets vendeur après négo (~8% sous le prix affiché FAI)
ELASTICITE_SURFACE_PM2 = 0.20  # Dégressivité du €/m² avec la surface : pm2 ~ S^(-0.20)


def _haversine_vec(lat1: float, lon1: float, lat2_arr: np.ndarray, lon2_arr: np.ndarray) -> np.ndarray:
    r = 6371.0
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2_arr), np.radians(lon2_arr)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _preparer_dvf_comp(df_dvf: pd.DataFrame | None) -> pd.DataFrame:
    if df_dvf is None or df_dvf.empty:
        return pd.DataFrame()
    d = df_dvf.copy()
    d["latitude"] = pd.to_numeric(d.get("latitude"), errors="coerce")
    d["longitude"] = pd.to_numeric(d.get("longitude"), errors="coerce")
    d["surface_m2"] = pd.to_numeric(d.get("surface_reelle_bati"), errors="coerce")
    d["prix"] = pd.to_numeric(d.get("valeur_fonciere"), errors="coerce")
    d = d[d["latitude"].notna() & d["longitude"].notna() & (d["surface_m2"] >= 9) & (d["prix"] >= 10000)].copy()
    d["pm2"] = d["prix"] / d["surface_m2"]
    d = d[(d["pm2"] >= 500) & (d["pm2"] <= 6000)].copy()
    d["type_bien"] = np.where(d.get("type_local", "").astype(str).str.lower().str.contains("maison"), "house", "flat")
    if "date_mutation" in d.columns:
        dt = pd.to_datetime(d["date_mutation"], errors="coerce")
        ref = pd.Timestamp.now()
        d["age_mois"] = ((ref - dt).dt.days / 30.4).clip(lower=0, upper=60).fillna(12.0)
        d["date_label"] = dt.dt.strftime("%Y-%m").fillna("DVF")
    else:
        d["age_mois"] = 12.0
        d["date_label"] = "DVF"
    return d.reset_index(drop=True)


def _col_binaire(df: pd.DataFrame, noms: tuple[str, ...]) -> np.ndarray:
    """Retourne un indicateur binaire (0/1) si au moins une colonne candidate vaut 1/True."""
    out = np.zeros(len(df), dtype=float)
    for nom in noms:
        if nom in df.columns:
            s = df[nom]
            if s.dtype == bool:
                val = s.astype(float).values
            else:
                num = pd.to_numeric(s, errors="coerce").fillna(0.0).values
                val = (num > 0).astype(float)
            out = np.maximum(out, val)
    return out


def _val_binaire_row(row: pd.Series, noms: tuple[str, ...]) -> float:
    for nom in noms:
        v = row.get(nom)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if isinstance(v, (bool, np.bool_)):
            if v:
                return 1.0
        else:
            try:
                if float(v) > 0:
                    return 1.0
            except (TypeError, ValueError):
                if str(v).strip().lower() in ("true", "oui", "1"):
                    return 1.0
    return 0.0


def _preparer_marche_comp(df_hist: pd.DataFrame | None) -> pd.DataFrame:
    if df_hist is None or df_hist.empty:
        return pd.DataFrame()
    d = df_hist.copy()
    d["latitude"] = pd.to_numeric(d.get("latitude"), errors="coerce")
    d["longitude"] = pd.to_numeric(d.get("longitude"), errors="coerce")
    d["surface_m2"] = pd.to_numeric(d.get("surface_m2"), errors="coerce")
    d["prix"] = pd.to_numeric(d.get("prix"), errors="coerce")
    d = d[d["latitude"].notna() & d["longitude"].notna() & (d["surface_m2"] >= 9) & (d["prix"] >= 10000)].copy()
    d["dpe_num"] = d.get("dpe", pd.Series(dtype=str)).astype(str).str.upper().map(ORDRE_DPE)
    d["travaux_euro"] = pd.to_numeric(d.get("cout_travaux_estime_euro", 0), errors="coerce").fillna(0.0)
    d["travaux_m2"] = d["travaux_euro"] / d["surface_m2"]
    d["prix_renov_eq"] = d["prix"] + COEF_DECODAGE_TRAVAUX * d["travaux_euro"]
    # Prix/m² ramené à un état « rénové équivalent » pour comparer équitablement avec le bien cible
    d["pm2_ajuste_renov"] = d["prix_renov_eq"] / d["surface_m2"]
    d["pm2"] = d["prix"] / d["surface_m2"]
    d["comp_parking"] = _col_binaire(d, ("has_parking", "has_garage", "has_box", "parking"))
    d["comp_exterieur"] = _col_binaire(d, ("has_terrasse", "has_jardin"))
    d["comp_ascenseur"] = _col_binaire(d, ("has_ascenseur",))
    return d.reset_index(drop=True)


def calculer_avm_comparables(
    df_cibles: pd.DataFrame,
    df_marche: pd.DataFrame | None = None,
    df_dvf: pd.DataFrame | None = None,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Calcule pour chaque ligne de `df_cibles` :
      - `avm_comp_dvf_euro` : estimation brute issue des ventes notariées DVF voisines
      - `avm_comp_dvf_nette_travaux_euro` : estimation DVF ajustée des travaux du bien cible
      - `avm_comp_marche_euro` : estimation issue des annonces voisines ajustée du différentiel de travaux
      - `avm_comp_blended_euro` : synthèse pondérée (60% DVF nette travaux + 40% Marché ajusté travaux)
      - `avm_comp_ecart_pct` : écart % entre prix affiché et `avm_comp_blended_euro`
      - `avm_comp_nb` : nombre total de comparables retenus
      - `avm_comp_score_moyen` : similarité moyenne (0-100) des meilleurs comparables
      - `avm_comparables_json` : top 5 comparables détaillés (JSON) pour affichage Dashboard
    """
    res = df_cibles.copy()
    dvf_c = _preparer_dvf_comp(df_dvf)
    marche_c = _preparer_marche_comp(df_marche)

    n = len(res)
    out_dvf = np.full(n, np.nan)
    out_dvf_net = np.full(n, np.nan)
    out_marche = np.full(n, np.nan)
    out_blended = np.full(n, np.nan)
    out_nb = np.zeros(n, dtype=int)
    out_score = np.full(n, np.nan)
    out_json = ["[]"] * n

    for idx_pos, (_, row) in enumerate(res.iterrows()):
        lat = pd.to_numeric(row.get("latitude"), errors="coerce")
        lon = pd.to_numeric(row.get("longitude"), errors="coerce")
        surf = pd.to_numeric(row.get("surface_m2"), errors="coerce")
        if pd.isna(lat) or pd.isna(lon) or pd.isna(surf) or surf <= 0:
            continue

        t_bien = str(row.get("type_bien", "flat"))
        ad_id = str(row.get("id", ""))
        dpe_b = ORDRE_DPE.get(str(row.get("dpe", "")).upper(), np.nan)
        trav_b = float(pd.to_numeric(row.get("cout_travaux_estime_euro", 0), errors="coerce") or 0.0)
        trav_m2_b = trav_b / surf
        b_parking = _val_binaire_row(row, ("has_parking", "has_garage", "has_box", "parking"))
        b_ext = _val_binaire_row(row, ("has_terrasse", "has_jardin"))
        b_asc = _val_binaire_row(row, ("has_ascenseur",))
        b_etage = float(pd.to_numeric(row.get("etage"), errors="coerce") or 0.0)

        meilleurs_fiches = []

        # --- 1. Comparables DVF Notaires ---
        val_dvf = np.nan
        val_dvf_net = np.nan
        nb_dvf = 0
        if not dvf_c.empty:
            sub_dvf = dvf_c[dvf_c["type_bien"] == t_bien]
            if not sub_dvf.empty:
                dists = _haversine_vec(lat, lon, sub_dvf["latitude"].values, sub_dvf["longitude"].values)
                ratio_s = np.abs(sub_dvf["surface_m2"].values - surf) / surf
                mask = (dists <= 0.8) & (ratio_s <= 0.20)
                if mask.sum() < 4:
                    mask = (dists <= 1.5) & (ratio_s <= 0.35)
                if mask.sum() < 4:
                    mask = (dists <= 3.0) & (ratio_s <= 0.50)
                if mask.sum() > 0:
                    cand = sub_dvf[mask].copy()
                    cand["dist_km"] = dists[mask]
                    cand["ratio_s"] = ratio_s[mask]
                    dist_comp = (
                        0.45 * np.minimum(1.0, cand["dist_km"] / 1.5)
                        + 0.35 * np.minimum(1.0, cand["ratio_s"] / 0.35)
                        + 0.20 * np.minimum(1.0, cand["age_mois"] / 24.0)
                    )
                    cand["sim_pct"] = np.round(100.0 * np.exp(-1.5 * dist_comp), 1)
                    # Décroissance temporelle DVF (demi-vie ~18 mois) pour prioriser les actes récents
                    cand["poids"] = np.exp(-2.5 * dist_comp) * np.exp(-cand["age_mois"] / 18.0)
                    top_d = cand.sort_values("sim_pct", ascending=False).head(top_k)
                    nb_dvf = len(top_d)
                    # Correction dégressivité de surface + recalage DVF net signé -> équivalent affiché FAI (+8%)
                    pm2_corr = top_d["pm2"].values * np.power(np.clip(top_d["surface_m2"].values / surf, 0.5, 1.8), ELASTICITE_SURFACE_PM2)
                    pm2_d = np.average(pm2_corr, weights=top_d["poids"].values)
                    val_dvf = round(float(pm2_d * surf * COEF_DVF_VERS_AFFICHE), 0)
                    val_dvf_net = round(max(10000.0, val_dvf - COEF_DECODAGE_TRAVAUX * trav_b), 0)
                    for _, c_row in top_d.head(3).iterrows():
                        commune = str(c_row.get("nom_commune", "Poitiers") or "Poitiers")
                        meilleurs_fiches.append({
                            "source": "DVF Notaire",
                            "titre": f"Vente {commune} ({c_row['date_label']})",
                            "distance_m": int(round(c_row["dist_km"] * 1000)),
                            "surface_m2": round(float(c_row["surface_m2"]), 1),
                            "prix": round(float(c_row["prix"]), 0),
                            "pm2": round(float(c_row["pm2"]), 0),
                            "travaux_euro": 0,
                            "similarite_pct": float(c_row["sim_pct"]),
                            "url": "",
                        })

        # --- 2. Comparables Marché (Annonces locales) avec différentiel travaux & équipements ---
        val_marche = np.nan
        nb_marche = 0
        nb_marche_600m = 0
        if not marche_c.empty:
            sub_m = marche_c[
                (marche_c["type_bien"] == t_bien)
                & (marche_c["id"].astype(str) != ad_id)
            ]
            if not sub_m.empty:
                dists = _haversine_vec(lat, lon, sub_m["latitude"].values, sub_m["longitude"].values)
                ratio_s = np.abs(sub_m["surface_m2"].values - surf) / surf
                mask = (dists <= 0.8) & (ratio_s <= 0.20)
                if mask.sum() < 4:
                    mask = (dists <= 1.5) & (ratio_s <= 0.35)
                if mask.sum() < 4:
                    mask = (dists <= 3.0) & (ratio_s <= 0.50)
                if mask.sum() > 0:
                    cand = sub_m[mask].copy()
                    cand["dist_km"] = dists[mask]
                    cand["ratio_s"] = ratio_s[mask]
                    nb_marche_600m = int((cand["dist_km"] <= 0.6).sum())
                    if not pd.isna(dpe_b):
                        dpe_gap = np.abs(cand["dpe_num"].fillna(dpe_b).values - dpe_b) / 6.0
                    else:
                        dpe_gap = np.zeros(len(cand))
                    trav_gap = np.minimum(1.0, np.abs(cand["travaux_m2"].values - trav_m2_b) / 600.0)
                    dist_comp = (
                        0.40 * np.minimum(1.0, cand["dist_km"] / 1.5)
                        + 0.30 * np.minimum(1.0, cand["ratio_s"] / 0.35)
                        + 0.15 * dpe_gap
                        + 0.15 * trav_gap
                    )
                    cand["sim_pct"] = np.round(100.0 * np.exp(-1.5 * dist_comp), 1)
                    cand["poids"] = np.exp(-2.5 * dist_comp)
                    top_m = cand.sort_values("sim_pct", ascending=False).head(top_k)
                    nb_marche = len(top_m)

                    # Correctifs équipements pour ramener chaque comparable aux caractéristiques du bien cible :
                    # parking/garage (±8 000 €), terrasse/jardin (±5 %), ascenseur (±4 % si étage >= 2), DPE (±3 % / classe)
                    delta_parking = 8000.0 * (b_parking - top_m["comp_parking"].values)
                    coef_ext = 1.0 + 0.05 * (b_ext - top_m["comp_exterieur"].values)
                    if t_bien == "flat" and b_etage >= 2:
                        coef_asc = 1.0 + 0.04 * (b_asc - top_m["comp_ascenseur"].values)
                    else:
                        coef_asc = np.ones(len(top_m))
                    if not pd.isna(dpe_b):
                        diff_dpe = np.clip(top_m["dpe_num"].fillna(dpe_b).values - dpe_b, -4.0, 4.0)
                        coef_dpe = 1.0 + 0.03 * diff_dpe
                    else:
                        coef_dpe = np.ones(len(top_m))

                    prix_eq = np.maximum(10000.0, top_m["prix_renov_eq"].values * coef_ext * coef_asc * coef_dpe + delta_parking)
                    pm2_eq = prix_eq / top_m["surface_m2"].values
                    pm2_corr_m = pm2_eq * np.power(np.clip(top_m["surface_m2"].values / surf, 0.5, 1.8), ELASTICITE_SURFACE_PM2)
                    pm2_renov_eq = np.average(pm2_corr_m, weights=top_m["poids"].values)
                    val_marche = round(max(10000.0, float(pm2_renov_eq * surf) - COEF_DECODAGE_TRAVAUX * trav_b), 0)
                    for _, c_row in top_m.head(3).iterrows():
                        meilleurs_fiches.append({
                            "source": f"Annonce ({c_row.get('source', 'marche')})",
                            "titre": str(c_row.get("titre", "") or f"Bien {c_row['surface_m2']:.0f}m²")[:70],
                            "distance_m": int(round(c_row["dist_km"] * 1000)),
                            "surface_m2": round(float(c_row["surface_m2"]), 1),
                            "prix": round(float(c_row["prix"]), 0),
                            "pm2": round(float(c_row["pm2"]), 0),
                            "travaux_euro": int(round(float(c_row["travaux_euro"]), 0)),
                            "similarite_pct": float(c_row["sim_pct"]),
                            "url": str(c_row.get("url", "") or ""),
                        })

        meilleurs_fiches = sorted(meilleurs_fiches, key=lambda x: x["similarite_pct"], reverse=True)[:top_k]
        out_dvf[idx_pos] = val_dvf
        out_dvf_net[idx_pos] = val_dvf_net
        out_marche[idx_pos] = val_marche
        out_nb[idx_pos] = nb_dvf + nb_marche
        if meilleurs_fiches:
            out_score[idx_pos] = round(float(np.mean([f["similarite_pct"] for f in meilleurs_fiches])), 1)
        out_json[idx_pos] = json.dumps(meilleurs_fiches, ensure_ascii=False)

        if not np.isnan(val_dvf_net) and not np.isnan(val_marche):
            w_marche = 0.65 if nb_marche_600m >= 4 else 0.50
            out_blended[idx_pos] = round((1.0 - w_marche) * val_dvf_net + w_marche * val_marche, 0)
        elif not np.isnan(val_dvf_net):
            out_blended[idx_pos] = val_dvf_net
        elif not np.isnan(val_marche):
            out_blended[idx_pos] = val_marche

    res["avm_comp_dvf_euro"] = out_dvf
    res["avm_comp_dvf_nette_travaux_euro"] = out_dvf_net
    res["avm_comp_marche_euro"] = out_marche
    res["avm_comp_blended_euro"] = out_blended
    res["avm_comp_nb"] = out_nb
    res["avm_comp_score_moyen"] = out_score
    res["avm_comparables_json"] = out_json

    if "prix_reel" in res.columns:
        denom = res["avm_comp_blended_euro"].replace(0, np.nan)
        res["avm_comp_ecart_pct"] = np.round((res["prix_reel"] - denom) / denom * 100.0, 1)

    return res


def comparer_ml_vs_avm(df_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Compare sur le même échantillon la précision du modèle ML seul, de l'AVM par comparables seul,
    et d'un hybride (75% ML + 25% AVM Comparables).
    """
    from sklearn.metrics import r2_score

    if df_pred is None or df_pred.empty or "avm_comp_blended_euro" not in df_pred.columns:
        return pd.DataFrame()

    sub = df_pred.dropna(subset=["prix_reel", "prix_predit", "avm_comp_blended_euro"]).copy()
    sub = sub[(sub["prix_reel"] > 0) & (sub["prix_predit"] > 0) & (sub["avm_comp_blended_euro"] > 0)]
    if len(sub) < 10:
        return pd.DataFrame()

    y = sub["prix_reel"].to_numpy(dtype=float)
    y_ml = sub["prix_predit"].to_numpy(dtype=float)
    y_avm = sub["avm_comp_blended_euro"].to_numpy(dtype=float)
    y_hyb = 0.75 * y_ml + 0.25 * y_avm

    lignes = []
    for nom, pred in [
        ("ML seul (HistGradientBoosting)", y_ml),
        ("AVM Comparables seul (travaux inclus)", y_avm),
        ("Hybride (75% ML + 25% AVM Comp.)", y_hyb),
    ]:
        err = np.abs(y - pred)
        ape = err / y * 100.0
        lignes.append({
            "Approche": nom,
            "Biens": len(sub),
            "R²": round(float(r2_score(y, pred)), 3),
            "MAE (€)": round(float(np.mean(err)), 0),
            "MdAPE (%)": round(float(np.median(ape)), 1),
            "<= 10%": round(float(np.mean(ape <= 10.0) * 100.0), 1),
            "<= 15%": round(float(np.mean(ape <= 15.0) * 100.0), 1),
        })
    return pd.DataFrame(lignes)
