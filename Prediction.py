"""
Script 3/3 : Prédiction (Inférence) sur les Offres Récentes — IA Immobilier Poitiers.

Responsabilité unique :
- Lit directement la table des offres récentes APRÈS Feature Engineering (`offres_recentes_features`).
- Charge le modèle pré-entraîné (`modele_poitiers.joblib`) sans le ré-entraîner.
- Calcule les prix prédits, les écarts par rapport aux ventes notariées DVF et les marges de négociation.
- Met à jour `offres_recentes_features` dans PostgreSQL et `csv/annonces_recentes_features.csv`.
- Fournit `charger_donnees_dashboard()` pour le démarrage instantané de `Dashboard.py`.

Usage :
    python Prediction.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor

import DB
import Entrainement as tr
import FeatureEngineering as fe


def predire_sur_offres_recentes(
    df_recent_feat: pd.DataFrame | None = None,
    model_path: str = tr.CHEMIN_MODELE,
) -> pd.DataFrame:
    """
    ÉTAPE 3 — PRÉDICTION (directement sur la table récente APRÈS Feature Engineering) :
    - Lit directement la table `offres_recentes_features` (`csv/annonces_recentes_features.csv`)
    - Charge le modèle sauvegardé (`modele_poitiers.joblib`) sans le ré-entraîner
    - Calcule les prix prédits, les écarts Notaire DVF et les marges de négociation
    - Sauvegarde `offres_recentes_features` dans PostgreSQL et `csv/annonces_recentes_features.csv`
    """
    if not os.path.exists(model_path):
        tr.entrainer_et_sauvegarder_modele(model_path=model_path, afficher=False)

    artefact = joblib.load(model_path)
    modele = artefact["modele"]
    colonnes_X = artefact["colonnes_X"]
    oof_par_id = artefact.get("oof_par_id", {})

    if df_recent_feat is None:
        df_recent_feat = DB.lire_table_ou_csv(DB.TABLE_RECENT_FEAT, DB.CSV_RECENT_FEAT)
        if df_recent_feat.empty or "dist_centre_km" not in df_recent_feat.columns:
            _, df_recent_feat = fe.construire_tables_features(verbose=False)

    # Nettoyage préventif : retire toute ancienne colonne de prédiction avant d'inférer
    df_recent_feat = fe.supprimer_colonnes_prediction(df_recent_feat)

    print(
        f"[Prédiction] Lecture directe de la table après Feature Engineering "
        f"'{DB.TABLE_RECENT_FEAT}' ({len(df_recent_feat)} lignes, {len(df_recent_feat.columns)} colonnes)..."
    )
    X_rec, y_rec = fe.construire_X_y(df_recent_feat)
    for col in colonnes_X:
        if col not in X_rec.columns:
            X_rec[col] = np.nan
    X_rec = X_rec[colonnes_X]

    df_res = df_recent_feat.loc[X_rec.index].copy()
    preds_directes = np.round(modele.predict(X_rec), 0)

    # Si l'annonce faisait partie de l'historique d'entraînement, on utilise sa prédiction hors-échantillon (OOF)
    # Sinon (nouvelle annonce jamais vue à l'entraînement), on utilise la prédiction directe du modèle sauvegardé
    ids_str = df_res["id"].astype(str).tolist()
    y_pred = np.array([oof_par_id.get(i, p) for i, p in zip(ids_str, preds_directes)], dtype=float)

    df_res["prix_reel"] = np.round(y_rec.values, 0)
    df_res["prix_predit"] = np.round(y_pred, 0)
    df_res["ecart_euro"] = np.round(df_res["prix_reel"] - df_res["prix_predit"], 0)
    df_res["erreur_abs_euro"] = np.abs(df_res["ecart_euro"])
    df_res["ecart_pct"] = np.round(
        (df_res["prix_reel"] - df_res["prix_predit"]) / df_res["prix_predit"] * 100, 1
    )
    df_res["erreur_abs_pct"] = np.abs(df_res["ecart_pct"])
    df_res["pm2_reel"] = np.round(df_res["prix_reel"] / df_res["surface_m2"], 0)
    df_res["pm2_predit"] = np.round(df_res["prix_predit"] / df_res["surface_m2"], 0)
    df_res["quartier_clean"] = df_res["quartier"].fillna("Inconnu")
    df_res["type_label"] = np.where(df_res["type_bien"] == "house", "Maison", "Appartement")
    df_res["titre_propre"] = np.where(
        df_res["titre"].notna() & (df_res["titre"].astype(str).str.strip() != ""),
        df_res["titre"].astype(str),
        df_res["type_label"] + " " + df_res["surface_m2"].round(0).astype(int).astype(str) + " m²",
    )

    # Simulation de la valeur une fois rénovée (état = 4, 0 travaux) via le modèle HistGradientBoosting
    X_renov = X_rec.copy()
    for col_z in (
        "llm_travaux_score", "trav_rafraichissement", "trav_cuisine_sdb",
        "trav_menuiseries", "trav_elec_plomberie", "trav_gros_oeuvre",
        "cout_travaux_estime_euro", "cout_travaux_m2",
    ):
        if col_z in X_renov.columns:
            X_renov[col_z] = 0.0
    if "llm_etat_score" in X_renov.columns:
        X_renov["llm_etat_score"] = np.maximum(X_renov["llm_etat_score"].fillna(3.0), 4.0)
    if "valeur_dvf_nette_travaux" in X_renov.columns and "valeur_dvf_ajustee_taille" in X_renov.columns:
        X_renov["valeur_dvf_nette_travaux"] = X_renov["valeur_dvf_ajustee_taille"]
    preds_renov = np.round(modele.predict(X_renov), 0)
    df_res["valeur_estimee_renovee"] = np.maximum(df_res["prix_predit"], preds_renov)

    df_res = tr.estimer_negociation(df_res)

    # 1. Intervalles de prédiction [Q10, Q90]
    modele_q10 = artefact.get("modele_q10")
    modele_q90 = artefact.get("modele_q90")
    if modele_q10 is not None and modele_q90 is not None:
        q10 = np.round(modele_q10.predict(X_rec), 0)
        q90 = np.round(modele_q90.predict(X_rec), 0)
        df_res["prix_predit_q10"] = np.minimum(q10, df_res["prix_predit"] * 0.97)
        df_res["prix_predit_q90"] = np.maximum(q90, df_res["prix_predit"] * 1.03)
    else:
        df_res["prix_predit_q10"] = np.round(df_res["prix_predit"] * 0.88, 0)
        df_res["prix_predit_q90"] = np.round(df_res["prix_predit"] * 1.12, 0)
    df_res["intervalle_largeur_pct"] = np.round(
        (df_res["prix_predit_q90"] - df_res["prix_predit_q10"]) / df_res["prix_predit"].clip(lower=1.0) * 100.0, 1
    )

    # 2. AVM Comparables ("biens à côté") intégrant les travaux + Score Hybride (75% ML + 25% AVM)
    try:
        import AVMComparable
        df_hist_brut = DB.lire_table_ou_csv(DB.TABLE_HIST_FEAT, DB.CSV_HIST_FEAT)
        df_dvf = fe.telecharger_ou_charger_dvf(fe.CHEMIN_DVF_CSV, verbose=False)
        df_res = AVMComparable.calculer_avm_comparables(df_res, df_marche=df_hist_brut, df_dvf=df_dvf, top_k=5)
    except Exception as e:
        print(f"[Prédiction] AVM Comparable ignoré : {e}")

    avm_s = pd.to_numeric(df_res.get("avm_comp_blended_euro"), errors="coerce")
    df_res["prix_hybride"] = np.round(
        np.where(avm_s.notna() & (avm_s > 0), 0.75 * df_res["prix_predit"] + 0.25 * avm_s, df_res["prix_predit"]), 0
    )
    df_res["ecart_hybride_pct"] = np.round(
        (df_res["prix_reel"] - df_res["prix_hybride"]) / df_res["prix_hybride"].clip(lower=1.0) * 100.0, 1
    )

    # 3. Score de confiance (0-100)
    larg_rel = (df_res["prix_predit_q90"] - df_res["prix_predit_q10"]) / df_res["prix_predit"].clip(lower=1.0)
    sc_intervalle = np.clip(1.0 - larg_rel / 0.55, 0.0, 1.0) * 40.0
    comp_nb = pd.to_numeric(df_res.get("avm_comp_nb", 0), errors="coerce").fillna(0.0)
    comp_sim = pd.to_numeric(df_res.get("avm_comp_score_moyen", 0), errors="coerce").fillna(0.0)
    sc_comp = np.minimum(1.0, comp_nb / 6.0) * 20.0 + (comp_sim / 100.0) * 15.0
    has_dpe = df_res.get("dpe", pd.Series(dtype=str)).notna() & (df_res.get("dpe", "").astype(str).str.upper().isin(list("ABCDEFG")))
    has_annee = pd.to_numeric(df_res.get("annee_construction"), errors="coerce").notna()
    has_gps = pd.to_numeric(df_res.get("latitude"), errors="coerce").notna()
    not_lim = pd.to_numeric(df_res.get("commune_donnees_limitees", 0), errors="coerce").fillna(0) == 0
    sc_data = has_dpe.astype(float) * 8.0 + has_annee.astype(float) * 6.0 + has_gps.astype(float) * 6.0 + not_lim.astype(float) * 5.0
    df_res["score_confiance"] = np.round(np.clip(sc_intervalle + sc_comp + sc_data, 0.0, 100.0), 0).astype(int)

    # 4. Explication locale SHAP (TreeExplainer sur HistGradientBoosting)
    try:
        import json
        import shap
        pipe = modele.regressor_
        prepa = pipe.named_steps["prepa"]
        gbm = pipe.named_steps["gbm"]
        X_trans = prepa.transform(X_rec)
        feat_names = [n.split("__", 1)[-1] for n in prepa.get_feature_names_out()]
        explainer = shap.TreeExplainer(gbm)
        shap_vals = explainer.shap_values(X_trans)
        base_euro = float(np.exp(explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value))
        shap_json_list, shap_resumes = [], []
        for row_i in range(len(df_res)):
            p_pred = float(df_res["prix_predit"].iloc[row_i])
            sv = shap_vals[row_i]
            impacts = p_pred * (np.exp(sv) - 1.0)
            paires = [
                {
                    "feature": feat_names[j],
                    "label": fe.LABELS_FEATURES.get(feat_names[j], feat_names[j]),
                    "impact_euro": round(float(impacts[j]), 0),
                }
                for j in range(len(feat_names))
                if abs(impacts[j]) >= 250.0
            ]
            paires.sort(key=lambda x: abs(x["impact_euro"]), reverse=True)
            top5 = paires[:6]
            shap_json_list.append(json.dumps({"base_euro": round(base_euro, 0), "facteurs": top5}, ensure_ascii=False))
            shap_resumes.append(" | ".join(f"{x['label']}: {x['impact_euro']:+,.0f}€" for x in top5[:3]))
        df_res["shap_top_facteurs_json"] = shap_json_list
        df_res["shap_resume"] = shap_resumes
    except Exception as e:
        print(f"[Prédiction] Calcul SHAP ignoré : {e}")
        df_res["shap_top_facteurs_json"] = "{}"
        df_res["shap_resume"] = ""

    # Sauvegarde dans la table dédiée aux prédictions (`offres_predictions` / `csv/annonces_predictions.csv`)
    # tout en laissant `offres_recentes_features` propre (uniquement les features d'entrée) !
    df_res.to_csv(DB.CSV_PREDICTIONS, index=False)
    DB.ecrire_table_sql(df_res, DB.TABLE_PREDICTIONS)
    print(f"[Prédiction] {len(df_res)} offres récentes prédites -> table '{DB.TABLE_PREDICTIONS}' & '{DB.CSV_PREDICTIONS}'.")
    return df_res


def charger_donnees_dashboard() -> dict:
    """
    Charge directement les prédictions déjà calculées (`offres_predictions`)
    et les importances du modèle sauvegardé (`modele_poitiers.joblib`) sans ré-entraîner.
    """
    df_pred = DB.lire_table_ou_csv(DB.TABLE_PREDICTIONS, DB.CSV_PREDICTIONS)
    if df_pred.empty or "prix_predit" not in df_pred.columns or not os.path.exists(tr.CHEMIN_MODELE):
        df_pred = predire_sur_offres_recentes()

    artefact = joblib.load(tr.CHEMIN_MODELE)
    importances = artefact.get("importances", [])

    q_stats = (
        df_pred.groupby("quartier_clean")
        .agg(
            nb=("id", "count"),
            pm2_reel=("pm2_reel", "median"),
            pm2_predit=("pm2_predit", "median"),
            pm2_notaire=("pm2_notaire_dvf", "median"),
            nego_moy=("taux_nego_estime_pct", "mean"),
            mae_quartier=("erreur_abs_euro", "mean"),
            mdape_quartier=("erreur_abs_pct", "median"),
        )
        .reset_index()
    )
    q_stats = q_stats[q_stats["nb"] >= 3].sort_values("pm2_reel", ascending=False)
    quartiers = [
        {
            "quartier": r["quartier_clean"],
            "nb": int(r["nb"]),
            "pm2_reel": round(float(r["pm2_reel"])),
            "pm2_predit": round(float(r["pm2_predit"])),
            "pm2_notaire": round(float(r["pm2_notaire"])),
            "nego_moy_pct": round(float(r["nego_moy"]), 1),
            "mae": round(float(r["mae_quartier"])),
            "mdape_pct": round(float(r["mdape_quartier"]), 1),
        }
        for _, r in q_stats.iterrows()
    ]

    return {
        "df_pred": df_pred,
        "kpis": tr.calculer_metriques_groupe(df_pred),
        "quartiers": quartiers,
        "importances": importances,
    }


def predire_bien(modele: TransformedTargetRegressor, bien: dict) -> float:
    """Prédit le prix d'un bien individuel passé sous forme de dictionnaire."""
    df_bien = pd.DataFrame([bien])
    for col in [
        "id", "titre", "description", "quartier", "type_bien", "dpe", "ges",
        "surface_m2", "pieces", "nbr_chambres", "etage", "annee_construction",
        "latitude", "longitude", "pm2_voisinage",
    ]:
        if col not in df_bien.columns:
            df_bien[col] = np.nan
    df_bien["prix"] = 1.0
    if "quartier" in df_bien.columns:
        df_bien["quartier"] = df_bien["quartier"].replace(fe.QUARTIERS_HARMONISES)
    if "type_bien" in df_bien.columns:
        df_bien["type_bien"] = df_bien["type_bien"].replace(fe.TYPES_BIEN_HARMONISES)
    df_bien = fe.features_texte(df_bien)
    df_bien = fe.enrichir_avec_ollama(df_bien, utiliser_api_ollama=False)
    df_bien = fe.features_geo(df_bien)
    if df_bien["pm2_voisinage"].isna().all() and os.path.exists(fe.CHEMIN_DVF_CSV):
        dvf = pd.read_csv(fe.CHEMIN_DVF_CSV, low_memory=False)
        df_bien["pm2_voisinage"] = fe.prix_m2_voisinage(df_bien, dvf)["pm2_voisinage"]
    X_bien, _ = fe.construire_X_y(df_bien)
    return float(modele.predict(X_bien)[0])


if __name__ == "__main__":
    df_pred = predire_sur_offres_recentes()
    top = df_pred.sort_values("ecart_pct", ascending=True).head(5)
    print("\nTop 5 opportunités détectées sur les offres récentes :")
    for _, r in top.iterrows():
        print(
            f"  • [{r['id']}] {r['type_label']} {r['surface_m2']:.0f} m² ({r['quartier_clean']}) : "
            f"Affiché {r['prix_reel']:,.0f} € vs Prédit IA {r['prix_predit']:,.0f} € ({r['ecart_pct']:+.1f}%) | "
            f"Cible négo : {r['prix_cible_nego']:,.0f} € (-{r['taux_nego_estime_pct']:.1f}%)"
        )
