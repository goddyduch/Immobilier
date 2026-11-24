"""
Script 2/3 : Entraînement & Évaluation du Modèle IA — IA Immobilier Poitiers.

Responsabilité unique :
- Lit directement la table historique APRÈS Feature Engineering (`offres_historique_features`).
- Évalue le modèle en validation croisée 5 plis (R², MAE, MdAPE, paliers, comparatif Notaires DVF).
- Calcule l'importance des variables par permutation.
- Entraîne le modèle final sur 100 % de `offres_historique_features` et le sauvegarde dans `modele_poitiers.joblib`.

Usage :
    python Entrainement.py
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

import DB
import FeatureEngineering as fe

CHEMIN_MODELE = "modele_poitiers.joblib"


def construire_modele() -> TransformedTargetRegressor:
    """Construit le pipeline ML (TargetEncoder + HistGradientBoostingRegressor sur log-prix)."""
    cv_interne = KFold(n_splits=5, shuffle=True, random_state=0)
    prepa = ColumnTransformer(
        [("cat", TargetEncoder(target_type="continuous", cv=cv_interne), fe.CATEGORIELLES)],
        remainder="passthrough",
    )
    modele = Pipeline([
        ("prepa", prepa),
        ("gbm", HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=8, random_state=0)),
    ])
    return TransformedTargetRegressor(regressor=modele, func=np.log, inverse_func=np.exp)


def construire_modele_quantile(quantile: float) -> TransformedTargetRegressor:
    """Construit un modèle quantile (Q10 ou Q90) pour les intervalles de prédiction."""
    cv_interne = KFold(n_splits=5, shuffle=True, random_state=0)
    prepa = ColumnTransformer(
        [("cat", TargetEncoder(target_type="continuous", cv=cv_interne), fe.CATEGORIELLES)],
        remainder="passthrough",
    )
    modele = Pipeline([
        ("prepa", prepa),
        ("gbm", HistGradientBoostingRegressor(
            loss="quantile", quantile=quantile, max_depth=3,
            learning_rate=0.05, max_iter=250, min_samples_leaf=10, random_state=0)),
    ])
    return TransformedTargetRegressor(regressor=modele, func=np.log, inverse_func=np.exp)


def estimer_negociation(df_res: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule pour chaque bien :
    - `prix_notaire_dvf` : valeur de marché basée sur les actes notariés DVF (taille proche / tranche surface)
    - `ecart_vs_notaire_pct` : surcote/décote du prix affiché par rapport aux ventes notariées réelles
    - `taux_nego_estime_pct` : marge de négociation réaliste (entre 1.5% et 18%)
    - `score_pouvoir_nego` : indice synthétique de pouvoir de négociation (/10)
    - `prix_cible_nego` : prix d'offre recommandé après négociation
    - `prix_total_projet`, `valeur_estimee_renovee`, `gain_net_apres_travaux`, `decote_nette_pct`
    """
    df_res = df_res.copy()

    # Priorité aux ventes DVF de taille proche, puis tranche de surface, puis voisinage global
    pm2_ref = None
    for col_dvf in ("pm2_dvf_taille_proche", "pm2_dvf_tranche_surface", "pm2_voisinage"):
        if col_dvf in df_res.columns and df_res[col_dvf].notna().any():
            pm2_ref = df_res[col_dvf] if pm2_ref is None else pm2_ref.fillna(df_res[col_dvf])

    if pm2_ref is not None:
        df_res["pm2_notaire_dvf"] = pm2_ref.fillna(df_res["pm2_predit"])
    else:
        df_res["pm2_notaire_dvf"] = np.round(df_res["pm2_predit"] * 0.94, 0)

    df_res["prix_notaire_dvf"] = np.round(df_res["pm2_notaire_dvf"] * df_res["surface_m2"], 0)
    df_res["ecart_vs_notaire_pct"] = np.round(
        (df_res["prix_reel"] - df_res["prix_notaire_dvf"]) / df_res["prix_notaire_dvf"] * 100, 1
    )

    base_nego = np.where(df_res["type_label"] == "Maison", 5.5, 3.5)
    surcote_ia = np.clip(df_res["ecart_pct"], -5.0, 25.0) * 0.25
    surcote_dvf = np.clip(df_res["ecart_vs_notaire_pct"], -5.0, 25.0) * 0.15

    dpe_s = df_res["dpe"].fillna("").astype(str).str.upper()
    bonus_dpe = np.where(dpe_s.isin(["F", "G"]), 3.5, np.where(dpe_s == "E", 1.5, 0.0))

    trav_s = pd.to_numeric(df_res.get("llm_travaux_score", 0), errors="coerce").fillna(0)
    bonus_travaux = trav_s * 1.3

    # Pouvoir de négociation supplémentaire lié à la durée en ligne et aux baisses de prix
    duree_j = pd.to_numeric(df_res.get("duree_en_ligne_jours", 0), errors="coerce").fillna(0)
    bonus_duree = np.where(duree_j >= 90, 2.0, np.where(duree_j >= 45, 1.0, 0.0))
    a_baisse = pd.to_numeric(df_res.get("a_baisse_prix", 0), errors="coerce").fillna(0)

    taux_nego = np.clip(base_nego + surcote_ia + surcote_dvf + bonus_dpe + bonus_travaux + bonus_duree, 1.5, 18.0)
    df_res["taux_nego_estime_pct"] = np.round(taux_nego, 1)
    df_res["prix_cible_nego"] = np.round(df_res["prix_reel"] * (1.0 - df_res["taux_nego_estime_pct"] / 100.0), 0)
    df_res["economie_nego_euro"] = np.round(df_res["prix_reel"] - df_res["prix_cible_nego"], 0)

    # Score de pouvoir de négociation (/10)
    score_nego = (
        2.0
        + np.clip(df_res["ecart_vs_notaire_pct"], 0, 25) * 0.16
        + np.where(dpe_s.isin(["F", "G"]), 1.8, np.where(dpe_s == "E", 0.8, 0.0))
        + np.clip(trav_s * 0.7, 0, 2.1)
        + np.where(duree_j >= 60, 1.2, np.where(duree_j >= 30, 0.6, 0.0))
        + np.where(a_baisse > 0, 1.0, 0.0)
    )
    df_res["score_pouvoir_nego"] = np.round(np.clip(score_nego, 1.0, 10.0), 1)

    # Calcul du Coût Total Projet & Valeur Rénovée (si non déjà simulé par le modèle)
    cout_trav = pd.to_numeric(df_res.get("cout_travaux_estime_euro", 0), errors="coerce").fillna(0.0)
    df_res["prix_total_projet"] = np.round(df_res["prix_reel"] + cout_trav, 0)
    if "valeur_estimee_renovee" not in df_res.columns:
        df_res["valeur_estimee_renovee"] = np.round(df_res["prix_predit"] + cout_trav * 0.85, 0)
    df_res["gain_net_apres_travaux"] = np.round(df_res["valeur_estimee_renovee"] - df_res["prix_total_projet"], 0)
    df_res["decote_nette_pct"] = np.round(
        (df_res["prix_total_projet"] - df_res["valeur_estimee_renovee"]) / np.maximum(df_res["valeur_estimee_renovee"], 1.0) * 100.0,
        1,
    )
    return df_res


def generer_predictions_cv(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Calcule les prédictions hors-échantillon (CV 5 plis), les écarts DVF et la marge de négociation."""
    X, y = fe.construire_X_y(df)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    y_pred = cross_val_predict(construire_modele(), X, y, cv=cv)

    df_res = df.loc[X.index].copy()
    df_res["prix_reel"] = np.round(y.values, 0)
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
    df_res = estimer_negociation(df_res)
    return df_res


def calculer_importances(df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    """Calcule l'importance des variables (en € d'impact sur la MAE) par permutation."""
    X, y = fe.construire_X_y(df)
    modele = construire_modele()
    modele.fit(X, y)
    perm = permutation_importance(
        modele, X, y, n_repeats=10, random_state=0, scoring="neg_mean_absolute_error"
    )
    imp = pd.DataFrame({
        "feature": X.columns,
        "label": [fe.LABELS_FEATURES.get(c, c) for c in X.columns],
        "impact_euro": np.round(perm.importances_mean, 0),
        "std_euro": np.round(perm.importances_std, 0),
    }).sort_values("impact_euro", ascending=False)
    return imp.head(top_n).reset_index(drop=True)


def calculer_metriques_groupe(df_sub: pd.DataFrame) -> dict:
    """Calcule l'ensemble des métriques de précision, d'écart DVF notaire et de marge de négociation."""
    if len(df_sub) == 0:
        return {}
    y_reel = df_sub["prix_reel"].to_numpy()
    y_pred = df_sub["prix_predit"].to_numpy()
    err_abs = df_sub["erreur_abs_euro"].to_numpy()
    err_pct = df_sub["erreur_abs_pct"].to_numpy()
    surf = df_sub["surface_m2"].to_numpy()

    mae = float(np.mean(err_abs))
    med_ae = float(np.median(err_abs))
    rmse = float(np.sqrt(np.mean((y_reel - y_pred) ** 2)))
    mape = float(np.mean(err_pct))
    mdape = float(np.median(err_pct))

    ss_res = float(np.sum((y_reel - y_pred) ** 2))
    ss_tot = float(np.sum((y_reel - np.mean(y_reel)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    pm2_med = float(np.median(y_reel / surf))
    mae_ref = float(np.mean(np.abs(y_reel - pm2_med * surf)))
    gain_vs_ref_pct = float((mae_ref - mae) / mae_ref * 100) if mae_ref > 0 else 0.0

    p_5 = float(np.mean(err_pct <= 5.0) * 100)
    p_10 = float(np.mean(err_pct <= 10.0) * 100)
    p_15 = float(np.mean(err_pct <= 15.0) * 100)
    p_20 = float(np.mean(err_pct <= 20.0) * 100)
    p_25_plus = float(np.mean(err_pct > 25.0) * 100)

    pm2_dvf_med = float(df_sub["pm2_notaire_dvf"].median()) if "pm2_notaire_dvf" in df_sub.columns else pm2_med
    ecart_notaire_med = (
        float(df_sub["ecart_vs_notaire_pct"].median())
        if "ecart_vs_notaire_pct" in df_sub.columns
        else 0.0
    )
    nego_moy_pct = (
        float(df_sub["taux_nego_estime_pct"].mean())
        if "taux_nego_estime_pct" in df_sub.columns
        else 5.0
    )
    economie_med = (
        float(df_sub["economie_nego_euro"].median())
        if "economie_nego_euro" in df_sub.columns
        else 0.0
    )

    return {
        "nb_biens": int(len(df_sub)),
        "prix_median": round(float(np.median(y_reel))),
        "pm2_median": round(pm2_med),
        "pm2_notaire_median": round(pm2_dvf_med),
        "ecart_notaire_median_pct": round(ecart_notaire_med, 1),
        "nego_moyenne_pct": round(nego_moy_pct, 1),
        "economie_nego_mediane": round(economie_med),
        "r2": round(r2, 2),
        "mae": round(mae),
        "med_ae": round(med_ae),
        "rmse": round(rmse),
        "mae_ref": round(mae_ref),
        "gain_vs_ref_pct": round(gain_vs_ref_pct, 1),
        "mape": round(mape, 1),
        "erreur_mediane_pct": round(mdape, 1),
        "within_5pct": round(p_5, 1),
        "within_10pct": round(p_10, 1),
        "within_15pct": round(p_15, 1),
        "within_20pct": round(p_20, 1),
        "outliers_25pct": round(p_25_plus, 1),
    }


def evaluer_complet(df: pd.DataFrame, afficher: bool = True) -> dict:
    """
    Évaluation complète du modèle + Analyse Prix Affichés vs Ventes Notariées (DVF)
    + Estimation des marges de négociation.
    """
    df_pred = generer_predictions_cv(df)
    global_metrics = calculer_metriques_groupe(df_pred)

    par_type = []
    for t_label, grp in df_pred.groupby("type_label"):
        m = calculer_metriques_groupe(grp)
        m["type_label"] = t_label
        par_type.append(m)

    bins = [0, 100_000, 180_000, 280_000, float("inf")]
    labels_tranches = ["< 100 k€", "100 k€ – 180 k€", "180 k€ – 280 k€", "> 280 k€"]
    df_pred["tranche_prix"] = pd.cut(df_pred["prix_reel"], bins=bins, labels=labels_tranches)
    par_tranche = []
    for tr_label in labels_tranches:
        grp = df_pred[df_pred["tranche_prix"] == tr_label]
        if len(grp) > 0:
            m = calculer_metriques_groupe(grp)
            m["tranche"] = tr_label
            par_tranche.append(m)

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

    importances_df = calculer_importances(df, top_n=10)
    importances = importances_df.to_dict(orient="records")

    if afficher:
        g = global_metrics
        print("\n" + "=" * 82)
        print("  RAPPORT D'ÉVALUATION IA & ANALYSE NOTAIRES (DVF) — POITIERS (86000)")
        print("=" * 82)
        print(f"Biens évalués : {g['nb_biens']} | Prix affiché médian : {g['prix_median']:,.0f} € ({g['pm2_median']:,.0f} €/m²)")
        print(f"Prix vendu Notaire (DVF voisinage) médian : {g['pm2_notaire_median']:,.0f} €/m² (Écart affiché vs signé : {g['ecart_notaire_median_pct']:+.1f}%)")
        print(f"Marge de négociation moyenne recommandée  : -{g['nego_moyenne_pct']:.1f}% (Économie médiane : -{g['economie_nego_mediane']:,.0f} €)")
        print("-" * 82)
        print("1. PERFORMANCE DU MODÈLE IA (Validation Croisée 5 plis) :")
        print(f"   • R² (variance expliquée)        : {g['r2']:.2f}")
        print(f"   • Erreur médiane (MdAPE)         : ±{g['erreur_mediane_pct']:.1f} % (±{g['med_ae']:,.0f} € sur 50% des biens)")
        print(f"   • Erreur moyenne (MAE / MAPE)    : ±{g['mae']:,.0f} € (±{g['mape']:.1f} %)")
        print(f"   • Gain vs estimation naïve (€/m²): +{g['gain_vs_ref_pct']:.1f} % de précision")
        print("-" * 82)
        print("2. PALIERS DE PRÉCISION CONCRETS :")
        print(f"   • <=  5 % d'écart : {g['within_5pct']:5.1f} % | <= 10 % : {g['within_10pct']:5.1f} % | <= 15 % : {g['within_15pct']:5.1f} % | <= 20 % : {g['within_20pct']:5.1f} %")
        print("-" * 82)
        print("3. COMPARATIF PRIX AFFICHÉ vs NOTAIRES (DVF) & NÉGOCIATION PAR TYPE :")
        for pt in par_type:
            print(
                f"   • {pt['type_label']:<12} ({pt['nb_biens']:3d}) : "
                f"Affiché {pt['pm2_median']:>4,.0f} €/m² vs Notaire DVF {pt['pm2_notaire_median']:>4,.0f} €/m² "
                f"({pt['ecart_notaire_median_pct']:+4.1f}%) | Négo conseillée : -{pt['nego_moyenne_pct']:.1f}%"
            )
        print("-" * 82)
        print("4. TOP VARIABLES QUI FONT LE PRIX :")
        for row in importances[:6]:
            print(f"   • {row['label']:<36} : impact moyen ±{row['impact_euro']:,.0f} €")
        try:
            import AVMComparable
            df_pred_avm = DB.lire_table_ou_csv(DB.TABLE_PREDICTIONS, DB.CSV_PREDICTIONS)
            df_cmp = AVMComparable.comparer_ml_vs_avm(df_pred_avm)
            if not df_cmp.empty:
                print("-" * 82)
                print("5. COMPARATIF PRÉCISION : ML vs AVM COMPARABLES vs HYBRIDE (offres récentes) :")
                for _, cr in df_cmp.iterrows():
                    print(
                        f"   • {cr['Approche']:<38} | R²={cr['R²']:.3f} | "
                        f"MAE=±{cr['MAE (€)']:,.0f} € | MdAPE=±{cr['MdAPE (%)']:.1f}% | <=15%: {cr['<= 15%']:.1f}%"
                    )
        except Exception:
            pass
        print("=" * 82)

    return {
        "df_pred": df_pred,
        "kpis": global_metrics,
        "par_type": par_type,
        "par_tranche": par_tranche,
        "quartiers": quartiers,
        "importances": importances,
    }


def evaluer(df: pd.DataFrame):
    return evaluer_complet(df, afficher=True)


def entrainer_modele(df_features: pd.DataFrame) -> TransformedTargetRegressor:
    """Entraîne le modèle directement sur une table après Feature Engineering."""
    X, y = fe.construire_X_y(df_features)
    modele = construire_modele()
    modele.fit(X, y)
    return modele


def entrainer_et_sauvegarder_modele(
    df_hist_feat: pd.DataFrame | None = None,
    model_path: str = CHEMIN_MODELE,
    afficher: bool = False,
) -> dict:
    """
    ÉTAPE 2 — ENTRAÎNEMENT (directement sur la table APRÈS Feature Engineering) :
    - Lit directement la table `offres_historique_features` (`csv/annonces_historique_features.csv`)
    - Évalue le modèle en validation croisée 5 plis
    - Entraîne le modèle final sur 100 % de `offres_historique_features` et le sauvegarde dans `modele_poitiers.joblib`
    - Met à jour `offres_historique_features` avec les colonnes d'évaluation
    """
    if df_hist_feat is None:
        df_hist_feat = DB.lire_table_ou_csv(DB.TABLE_HIST_FEAT, DB.CSV_HIST_FEAT)
        if df_hist_feat.empty or "dist_centre_km" not in df_hist_feat.columns:
            df_hist_feat, _ = fe.construire_tables_features(verbose=False)

    # On s'assure que la table d'entraînement ne contient aucune ancienne colonne de prédiction
    df_hist_feat = fe.supprimer_colonnes_prediction(df_hist_feat)

    print(
        f"[Entraînement] Lecture directe de la table après Feature Engineering "
        f"'{DB.TABLE_HIST_FEAT}' ({len(df_hist_feat)} lignes, {len(df_hist_feat.columns)} colonnes de features)..."
    )
    eval_res = evaluer_complet(df_hist_feat, afficher=afficher)
    df_hist_eval = eval_res["df_pred"]

    X_hist, y_hist = fe.construire_X_y(df_hist_feat)
    modele_final = construire_modele()
    modele_final.fit(X_hist, y_hist)

    modele_q10 = construire_modele_quantile(0.10)
    modele_q10.fit(X_hist, y_hist)
    modele_q90 = construire_modele_quantile(0.90)
    modele_q90.fit(X_hist, y_hist)

    oof_par_id = dict(zip(df_hist_eval["id"].astype(str), df_hist_eval["prix_predit"]))

    artefact = {
        "modele": modele_final,
        "modele_q10": modele_q10,
        "modele_q90": modele_q90,
        "colonnes_X": list(X_hist.columns),
        "oof_par_id": oof_par_id,
        "importances": eval_res["importances"],
        "kpis": eval_res["kpis"],
    }
    joblib.dump(artefact, model_path)

    # On garde `offres_historique_features` propre (sans colonnes de prédiction)
    print(
        f"[Entraînement] Modèle entraîné sur '{DB.TABLE_HIST_FEAT}' ({len(df_hist_eval)} offres, "
        f"{len(X_hist.columns)} variables explicatives X -> cible y='prix') -> sauvegardé dans '{model_path}'."
    )
    return artefact


if __name__ == "__main__":
    entrainer_et_sauvegarder_modele(afficher=True)

