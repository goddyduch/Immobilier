"""
Étude Poussée des Features & Tracking MLflow (Modèle verrouillé sur HistGradientBoostingRegressor).

Ce script réalise :
1. `importer_table_historique()` : Charge `offres_historique_features` (sans aucune fuite ni colonne de prédiction).
2. `analyser_ablation_histgb()`  : Teste 8 combinaisons progressives de familles de features en double validation :
   • `KFold(5)` (précision globale)
   • `GroupKFold` par quartier/secteur (généralisation géographique)
3. `audit_features_detaille()`   : Calcule pour chaque variable :
   • Permutation Importance (€ d'impact sur la MAE)
   • Drop-Column Importance (gain ou perte réelle de MAE et R² quand on supprime la variable)
   • Corrélation avec le prix
4. Sauvegarde `csv/audit_ablation_histgb.csv`, `csv/audit_features_histgb.csv`, loggue tous les runs dans MLflow
   (`Immobilier_Poitiers_HistGB_Features`) et génère `csv/rapport_improvement_mlflow.html`.

Usage :
    python Improvement.py
    python main.py improve
    mlflow ui --port 5000
"""

import os
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict

from src.ingestion import db as DB
from src.ml import train as tr
from src.ml import features as fe

warnings.filterwarnings("ignore", category=UserWarning)

MLFLOW_EXPERIMENT_NAME = "Immobilier_Poitiers_HistGB_Features"
RAPPORT_HTML_PATH = os.path.join("csv", "rapport_improvement_mlflow.html")

try:
    import mlflow
    import mlflow.sklearn

    MLFLOW_DISPO = True
except ImportError:
    MLFLOW_DISPO = False


def importer_table_historique() -> pd.DataFrame:
    """Importe `offres_historique_features` sans aucune colonne de prédiction."""
    df_hist = DB.lire_table_ou_csv(DB.TABLE_HIST_FEAT, DB.CSV_HIST_FEAT)
    if df_hist.empty or "iris_code" not in df_hist.columns or "vis_saturation" not in df_hist.columns:
        df_hist, _ = fe.construire_tables_features(verbose=False)
    df_hist = fe.supprimer_colonnes_prediction(df_hist)
    df_hist = df_hist[df_hist["prix"].notna() & (df_hist["prix"] > 0)].copy()
    df_hist["dpe_num"] = df_hist["dpe"].map(fe.ORDRE_DPE)
    df_hist["ges_num"] = df_hist["ges"].map(fe.ORDRE_DPE)
    for c in fe.CATEGORIELLES:
        if c not in df_hist.columns:
            df_hist[c] = "inconnu"
        else:
            df_hist[c] = df_hist[c].fillna("inconnu").astype(str)
    print(f"[Import Historique] '{DB.TABLE_HIST_FEAT}' : {len(df_hist)} offres × {len(df_hist.columns)} colonnes.")
    return df_hist


def _calculer_chi2_adequation(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Test du chi-deux d'adéquation entre distribution observée et prédite par tranches de prix."""
    if len(y_true) < 15:
        return 0.0, 1.0
    try:
        from scipy.stats import chisquare
        n_bins = min(8, max(3, len(y_true) // 15))
        bins = np.unique(np.quantile(y_true, np.linspace(0.0, 1.0, n_bins + 1)))
        if len(bins) < 3:
            return 0.0, 1.0
        bins[0] = -np.inf
        bins[-1] = np.inf
        obs, _ = np.histogram(y_true, bins=bins)
        exp, _ = np.histogram(y_pred, bins=bins)
        exp = np.where(exp == 0, 1e-6, exp)
        exp = exp * (obs.sum() / exp.sum())
        stat, pval = chisquare(f_obs=obs, f_exp=exp)
        return round(float(stat), 2), round(float(pval), 4)
    except Exception:
        return 0.0, 1.0


def calculer_metriques_cv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    surface: np.ndarray,
    prefix: str = "",
    q10: np.ndarray | None = None,
    q90: np.ndarray | None = None,
) -> dict:
    if len(y_true) == 0:
        return {}
    err_abs = np.abs(y_true - y_pred)
    err_pct = err_abs / np.maximum(y_true, 1.0) * 100.0
    mae = float(np.mean(err_abs))
    med_ae = float(np.median(err_abs))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(err_pct))
    mdape = float(np.median(err_pct))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    pm2_med = float(np.median(y_true / np.maximum(surface, 1.0)))
    mae_naif = float(np.mean(np.abs(y_true - pm2_med * surface)))
    gain_naif = float((mae_naif - mae) / mae_naif * 100.0) if mae_naif > 0 else 0.0
    chi2_stat, chi2_pval = _calculer_chi2_adequation(y_true, y_pred)

    couv_pct = None
    if q10 is not None and q90 is not None and len(q10) == len(y_true):
        couv_pct = round(float(np.mean((y_true >= q10) & (y_true <= q90)) * 100.0), 1)

    p = f"{prefix}_" if prefix else ""
    out = {
        f"{p}nb_biens": int(len(y_true)),
        f"{p}r2": round(r2, 4),
        f"{p}chi2_stat": chi2_stat,
        f"{p}chi2_pvalue": chi2_pval,
        f"{p}mae": round(mae, 0),
        f"{p}med_ae": round(med_ae, 0),
        f"{p}rmse": round(rmse, 0),
        f"{p}mape": round(mape, 2),
        f"{p}mdape": round(mdape, 2),
        f"{p}within_10pct": round(float(np.mean(err_pct <= 10.0) * 100), 1),
        f"{p}within_15pct": round(float(np.mean(err_pct <= 15.0) * 100), 1),
        f"{p}within_20pct": round(float(np.mean(err_pct <= 20.0) * 100), 1),
        f"{p}gain_vs_naif_pct": round(gain_naif, 1),
    }
    if couv_pct is not None:
        out[f"{p}couverture_q10_q90_pct"] = couv_pct
    return out


def _construire_histgb_pour_cols(cat_cols: list[str]):
    from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import TargetEncoder

    cv_interne = KFold(n_splits=5, shuffle=True, random_state=0)
    transformers = [("cat", TargetEncoder(target_type="continuous", cv=cv_interne), cat_cols)] if cat_cols else []
    prepa = ColumnTransformer(transformers, remainder="passthrough")
    modele = Pipeline([
        ("prepa", prepa),
        ("gbm", HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=8, random_state=0
        )),
    ])
    return TransformedTargetRegressor(regressor=modele, func=np.log, inverse_func=np.exp)


def evaluer_jeu_features_histgb(df: pd.DataFrame, cols: list[str], y: np.ndarray, surface: np.ndarray, groupes: np.ndarray) -> tuple[dict, np.ndarray]:
    """Évalue un jeu de features avec HistGradientBoosting en KFold(5) + GroupKFold."""
    cols_dispo = [c for c in dict.fromkeys(cols) if c in df.columns]
    cat_cols = [c for c in cols_dispo if c in fe.CATEGORIELLES]
    num_cols = [c for c in cols_dispo if c not in fe.CATEGORIELLES]

    X = df[num_cols + cat_cols].copy()
    X[num_cols] = X[num_cols].astype(float)

    cv_kf = KFold(n_splits=5, shuffle=True, random_state=0)
    y_pred_kf = cross_val_predict(_construire_histgb_pour_cols(cat_cols), X, y, cv=cv_kf)
    met = calculer_metriques_cv(y, y_pred_kf, surface)

    # Validation géographique (GroupKFold par quartier/secteur)
    n_groups = len(np.unique(groupes))
    if n_groups >= 5:
        cv_grp = GroupKFold(n_splits=min(5, n_groups))
        y_pred_grp = cross_val_predict(_construire_histgb_pour_cols(cat_cols), X, y, cv=cv_grp, groups=groupes)
        met_grp = calculer_metriques_cv(y, y_pred_grp, surface, prefix="grp")
        met["grp_r2"] = met_grp["grp_r2"]
        met["grp_mae"] = met_grp["grp_mae"]
        met["grp_mdape"] = met_grp["grp_mdape"]
    return met, y_pred_kf


def logger_run_mlflow(nom_run: str, tags: dict, params: dict, metriques: dict, modele=None):
    if not MLFLOW_DISPO:
        return
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=nom_run):
        mlflow.set_tags(tags)
        mlflow.log_params(params)
        mlflow.log_metrics(metriques)
        if modele is not None:
            try:
                mlflow.sklearn.log_model(modele, name="modele", serialization_format="cloudpickle")
            except TypeError:
                mlflow.sklearn.log_model(modele, artifact_path="modele", serialization_format="cloudpickle")


def executer_etude_features(df_hist: pd.DataFrame) -> dict:
    """
    1. Ablation Study progressive (8 étapes) sur HistGradientBoostingRegressor.
    2. Audit individuel (Permutation Importance + Drop-Column Importance) de chaque feature.
    """
    print("\n" + "=" * 88)
    print("  1. ÉTUDE D'ABLATION DES FEATURES (MODÈLE : HistGradientBoostingRegressor)")
    print("=" * 88)

    y = df_hist["prix"].astype(float).to_numpy()
    surface = df_hist["surface_m2"].astype(float).to_numpy()
    groupes = df_hist["quartier"].fillna(df_hist.get("code_postal", "inconnu")).astype(str).to_numpy()

    f_base = ["surface_m2", "pieces", "etage", "annee_construction", "dpe_num", "quartier", "type_bien"]
    f_surf = ["surface_par_piece", "surface_relative_marche", "age_bien"]
    f_texte_llm = [
        "txt_longueur", "surface_ext_m2", "charges_annuelles",
        "llm_etat_score", "llm_travaux_score", "llm_note_offre",
        "txt_piscine", "txt_a_renover", "txt_jardin", "txt_terrasse",
        "txt_parking", "txt_cave", "txt_lumineux", "txt_pac",
        "txt_immeuble", "txt_standing",
    ]
    f_geo_od = [
        "latitude", "dist_centre_km", "dist_gare_km",
        "dist_campus_km", "dist_futuroscope_km",
        "est_en_qpv", "iris_code",
    ]
    f_dvf = ["pm2_voisinage", "pm2_dvf_pondere_dist", "pm2_dvf_tranche_surface", "valeur_dvf_ajustee_taille"]
    f_travaux = [
        "trav_rafraichissement", "trav_cuisine_sdb", "trav_menuiseries",
        "trav_elec_plomberie", "valeur_dvf_nette_travaux",
    ]
    f_photos = ["vis_luminosite", "vis_contraste", "vis_nettete", "vis_saturation", "vis_etat_vlm"]

    etapes = [
        ("1_Structure_Base", f_base),
        ("2_+_Degressivite_Surface", f_base + f_surf),
        ("3_+_Texte_et_LLM", f_base + f_surf + f_texte_llm),
        ("4_+_Geo_et_OpenData_(QPV_HLM_IRIS)", f_base + f_surf + f_texte_llm + f_geo_od),
        ("5_+_DVF_Notaires_Avance", f_base + f_surf + f_texte_llm + f_geo_od + f_dvf),
        ("6_+_Cout_Travaux_Estime", f_base + f_surf + f_texte_llm + f_geo_od + f_dvf + f_travaux),
        ("7_+_Photos_Visuel", f_base + f_surf + f_texte_llm + f_geo_od + f_dvf + f_travaux + f_photos),
    ]

    lignes_abl = []
    preds_etapes = {}
    for nom, cols in etapes:
        cols_dispo = [c for c in dict.fromkeys(cols) if c in df_hist.columns and df_hist[c].notna().any()]
        met, y_pred = evaluer_jeu_features_histgb(df_hist, cols_dispo, y, surface, groupes)
        preds_etapes[nom] = y_pred
        logger_run_mlflow(
            nom_run=nom,
            tags={"phase": "ablation_histgb", "modele": "HistGradientBoosting"},
            params={"nb_features": len(cols_dispo), "etape": nom},
            metriques=met,
        )
        lignes_abl.append({"etape": nom, "nb_features": len(cols_dispo), **met})
        print(
            f"  • {nom:<36} ({len(cols_dispo):2d} vars) -> "
            f"R²={met['r2']:.3f} | MAE={met['mae']:>6,.0f} € | MdAPE=±{met['mdape']:.1f}% | "
            f"<=15%: {met['within_15pct']:4.1f}% | GroupKFold MAE={met.get('grp_mae', 0):>6,.0f} €"
        )

    # Audit individuel (Permutation + Drop-Column Importance) sur l'ensemble complet
    print("\n" + "=" * 88)
    print("  2. AUDIT INDIVIDUEL : PERMUTATION & DROP-COLUMN IMPORTANCE (BRUIT vs UTILE)")
    print("=" * 88)
    X_full, y_full = fe.construire_X_y(df_hist)
    cols_full = list(X_full.columns)
    met_full, _ = evaluer_jeu_features_histgb(df_hist, cols_full, y, surface, groupes)

    modele_full = tr.construire_modele()
    modele_full.fit(X_full, y_full)
    perm = permutation_importance(
        modele_full, X_full, y_full, n_repeats=8, random_state=0, scoring="neg_mean_absolute_error"
    )

    audit_rows = []
    for idx_c, col in enumerate(cols_full):
        cols_sans = [c for c in cols_full if c != col]
        met_sans, _ = evaluer_jeu_features_histgb(df_hist, cols_sans, y, surface, groupes)
        # Positif = la feature réduit la MAE (utile) ; Négatif = la supprimer améliore la MAE (bruit)
        gain_mae_drop = round(float(met_sans["mae"] - met_full["mae"]), 0)
        gain_r2_drop = round(float(met_full["r2"] - met_sans["r2"]), 4)
        imp_perm = round(float(perm.importances_mean[idx_c]), 0)
        statut = "Utile" if (gain_mae_drop > 0 or imp_perm >= 800) else ("Neutre" if gain_mae_drop == 0 else "Bruit (à retirer)")
        audit_rows.append({
            "feature": col,
            "label": fe.LABELS_FEATURES.get(col, col),
            "impact_perm_euro": imp_perm,
            "gain_drop_mae_euro": gain_mae_drop,
            "gain_drop_r2": gain_r2_drop,
            "statut": statut,
        })

    df_audit = pd.DataFrame(audit_rows).sort_values(["gain_drop_mae_euro", "impact_perm_euro"], ascending=False).reset_index(drop=True)

    # Étape 8 : Modèle épuré sans les variables identifiées comme bruit pur (gain_drop_mae < -100 et perm < 200)
    bruit_cols = set(df_audit[(df_audit["gain_drop_mae_euro"] < -100) & (df_audit["impact_perm_euro"] < 200)]["feature"].tolist())
    cols_epurees = [c for c in cols_full if c not in bruit_cols or c in fe.CATEGORIELLES]
    met_epure, y_pred_epure = evaluer_jeu_features_histgb(df_hist, cols_epurees, y, surface, groupes)
    preds_etapes["8_Selection_Sans_Bruit"] = y_pred_epure
    logger_run_mlflow(
        nom_run="8_Selection_Sans_Bruit",
        tags={"phase": "ablation_histgb", "modele": "HistGradientBoosting"},
        params={"nb_features": len(cols_epurees), "etape": "8_Selection_Sans_Bruit", "retirees": len(bruit_cols)},
        metriques=met_epure,
        modele=modele_full,
    )
    lignes_abl.append({"etape": "8_Selection_Sans_Bruit", "nb_features": len(cols_epurees), **met_epure})
    print(
        f"  • {'8_Selection_Sans_Bruit':<36} ({len(cols_epurees):2d} vars) -> "
        f"R²={met_epure['r2']:.3f} | MAE={met_epure['mae']:>6,.0f} € | MdAPE=±{met_epure['mdape']:.1f}% | "
        f"<=15%: {met_epure['within_15pct']:4.1f}% | GroupKFold MAE={met_epure.get('grp_mae', 0):>6,.0f} €"
    )

    print("\n[Top 12 Features les plus utiles (Gain réel MAE en Drop-Column & Permutation)]")
    for _, r in df_audit.head(12).iterrows():
        print(f"  • {r['label']:<42} : Drop-Col {r['gain_drop_mae_euro']:+6,.0f} € | Perm ±{r['impact_perm_euro']:>6,.0f} € | {r['statut']}")

    df_bruit = df_audit[df_audit["statut"].str.startswith("Bruit")]
    if not df_bruit.empty:
        print(f"\n[Features identifiées comme bruit ({len(df_bruit)})] : " + ", ".join(df_bruit["label"].head(8).tolist()))

    df_abl = pd.DataFrame(lignes_abl)
    df_abl.to_csv(DB.CSV_AUDIT_ABLATION, index=False)
    df_audit.to_csv(DB.CSV_AUDIT_FEATURES, index=False)
    return {
        "ablation": df_abl,
        "audit": df_audit,
        "preds_etapes": preds_etapes,
        "y_reel": y,
        "surface": surface,
    }


def generer_rapport_html(res: dict, html_path: str = RAPPORT_HTML_PATH) -> str:
    df_abl = res["ablation"]
    df_aud = res["audit"]
    top_pos = df_aud.head(15).iloc[::-1]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "1. Ablation Study : Évolution de la MAE (€) par Famille de Features",
            "2. Top 15 Features par Gain Réel MAE (Drop-Column €)",
            "3. Effet Dégressif Surface : Prix/m² Réel vs Prédit HistGradientBoosting",
            "4. Généralisation : KFold MAE vs GroupKFold (par Quartier) MAE",
        ),
        horizontal_spacing=0.14,
        vertical_spacing=0.14,
    )

    fig.add_trace(
        go.Bar(
            x=df_abl["etape"],
            y=df_abl["mae"],
            name="MAE KFold (€)",
            marker_color="#3b82f6",
            text=[f"{int(m):,} €<br>R²={r:.3f}".replace(",", " ") for m, r in zip(df_abl["mae"], df_abl["r2"])],
            textposition="auto",
        ),
        row=1, col=1,
    )

    couleurs_drop = ["#10b981" if v >= 0 else "#ef4444" for v in top_pos["gain_drop_mae_euro"]]
    fig.add_trace(
        go.Bar(
            y=top_pos["label"],
            x=top_pos["gain_drop_mae_euro"],
            name="Gain MAE Drop-Column (€)",
            orientation="h",
            marker_color=couleurs_drop,
            text=[f"{int(v):+,} €".replace(",", " ") for v in top_pos["gain_drop_mae_euro"]],
            textposition="auto",
        ),
        row=1, col=2,
    )

    # Courbe dégressivité surface
    y_reel = res["y_reel"]
    surf = res["surface"]
    y_best = res["preds_etapes"].get("8_Selection_Sans_Bruit", res["preds_etapes"]["7_+_Photos_Visuel"])
    df_s = pd.DataFrame({"surf": surf, "pm2_reel": y_reel / surf, "pm2_pred": y_best / surf})
    df_s["tranche"] = pd.cut(df_s["surf"], bins=[0, 35, 55, 80, 110, 500], labels=["<35 m²", "35-55 m²", "55-80 m²", "80-110 m²", ">110 m²"])
    g_s = df_s.groupby("tranche", observed=False).median().reset_index()

    fig.add_trace(go.Scatter(x=g_s["tranche"].astype(str), y=g_s["pm2_reel"], mode="lines+markers", name="Prix/m² Affiché Médian", line={"color": "#f59e0b", "width": 3}), row=2, col=1)
    fig.add_trace(go.Scatter(x=g_s["tranche"].astype(str), y=g_s["pm2_pred"], mode="lines+markers", name="Prix/m² Prédit HistGB", line={"color": "#10b981", "width": 3}), row=2, col=1)

    if "grp_mae" in df_abl.columns:
        fig.add_trace(go.Bar(x=df_abl["etape"], y=df_abl["mae"], name="KFold MAE (€)", marker_color="#3b82f6"), row=2, col=2)
        fig.add_trace(go.Bar(x=df_abl["etape"], y=df_abl["grp_mae"], name="GroupKFold Quartier MAE (€)", marker_color="#a855f7"), row=2, col=2)

    fig.update_layout(
        title="Audit Features HistGradientBoosting — Grand Poitiers (Tracking MLflow)",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font={"color": "#f8fafc", "size": 11},
        height=880,
        barmode="group",
    )
    os.makedirs(os.path.dirname(html_path) or "csv", exist_ok=True)
    fig.write_html(html_path, include_plotlyjs="cdn")
    if MLFLOW_DISPO:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run(run_name="Rapport_Audit_HistGB"):
            mlflow.log_artifact(html_path)
    print(f"\n[Graphiques] Rapport sauvegardé dans '{html_path}' (et loggué dans MLflow).")
    return html_path


CATEGORIES_TAILLE = [
    ("< 20 m²", 0.0, 20.0),
    ("20 – 35 m²", 20.0, 35.0),
    ("35 – 60 m²", 35.0, 60.0),
    ("60 – 90 m²", 60.0, 90.0),
    ("90 – 130 m²", 90.0, 130.0),
    (">= 130 m²", 130.0, 1e6),
]


def evaluer_multivilles_complet(
    df_hist: pd.DataFrame | None = None,
    out_json_path: str = os.path.join("models", "evaluation_report.json"),
) -> dict:
    """
    Double évaluation comparative hybride (ML sur `log(prix)` + AVM vs ML sur résidus + AVM),
    calculée globalement, par ville et par catégorie de taille (< 20 m² jusqu'à >= 130 m²),
    puis sauvegardée dans `models/evaluation_report.json`.
    """
    import json
    from datetime import datetime, timezone
    from src.ingestion.territoires import VILLES_CENTRES, resoudre_territoire

    if df_hist is None:
        df_hist = importer_table_historique()
    if df_hist is None or df_hist.empty:
        return {"error": "Aucune donnée historique disponible"}

    # Garder uniquement les 12 villes-centres V1 si la colonne est présente
    if "est_ville_centre" in df_hist.columns:
        sub_vc = df_hist[df_hist["est_ville_centre"] == True]  # noqa: E712
        if len(sub_vc) >= 30:
            df_hist = sub_vc.copy()

    X, y = fe.construire_X_y(df_hist)
    y_arr = y.to_numpy(dtype=float)
    surf = pd.to_numeric(df_hist.loc[X.index, "surface_m2"], errors="coerce").fillna(50.0).to_numpy(dtype=float)

    n_splits = min(5, max(2, len(X) // 30))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Hybride 1 : ML sur log(prix) + AVM si disponible
    modele_direct = tr.construire_modele()
    y_ml_direct = cross_val_predict(modele_direct, X, y, cv=kf, n_jobs=-1)

    # Ancrage AVM/DVF local pour fusion et pour Hybride 2 (résidus)
    avm_col = pd.to_numeric(
        df_hist.loc[X.index].get("avm_comp_blended_euro", df_hist.loc[X.index].get("valeur_dvf_ajustee_taille")),
        errors="coerce",
    )
    dvf_ref = pd.to_numeric(df_hist.loc[X.index].get("pm2_voisinage"), errors="coerce") * surf
    ancrage = avm_col.fillna(dvf_ref).fillna(float(np.median(y_arr))).clip(lower=15000.0).to_numpy(dtype=float)

    y_hyb_direct = np.where(np.isfinite(avm_col.to_numpy()), 0.75 * y_ml_direct + 0.25 * ancrage, y_ml_direct)

    # Hybride 2 : ML sur résidu log(prix / ancrage_dvf) + fusion AVM
    y_residu_cible = np.log(np.maximum(y_arr, 1000.0) / np.maximum(ancrage, 1000.0))
    from sklearn.base import clone
    modele_residu = clone(tr.construire_modele().regressor)
    y_pred_res_log = cross_val_predict(modele_residu, X, y_residu_cible, cv=kf, n_jobs=-1)
    y_ml_res = ancrage * np.exp(np.clip(y_pred_res_log, -1.5, 1.5))
    y_hyb_res = np.where(np.isfinite(avm_col.to_numpy()), 0.75 * y_ml_res + 0.25 * ancrage, y_ml_res)

    # Intervalles Q10 / Q90 pour mesurer la couverture réelle
    try:
        q10_pred = cross_val_predict(tr.construire_modele_quantile(0.10), X, y, cv=kf, n_jobs=-1)
        q90_pred = cross_val_predict(tr.construire_modele_quantile(0.90), X, y, cv=kf, n_jobs=-1)
    except Exception:
        q10_pred, q90_pred = None, None

    m_direct = calculer_metriques_cv(y_arr, y_hyb_direct, surf, q10=q10_pred, q90=q90_pred)
    m_residus = calculer_metriques_cv(y_arr, y_hyb_res, surf, q10=q10_pred, q90=q90_pred)
    gagnant = "hybride_residus" if m_residus.get("mdape", 999) < m_direct.get("mdape", 999) else "hybride_log_prix"
    y_best = y_hyb_res if gagnant == "hybride_residus" else y_hyb_direct

    global_metrics = m_residus if gagnant == "hybride_residus" else m_direct

    # Métriques par ville
    sub_meta = df_hist.loc[X.index]
    cps = sub_meta["code_postal"].tolist() if "code_postal" in sub_meta.columns else [None] * len(sub_meta)
    villes = sub_meta["ville"].tolist() if "ville" in sub_meta.columns else [None] * len(sub_meta)
    codes_terr = np.array([
        (sub_meta["code_territoire"].iloc[i] if "code_territoire" in sub_meta.columns and pd.notna(sub_meta["code_territoire"].iloc[i]) else resoudre_territoire(cps[i], villes[i]) or "poitiers")
        for i in range(len(sub_meta))
    ])

    par_ville = []
    for code, cfg in VILLES_CENTRES.items():
        mask_v = codes_terr == code
        n_v = int(mask_v.sum())
        if n_v < 5:
            par_ville.append({
                "code_territoire": code,
                "nom": cfg.nom,
                "nb_biens": n_v,
                "maturite": "En calibration",
            })
            continue
        mv = calculer_metriques_cv(
            y_arr[mask_v], y_best[mask_v], surf[mask_v],
            q10=q10_pred[mask_v] if q10_pred is not None else None,
            q90=q90_pred[mask_v] if q90_pred is not None else None,
        )
        est_fiable = n_v >= 80 and mv.get("mdape", 99) <= 12.0 and mv.get("within_20pct", 0) >= 80.0
        par_ville.append({
            "code_territoire": code,
            "nom": cfg.nom,
            "maturite": "Fiable" if est_fiable else "En calibration",
            **mv,
        })

    # Métriques par catégorie de taille (<20 m² jusqu'à >=130 m²)
    par_categorie_taille = []
    for label_t, s_min, s_max in CATEGORIES_TAILLE:
        mask_t = (surf >= s_min) & (surf < s_max)
        n_t = int(mask_t.sum())
        if n_t == 0:
            par_categorie_taille.append({"categorie": label_t, "nb_biens": 0})
            continue
        mt = calculer_metriques_cv(
            y_arr[mask_t], y_best[mask_t], surf[mask_t],
            q10=q10_pred[mask_t] if q10_pred is not None else None,
            q90=q90_pred[mask_t] if q90_pred is not None else None,
        )
        par_categorie_taille.append({"categorie": label_t, **mt})

    # Comparatif Avant / Après
    avant_metrics = None
    if os.path.exists(out_json_path):
        try:
            with open(out_json_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
                avant_metrics = prev.get("global_metrics")
        except Exception:
            avant_metrics = None

    rapport = {
        "model_version": os.path.basename(tr.CHEMIN_MODELE),
        "date_evaluation": datetime.now(timezone.utc).isoformat(),
        "approche_retenue": gagnant,
        "global_metrics": global_metrics,
        "comparatif_modeles": {
            "hybride_log_prix": m_direct,
            "hybride_residus": m_residus,
            "gagnant": gagnant,
        },
        "par_ville": par_ville,
        "par_categorie_taille": par_categorie_taille,
        "avant_apres": {
            "avant": avant_metrics,
            "apres": global_metrics,
        },
    }
    os.makedirs(os.path.dirname(out_json_path) or "models", exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(rapport, f, ensure_ascii=False, indent=2)
    print(f"[Évaluation Multi-Villes] Rapport sauvegardé -> {out_json_path} (gagnant : {gagnant}, MdAPE={global_metrics.get('mdape')}%, ±20%={global_metrics.get('within_20pct')}%)")
    return rapport


def executer_improvement(ouvrir_graphe: bool = False, **_kwargs) -> dict:
    df_hist = importer_table_historique()
    res = executer_etude_features(df_hist)
    html_path = generer_rapport_html(res)
    if ouvrir_graphe:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(html_path)}")
    return res


if __name__ == "__main__":
    executer_improvement()
