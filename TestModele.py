"""
Script indépendant d'expérimentation ("Bac à sable") — Modifiable à votre guise.

Ce qu'il fait :
1. Extrait depuis PostgreSQL (ou `csv/` en secours) :
   - La table historique nettoyée (`offres_historique_features`) -> pour entraîner et évaluer
   - La table récente nettoyée à prédire (`offres_recentes_features`) -> pour prédire
2. Entraîne un modèle Random Forest (`RandomForestRegressor`) avec les paramètres définis en haut du fichier.
3. Calcule et affiche les métriques en ligne de commande (R², MAE, MedAE, RMSE, MAPE, MdAPE, paliers).
4. Enregistre automatiquement le run dans MLflow dans l'onglet (expérience) **"test"**.

Usage :
    python TestModele.py
    mlflow ui --port 5000     # Puis ouvrir http://127.0.0.1:5000 -> onglet "test"
"""

import getpass
import os
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder


# ==============================================================================
# 1. ZONE À MODIFIER À VOTRE GUISE (Hyperparamètres, Features, Nom du Run)
# ==============================================================================
NOM_ONGLET_MLFLOW = "test"
NOM_RUN = "RandomForest_v1"

# Hyperparamètres du Random Forest (modifiables librement)
PARAMS_RF = {
    "n_estimators": 300,        # Nombre d'arbres
    "max_depth": 12,            # Profondeur max d'un arbre (None = illimité)
    "min_samples_split": 4,     # Min d'échantillons pour diviser un nœud
    "min_samples_leaf": 2,      # Min d'échantillons dans une feuille
    "max_features": "sqrt",     # "sqrt", "log2", ou 1.0 (toutes les features)
    "random_state": 42,
    "n_jobs": -1,
}

# Variables numériques à utiliser (commentez/décommentez pour tester)
FEATURES_NUMERIQUES = [
    "surface_m2",
    "pieces",
    "nbr_chambres",
    "etage",
    "annee_construction",
    "latitude",
    "longitude",
    "dist_centre_km",
    "dist_gare_km",
    "dpe_num",
    "ges_num",
    "txt_longueur",
    "surface_ext_m2",
    "charges_annuelles",
    "llm_etat_score",
    "llm_travaux_score",
    "llm_note_offre",
    "pm2_voisinage",
    "txt_jardin",
    "txt_terrasse",
    "txt_balcon",
    "txt_parking",
    "txt_ascenseur",
    "txt_cave",
    "txt_renove",
    "txt_a_renover",
    "txt_vendu_loue",
    "txt_dernier_etage",
    "txt_lumineux",
    "txt_calme",
]

# Variables catégorielles à encoder
FEATURES_CATEGORIELLES = [
    "quartier",
    "reseau",
    "type_bien",
]

# Utiliser log(prix) comme cible ? (True améliore souvent la précision sur les prix immobiliers)
UTILISER_LOG_PRIX = True


# ==============================================================================
# 2. EXTRACTION DES DONNÉES NETTOYÉES DEPUIS POSTGRESQL (OU CSV)
# ==============================================================================
USER = getpass.getuser()
ENGINE_URL = f"postgresql+psycopg2://{USER}@localhost:5432/immobilier"
ORDRE_DPE = {c: i for i, c in enumerate("ABCDEFG", start=1)}


def extraire_donnees() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extrait :
    - `df_hist`    : la table historique nettoyée (`offres_historique_features`)
    - `df_predire` : la table récente nettoyée à prédire (`offres_recentes_features`)
    """
    try:
        engine = create_engine(ENGINE_URL)
        with engine.connect() as conn:
            df_hist = pd.read_sql(text("SELECT * FROM offres_historique_features"), conn)
            df_predire = pd.read_sql(text("SELECT * FROM offres_recentes_features"), conn)
        source = "PostgreSQL (base 'immobilier')"
    except Exception:
        df_hist = pd.read_csv(os.path.join("csv", "annonces_historique_features.csv"))
        df_predire = pd.read_csv(os.path.join("csv", "annonces_recentes_features.csv"))
        source = "Fichiers CSV miroirs (dossier 'csv/')"

    print(f"[Extraction] Source : {source}")
    print(f"  • Table historique nettoyée (entraînement) : {len(df_hist)} lignes")
    print(f"  • Table récente nettoyée (à prédire)       : {len(df_predire)} lignes")
    return df_hist, df_predire


def preparer_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Prépare la matrice X et le vecteur cible y à partir des colonnes choisies en haut du script."""
    df_clean = df[df["prix"].notna() & (df["prix"] > 0)].copy()
    df_clean["dpe_num"] = df_clean["dpe"].map(ORDRE_DPE)
    df_clean["ges_num"] = df_clean["ges"].map(ORDRE_DPE)

    num_cols = [c for c in FEATURES_NUMERIQUES if c in df_clean.columns]
    cat_cols = [c for c in FEATURES_CATEGORIELLES if c in df_clean.columns]

    for c in cat_cols:
        df_clean[c] = df_clean[c].fillna("inconnu").astype(str)

    X = df_clean[num_cols + cat_cols].copy()
    X[num_cols] = X[num_cols].astype(float)
    y = df_clean["prix"].astype(float)
    return X, y, df_clean


# ==============================================================================
# 3. CONSTRUCTION DU RANDOM FOREST, ÉVALUATION & PRÉDICTION
# ==============================================================================
def construire_pipeline_rf(cat_cols: list[str]):
    """Crée la pipeline de prétraitement + RandomForestRegressor."""
    cv_interne = KFold(n_splits=5, shuffle=True, random_state=42)
    preprocesseur = ColumnTransformer(
        transformers=[
            ("cat", TargetEncoder(target_type="continuous", cv=cv_interne), cat_cols),
        ],
        remainder=SimpleImputer(strategy="median"),
    )
    rf = RandomForestRegressor(**PARAMS_RF)
    pipe = Pipeline([
        ("prepa", preprocesseur),
        ("rf", rf),
    ])
    if UTILISER_LOG_PRIX:
        return TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp)
    return pipe


def calculer_metriques(y_reel: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcule l'ensemble des métriques de performance."""
    err_abs = np.abs(y_reel - y_pred)
    err_pct = err_abs / np.maximum(y_pred, 1.0) * 100.0

    mae = float(np.mean(err_abs))
    med_ae = float(np.median(err_abs))
    rmse = float(np.sqrt(np.mean((y_reel - y_pred) ** 2)))
    mape = float(np.mean(err_pct))
    mdape = float(np.median(err_pct))

    ss_res = float(np.sum((y_reel - y_pred) ** 2))
    ss_tot = float(np.sum((y_reel - np.mean(y_reel)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "R2": round(r2, 4),
        "MAE_euro": round(mae, 0),
        "MedAE_euro": round(med_ae, 0),
        "RMSE_euro": round(rmse, 0),
        "MAPE_pct": round(mape, 2),
        "MdAPE_pct": round(mdape, 2),
        "Part_erreur_inf_5pct": round(float(np.mean(err_pct <= 5.0) * 100), 1),
        "Part_erreur_inf_10pct": round(float(np.mean(err_pct <= 10.0) * 100), 1),
        "Part_erreur_inf_15pct": round(float(np.mean(err_pct <= 15.0) * 100), 1),
        "Part_erreur_inf_20pct": round(float(np.mean(err_pct <= 20.0) * 100), 1),
    }


# ==============================================================================
# 4. EXÉCUTION & ENREGISTREMENT DANS L'ONGLET MLFLOW "test"
# ==============================================================================
def main():
    # 1. Extraction des 2 tables nettoyées
    df_hist, df_predire = extraire_donnees()
    X_train, y_train, _ = preparer_X_y(df_hist)
    X_pred, y_pred_reel, df_pred_clean = preparer_X_y(df_predire)

    cat_cols = [c for c in FEATURES_CATEGORIELLES if c in X_train.columns]
    modele = construire_pipeline_rf(cat_cols)

    # 2. Validation croisée 5 plis sur l'historique pour avoir les vraies métriques hors-échantillon
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_cv_pred = cross_val_predict(modele, X_train, y_train, cv=cv)
    metriques = calculer_metriques(y_train.to_numpy(), y_cv_pred)

    # 3. Entraînement sur 100 % de l'historique nettoyé & prédiction sur la table à prédire
    modele.fit(X_train, y_train)
    predictions_recentes = np.round(modele.predict(X_pred), 0)

    df_resultats = df_pred_clean[["id", "ville", "quartier", "type_bien", "surface_m2", "prix"]].copy()
    df_resultats["prix_predit_rf"] = predictions_recentes
    df_resultats["ecart_euro"] = df_resultats["prix"] - df_resultats["prix_predit_rf"]
    df_resultats["ecart_pct"] = np.round(
        df_resultats["ecart_euro"] / df_resultats["prix_predit_rf"] * 100, 1
    )

    # 4. Affichage des métriques en ligne de commande
    print("\n" + "=" * 72)
    print(f"  RÉSULTATS RANDOM FOREST — ONGLET MLFLOW : '{NOM_ONGLET_MLFLOW}' ({NOM_RUN})")
    print("=" * 72)
    print(f"  • Variables utilisées : {X_train.shape[1]} features sur {len(X_train)} offres historiques")
    print(f"  • R² (variance expl.) : {metriques['R2']:.4f}")
    print(f"  • MAE (erreur moy.)   : {metriques['MAE_euro']:,.0f} € (MAPE : ±{metriques['MAPE_pct']:.2f}%)")
    print(f"  • MedAE (erreur méd.) : {metriques['MedAE_euro']:,.0f} € (MdAPE : ±{metriques['MdAPE_pct']:.2f}%)")
    print(f"  • RMSE                : {metriques['RMSE_euro']:,.0f} €")
    print("-" * 72)
    print(
        f"  • Paliers de précision : "
        f"<=5% : {metriques['Part_erreur_inf_5pct']}% | "
        f"<=10% : {metriques['Part_erreur_inf_10pct']}% | "
        f"<=15% : {metriques['Part_erreur_inf_15pct']}% | "
        f"<=20% : {metriques['Part_erreur_inf_20pct']}%"
    )
    print("-" * 72)
    print("  • Aperçu des 5 plus fortes décotes sur la table à prédire :")
    top5 = df_resultats.sort_values("ecart_pct", ascending=True).head(5)
    for _, r in top5.iterrows():
        print(
            f"    - [{r['id']}] {r['type_bien']} {r['surface_m2']:.0f} m² ({r['quartier']}) : "
            f"Affiché {r['prix']:,.0f} € vs Prédit RF {r['prix_predit_rf']:,.0f} € ({r['ecart_pct']:+.1f}%)"
        )
    print("=" * 72)

    # 5. Enregistrement dans MLflow dans l'onglet "test"
    mlflow.set_experiment(NOM_ONGLET_MLFLOW)
    with mlflow.start_run(run_name=NOM_RUN):
        mlflow.log_params({
            **PARAMS_RF,
            "utiliser_log_prix": UTILISER_LOG_PRIX,
            "nb_features": X_train.shape[1],
            "nb_lignes_train": len(X_train),
            "nb_lignes_pred": len(X_pred),
        })
        mlflow.log_metrics(metriques)
        try:
            mlflow.sklearn.log_model(
                modele,
                name="random_forest_model",
                serialization_format="cloudpickle",
            )
        except TypeError:
            mlflow.sklearn.log_model(
                modele,
                artifact_path="random_forest_model",
                serialization_format="cloudpickle",
            )

    print(f"\n✅ Run '{NOM_RUN}' enregistré dans MLflow sous l'onglet (Experiment) '{NOM_ONGLET_MLFLOW}'.")
    print("   Lancez `mlflow ui --port 5000` puis cliquez sur l'onglet 'test' à gauche.")


if __name__ == "__main__":
    main()
