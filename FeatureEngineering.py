"""
Préparation des annonces pour prédire le prix (ML supervisé).

    pip install pandas numpy scikit-learn
    python preparation_ml.py          # lit annonces.csv

Étapes : doublons -> features texte -> features géographiques
         -> (option) prix/m² de voisinage via DVF -> modèle avec validation croisée.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import BallTree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

# Repères de Poitiers (coordonnées approximatives, à vérifier)
CENTRE = (46.5802, 0.3404)
GARE = (46.5826, 0.3333)


# --------------------------------------------------------------------------
# 1. Doublons
# --------------------------------------------------------------------------
def marquer_doublons(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les id identiques, puis MARQUE (sans supprimer) les quasi-doublons."""
    df = df.drop_duplicates(subset="id").copy()

    # Même bien publié par deux agences/sites : même prix, surface, pièces, CP
    cle = ["prix", "surface_m2", "pieces", "code_postal"]
    df["doublon_probable"] = df.duplicated(subset=cle, keep=False)

    n = int(df["doublon_probable"].sum())
    print(f"{n} lignes sont des doublons probables (à vérifier à la main).")
    if n:
        print(df[df["doublon_probable"]].sort_values(cle)[["id", "titre", "prix", "surface_m2", "quartier"]])
    return df


def garder_un_doublon(df: pd.DataFrame) -> pd.DataFrame:
    """Après vérification : ne garde que la 1re annonce de chaque groupe."""
    cle = ["prix", "surface_m2", "pieces", "code_postal"]
    return df.drop_duplicates(subset=cle, keep="first").drop(columns="doublon_probable")


# --------------------------------------------------------------------------
# 2. Texte -> variables utilisables
# --------------------------------------------------------------------------
MOTS = {
    "terrasse": r"terrasse",
    "balcon": r"balcon|loggia",
    "parking": r"parking|garage|stationnement",
    "ascenseur": r"ascenseur",
    "cave": r"\bcave\b",
    "jardin": r"jardin",
    "renove": r"r[ée]nov",
    "a_renover": r"travaux|à rafra[iî]chir|à r[ée]nover",
    "vendu_loue": r"vendu lou",
    "dernier_etage": r"dernier [ée]tage",
    "lumineux": r"lumineu",
    "calme": r"calme",
}


def features_texte(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    desc = df["description"].fillna("").str.lower()
    for nom, motif in MOTS.items():
        df[f"txt_{nom}"] = desc.str.contains(motif, regex=True).astype(int)
    df["txt_longueur"] = desc.str.len()

    # Agence : on garde le réseau ("Iad France"), pas le nom du conseiller (bruit)
    df["reseau"] = (
        df["description"].fillna("").str.extract(r"^([A-Za-zÀ-ÿ' ]+?)\s+-\s", expand=False)
    )
    return df


# --------------------------------------------------------------------------
# 3. Géographie
# --------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def features_geo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dist_centre_km"] = haversine_km(df["latitude"], df["longitude"], *CENTRE)
    df["dist_gare_km"] = haversine_km(df["latitude"], df["longitude"], *GARE)
    return df


def prix_m2_voisinage(df: pd.DataFrame, dvf: pd.DataFrame, k: int = 30, rayon_m: int = 1000) -> pd.Series:
    """
    Médiane du prix/m² des k ventes réelles (DVF) les plus proches de chaque annonce,
    dans un rayon de `rayon_m` mètres. Plus fiable qu'un nom de quartier.

    `dvf` : fichier "DVF géolocalisé" de la commune (data.gouv.fr), avec les colonnes
    id_mutation, nature_mutation, valeur_fonciere, surface_reelle_bati, type_local,
    latitude, longitude.
    """
    dvf = dvf[(dvf["nature_mutation"] == "Vente") & (dvf["type_local"] == "Appartement")].copy()
    # une vente multi-lots répartit mal le prix : on garde les ventes à 1 seule ligne
    dvf = dvf[dvf.groupby("id_mutation")["id_mutation"].transform("size") == 1]
    dvf = dvf.dropna(subset=["valeur_fonciere", "surface_reelle_bati", "latitude", "longitude"])
    dvf = dvf[dvf["surface_reelle_bati"] > 10]
    dvf["pm2"] = dvf["valeur_fonciere"] / dvf["surface_reelle_bati"]
    dvf = dvf[dvf["pm2"].between(500, 8000)]  # écarte les valeurs aberrantes

    tree = BallTree(np.radians(dvf[["latitude", "longitude"]].to_numpy()), metric="haversine")
    ok = df[["latitude", "longitude"]].notna().all(axis=1)
    pts = np.radians(df.loc[ok, ["latitude", "longitude"]].to_numpy())

    dist, idx = tree.query(pts, k=min(k, len(dvf)))
    pm2 = dvf["pm2"].to_numpy()[idx]
    pm2 = np.where(dist * 6_371_000 <= rayon_m, pm2, np.nan)

    out = pd.Series(np.nan, index=df.index, name="pm2_voisinage")
    with np.errstate(all="ignore"):
        out[ok] = np.nanmedian(pm2, axis=1)
    return out


# --------------------------------------------------------------------------
# 4. Modèle
# --------------------------------------------------------------------------
ORDRE_DPE = {c: i for i, c in enumerate("ABCDEFG", start=1)}
CATEGORIELLES = ["quartier", "reseau"]


def construire_X_y(df: pd.DataFrame):
    df = df.copy()
    df["dpe_num"] = df["dpe"].map(ORDRE_DPE)
    df["ges_num"] = df["ges"].map(ORDRE_DPE)
    for c in CATEGORIELLES:
        df[c] = df[c].fillna("inconnu")

    numeriques = [
        "surface_m2", "pieces", "nbr_chambres", "etage", "annee_construction",
        "latitude", "longitude", "dist_centre_km", "dist_gare_km",
        "dpe_num", "ges_num", "txt_longueur",
    ] + [c for c in df.columns if c.startswith("txt_") and c != "txt_longueur"]
    if "pm2_voisinage" in df.columns:
        numeriques.append("pm2_voisinage")

    X = df[numeriques + CATEGORIELLES].copy()
    X[numeriques] = X[numeriques].astype(float)  # Int64 -> float, NA -> NaN
    return X, df["prix"].astype(float)


def construire_modele() -> TransformedTargetRegressor:
    # TargetEncoder : remplace chaque quartier par le prix moyen associé, calculé
    # DANS les plis d'entraînement (cross-fitting) -> pas de fuite vers le test.
    prepa = ColumnTransformer(
        [("cat", TargetEncoder(target_type="continuous", random_state=0), CATEGORIELLES)],
        remainder="passthrough",
    )
    modele = Pipeline([
        ("prepa", prepa),
        ("gbm", HistGradientBoostingRegressor(  # gère les NaN nativement
            max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=8, random_state=0)),
    ])
    # On prédit log(prix) : les prix sont asymétriques, l'erreur devient relative
    return TransformedTargetRegressor(regressor=modele, func=np.log, inverse_func=np.exp)


def evaluer(df: pd.DataFrame):
    X, y = construire_X_y(df)
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    mae = -cross_val_score(construire_modele(), X, y, cv=cv, scoring="neg_mean_absolute_error")
    r2 = cross_val_score(construire_modele(), X, y, cv=cv, scoring="r2")

    # Référence bête : prix/m² médian x surface (si le modèle ne la bat pas, il ne sert à rien)
    pm2_med = (y / X["surface_m2"]).median()
    mae_ref = (y - pm2_med * X["surface_m2"]).abs().mean()

    print(f"MAE modèle   : {mae.mean():,.0f} € (+/- {mae.std():,.0f})")
    print(f"R²           : {r2.mean():.2f}")
    print(f"MAE référence: {mae_ref:,.0f} € (prix/m² médian x surface)")


if __name__ == "__main__":
    df = pd.read_csv("annonces.csv")
    df = marquer_doublons(df)
    df = garder_un_doublon(df)
    df = features_texte(df)
    df = features_geo(df)

    # Option : enrichir avec les ventes réelles DVF
    # dvf = pd.read_csv("dvf_poitiers.csv")
    # df["pm2_voisinage"] = prix_m2_voisinage(df, dvf)

    evaluer(df)