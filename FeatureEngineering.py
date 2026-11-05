"""
Préparation des annonces, enrichissement LLM (Ollama), intégration des ventes
notariées réelles (DVF Poitiers) et évaluation approfondie pour prédire le prix
et estimer la marge de négociation.

    python FeatureEngineering.py          # télécharge DVF, enrichit via Ollama et évalue
    python Dashboard.py                   # lance le dashboard interactif Dash
"""

import json
import os
import re
import urllib.request
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import BallTree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

# Repères de Poitiers (coordonnées centre-ville & gare)
CENTRE = (46.5802, 0.3404)
GARE = (46.5826, 0.3333)

LABELS_FEATURES = {
    "surface_m2": "Surface habitable (m²)",
    "pm2_voisinage": "Prix/m² Notaire DVF voisinage",
    "type_bien": "Type de bien (Maison / Appart.)",
    "quartier": "Quartier",
    "dist_centre_km": "Distance Centre-ville (km)",
    "dist_gare_km": "Distance Gare (km)",
    "dpe_num": "Note DPE (A à G)",
    "annee_construction": "Année de construction",
    "ges_num": "Note GES (A à G)",
    "nbr_chambres": "Nombre de chambres",
    "pieces": "Nombre de pièces",
    "etage": "Étage",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "txt_longueur": "Longueur du descriptif",
    "reseau": "Agence / Réseau",
    "surface_ext_m2": "Surface terrain/jardin extraite (m²)",
    "charges_annuelles": "Charges annuelles extraites (€)",
    "llm_etat_score": "Note État du bien (LLM 1-5)",
    "llm_travaux_score": "Niveau de travaux (LLM 0-3)",
    "llm_note_offre": "Note Attractivité (LLM /10)",
    "txt_jardin": "Mot-clé : Jardin",
    "txt_a_renover": "Mot-clé : À rénover / Travaux",
    "txt_terrasse": "Mot-clé : Terrasse",
    "txt_parking": "Mot-clé : Parking / Garage",
    "txt_balcon": "Mot-clé : Balcon / Loggia",
    "txt_ascenseur": "Mot-clé : Ascenseur",
    "txt_cave": "Mot-clé : Cave",
    "txt_renove": "Mot-clé : Rénové",
    "txt_vendu_loue": "Mot-clé : Vendu loué",
    "txt_dernier_etage": "Mot-clé : Dernier étage",
    "txt_lumineux": "Mot-clé : Lumineux",
    "txt_calme": "Mot-clé : Calme",
}

# --------------------------------------------------------------------------
# 1. Nettoyage, URLs & Doublons
# --------------------------------------------------------------------------
QUARTIERS_HARMONISES = {
    "Centre ville": "Centre-ville",
    "Saint-eloi breuil-mingot": "Saint-Eloi - Breuil - Mingot",
    "Trois cites": "Trois Cités",
    "Montbernage pont-neuf": "Montbernage - Pont Neuf",
    "Faubourgs ouest": "Ouest",
}

TYPES_BIEN_HARMONISES = {
    "Apartment": "flat",
}

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def reparer_url(row: pd.Series) -> str:
    """Garantit un lien direct vers l'annonce (Jinka ou BienIci) au lieu d'un lien PDF/Géorisques."""
    url = str(row.get("url", "") or "").strip()
    ad_id = str(row.get("id", "") or "").strip()
    if (
        not url
        or url == "nan"
        or "bareme_honoraires" in url
        or "georisques.gouv.fr" in url
    ):
        if UUID_RE.match(ad_id):
            return f"https://www.jinka.fr/ad/{ad_id}"
        if ad_id:
            return f"https://www.bienici.com/annonce/{ad_id}"
        return ""
    return url


def nettoyer_annonces(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Supprime les lignes sans prix ou surface valide, harmonise les catégories et répare les URLs."""
    df = df.copy()
    avant = len(df)
    df = df.dropna(subset=["prix", "surface_m2"])
    df = df[(df["prix"] > 0) & (df["surface_m2"] > 0)].copy()
    suppr = avant - len(df)
    if verbose and suppr:
        print(f"{suppr} annonces supprimées (prix ou surface manquant/invalide).")

    if "quartier" in df.columns:
        df["quartier"] = df["quartier"].replace(QUARTIERS_HARMONISES)
    if "type_bien" in df.columns:
        df["type_bien"] = df["type_bien"].replace(TYPES_BIEN_HARMONISES)
    df["url"] = df.apply(reparer_url, axis=1)
    return df


def marquer_doublons(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Supprime les id identiques, puis MARQUE (sans supprimer) les quasi-doublons."""
    df = df.drop_duplicates(subset="id").copy()

    cle = ["prix", "surface_m2", "pieces", "code_postal"]
    df["doublon_probable"] = df.duplicated(subset=cle, keep=False)

    n = int(df["doublon_probable"].sum())
    if verbose:
        print(f"{n} lignes sont des doublons probables (même prix, surface, pièces, CP).")
    return df


def garder_un_doublon(df: pd.DataFrame) -> pd.DataFrame:
    """Ne garde que la 1re annonce de chaque groupe de doublons."""
    cle = ["prix", "surface_m2", "pieces", "code_postal"]
    cols_to_drop = ["doublon_probable"] if "doublon_probable" in df.columns else []
    return df.drop_duplicates(subset=cle, keep="first").drop(columns=cols_to_drop)


# --------------------------------------------------------------------------
# 2. Texte (Regex intelligentes + Enrichissement LLM Ollama)
# --------------------------------------------------------------------------
MOTS = {
    "terrasse": r"terrasse",
    "balcon": r"balcon|loggia",
    "parking": r"parking|garage|stationnement|box",
    "ascenseur": r"(?<!sans\s)ascenseur",
    "cave": r"\bcave\b|\bsous-sol\b",
    "jardin": r"jardin|parcelle|terrain",
    "renove": r"r[ée]nov[ée]|refait\s+[àa]\s+neuf|parfait\s+[ée]tat|aucun\s+travaux",
    # Attention au piège "aucun travaux à prévoir" / "pas de travaux" !
    "a_renover": r"(?<!aucun\s)(?<!pas\sde\s)(?<!sans\s)travaux\s+[àa]\s+pr[ée]voir|[àa]\s+rafra[iî]chir|[àa]\s+r[ée]nover|r[ée]novation\s+[àa]\s+pr[ée]voir|remettre\s+aux\s+normes",
    "vendu_loue": r"vendu\s+lou[ée]|actuellement\s+lou[ée]|locataire\s+en\s+place",
    "dernier_etage": r"dernier\s+[ée]tage",
    "lumineux": r"lumineu|baign[ée]\s+de\s+lumi[èe]re|plein\s+sud",
    "calme": r"\bcalme\b|sans\s+vis[- ]+[àa][- ]vis",
}


def extraire_chiffres_texte(desc_serie: pd.Series) -> pd.DataFrame:
    """Extrait par regex la surface extérieure (terrain/jardin) et les charges annuelles."""
    s = desc_serie.fillna("").str.lower()

    # Surface terrain / jardin / parcelle (ex: "terrain de 362 m²", "jardin privatif de 160m2")
    terrain = s.str.extract(
        r"(?:terrain|jardin(?:\s+privatif)?|parcelle)\s+(?:clos\s+|arbor[ée]\s+)?(?:de\s+|d'environ\s+)?(\d{2,5})\s*m[²2]",
        expand=False,
    )
    # Charges annuelles (ex: "charges annuelles : 1572 €" ou "soit 1572 € annuel")
    charges = s.str.extract(
        r"(?:charges\s+annuelles\s*:\s*|soit\s+)(\d{2,5})(?:[.,]\d+)?\s*(?:€|euros|¤)",
        expand=False,
    )
    return pd.DataFrame({
        "surface_ext_m2": pd.to_numeric(terrain, errors="coerce").fillna(0.0),
        "charges_annuelles": pd.to_numeric(charges, errors="coerce"),
    })


def features_texte(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    desc = df["description"].fillna("").str.lower()
    for nom, motif in MOTS.items():
        df[f"txt_{nom}"] = desc.str.contains(motif, regex=True).astype(int)
    df["txt_longueur"] = desc.str.len()

    extra = extraire_chiffres_texte(df["description"])
    df["surface_ext_m2"] = extra["surface_ext_m2"]
    df["charges_annuelles"] = extra["charges_annuelles"]

    # Agence : on garde le réseau ("Iad France"), pas le nom du conseiller (bruit)
    df["reseau"] = (
        df["description"].fillna("").str.extract(r"^([A-Za-zÀ-ÿ' ]+?)\s+-\s", expand=False)
    )
    return df


def calculer_heuristique_llm(row: pd.Series) -> dict:
    """
    Calcul déterministe expert (et fallback instantané) des scores qualitatifs :
    - llm_etat_score (1 à 5)
    - llm_travaux_score (0 à 3)
    - llm_note_offre (note sur 10)
    - llm_resume (synthèse des atouts/faiblesses)
    """
    desc = str(row.get("description", "") or "").lower()
    dpe = str(row.get("dpe", "") or "").upper()
    annee = row.get("annee_construction")

    # 1. Niveau de travaux (0 = aucun, 1 = rafraîchissement, 2 = rénovation partielle, 3 = rénovation lourde)
    if re.search(r"enti[èe]rement\s+[àa]\s+r[ée]nover|gros\s+travaux|r[ée]novation\s+compl[èe]te", desc):
        travaux = 3
    elif re.search(r"[àa]\s+r[ée]nover|remettre\s+aux\s+normes|simple\s+vitrage", desc) or dpe in ("F", "G"):
        travaux = 2
    elif re.search(r"[àa]\s+rafra[iî]chir|rafra[iî]chissement|quelques\s+travaux", desc):
        travaux = 1
    else:
        travaux = 0

    # 2. Note d'état du bien (1 à 5)
    if pd.notna(annee) and float(annee) >= 2020 or "neuf" in desc or "re 2020" in desc:
        etat = 5
    elif travaux == 0 and re.search(r"r[ée]nov[ée]|parfait\s+[ée]tat|tr[èe]s\s+bon\s+[ée]tat|aucun\s+travaux", desc):
        etat = 4
    elif travaux == 0:
        etat = 3
    elif travaux == 1:
        etat = 2
    else:
        etat = 1

    # 3. Note d'attractivité de l'offre (/10)
    note = 5.0
    note += 0.8 if row.get("txt_parking", 0) else 0.0
    note += 0.8 if (row.get("txt_terrasse", 0) or row.get("txt_jardin", 0)) else (0.4 if row.get("txt_balcon", 0) else 0.0)
    note += 0.5 if row.get("txt_lumineux", 0) else 0.0
    note += 0.5 if row.get("txt_calme", 0) else 0.0
    note += 0.4 if row.get("txt_ascenseur", 0) else 0.0
    note += (etat - 3) * 0.7
    if dpe in ("A", "B"):
        note += 0.8
    elif dpe == "C":
        note += 0.4
    elif dpe in ("F", "G"):
        note -= 1.2
    note = float(np.clip(round(note, 1), 1.0, 9.8))

    # 4. Résumé synthétique & arguments
    atouts, points_vigilance = [], []
    if etat >= 4:
        atouts.append("rénové/sans travaux")
    if row.get("txt_jardin", 0):
        surf_ext = row.get("surface_ext_m2", 0)
        atouts.append(f"jardin ({int(surf_ext)} m²)" if surf_ext and surf_ext > 0 else "jardin")
    elif row.get("txt_terrasse", 0):
        atouts.append("terrasse")
    elif row.get("txt_balcon", 0):
        atouts.append("balcon")
    if row.get("txt_parking", 0):
        atouts.append("stationnement")
    if dpe in ("A", "B", "C"):
        atouts.append(f"bon DPE ({dpe})")

    if travaux >= 2:
        points_vigilance.append("travaux à prévoir")
    elif travaux == 1:
        points_vigilance.append("à rafraîchir")
    if dpe in ("F", "G"):
        points_vigilance.append(f"passoire thermique (DPE {dpe})")
    elif dpe == "E":
        points_vigilance.append("DPE E")
    if row.get("txt_vendu_loue", 0):
        points_vigilance.append("vendu loué")

    txt_atouts = ", ".join(atouts[:3]) if atouts else "prestations standards"
    txt_vig = f" | Vigilance : {', '.join(points_vigilance)}" if points_vigilance else ""
    resume = f"Atouts : {txt_atouts}{txt_vig}"

    return {
        "llm_etat_score": int(etat),
        "llm_travaux_score": int(travaux),
        "llm_note_offre": float(note),
        "llm_resume": resume,
    }


def enrichir_avec_ollama(
    df: pd.DataFrame,
    cache_path: str = "ollama_features_cache.json",
    utiliser_api_ollama: bool = True,
    model: str = "qwen2.5:0.5b",
    max_appels_ollama: int = 300,
) -> pd.DataFrame:
    """
    Enrichit le DataFrame avec 4 variables structurées issues de l'analyse du texte :
    - llm_etat_score (1 à 5)
    - llm_travaux_score (0 à 3)
    - llm_note_offre (note sur 10)
    - llm_resume (synthèse courte + points de négociation)

    Utilise un cache JSON local (ollama_features_cache.json). Si le serveur local Ollama
    est actif, interroge `qwen2.5:0.5b` en mode JSON structuré pour chaque annonce non
    encore analysée par Ollama.
    """
    df = df.copy()
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    ollama_dispo = False
    if utiliser_api_ollama:
        try:
            import ollama
            ollama.list()
            ollama_dispo = True
        except Exception:
            ollama_dispo = False

    appels = 0
    modifies = 0
    resultats = []

    for _, row in df.iterrows():
        ad_id = str(row.get("id", ""))
        entree_cache = cache.get(ad_id)

        # Si déjà généré par Ollama (ou si Ollama est éteint et qu'on a déjà le cache), on réutilise
        if entree_cache and (entree_cache.get("source") == "ollama" or not ollama_dispo):
            resultats.append(entree_cache)
            continue

        heur = calculer_heuristique_llm(row)
        heur["source"] = "heuristique"
        desc = str(row.get("description", "") or "").strip()

        if ollama_dispo and desc and appels < max_appels_ollama:
            try:
                import ollama
                prompt = (
                    "Tu es un expert immobilier. Analyse cette annonce et réponds UNIQUEMENT en JSON "
                    "avec ces 4 clés :\n"
                    '- "etat_score": entier de 1 (à rénover entièrement) à 5 (neuf/rénové parfait)\n'
                    '- "travaux_score": entier de 0 (aucun travaux) à 3 (gros travaux)\n'
                    '- "note_offre": note sur 10 de la qualité du bien (luminosité, extérieur, parking, DPE)\n'
                    '- "resume": 1 phrase très courte résumant les points forts et les travaux éventuels\n\n'
                    f"Annonce : {desc[:800]}"
                )
                resp = ollama.chat(
                    model=model,
                    format="json",
                    messages=[
                        {"role": "system", "content": "Tu réponds uniquement en JSON valide en français."},
                        {"role": "user", "content": prompt},
                    ],
                    options={"temperature": 0.1},
                )
                parsed = json.loads(resp["message"]["content"])
                etat_val = int(np.clip(int(parsed.get("etat_score", heur["llm_etat_score"])), 1, 5))
                trav_val = int(np.clip(int(parsed.get("travaux_score", heur["llm_travaux_score"])), 0, 3))
                note_val = float(np.clip(float(parsed.get("note_offre", heur["llm_note_offre"])), 1.0, 10.0))
                res_txt = str(parsed.get("resume") or heur["llm_resume"]).strip()[:140]
                heur = {
                    "llm_etat_score": etat_val,
                    "llm_travaux_score": trav_val,
                    "llm_note_offre": round(note_val, 1),
                    "llm_resume": res_txt,
                    "source": "ollama",
                }
                appels += 1
                if appels % 20 == 0:
                    print(f"Ollama ({model}) : {appels} annonces analysées...")
            except Exception:
                pass

        cache[ad_id] = heur
        modifies += 1
        resultats.append(heur)

    if modifies > 0:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    res_df = pd.DataFrame(resultats, index=df.index)
    for col in ["llm_etat_score", "llm_travaux_score", "llm_note_offre", "llm_resume"]:
        df[col] = res_df[col]
    return df


# --------------------------------------------------------------------------
# 3. Géographie & Ventes Notariées Réelles (DVF Poitiers - 86194)
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


def telecharger_ou_charger_dvf(dvf_path: str = "dvf_poitiers.csv", verbose: bool = True) -> pd.DataFrame | None:
    """
    Charge le fichier local des ventes notariées DVF de Poitiers (code INSEE 86194),
    ou le télécharge automatiquement depuis data.gouv.fr (Etalab geo-dvf 2023-2025).
    """
    if os.path.exists(dvf_path) and os.path.getsize(dvf_path) > 1000:
        try:
            return pd.read_csv(dvf_path, low_memory=False)
        except Exception:
            pass

    urls = [
        "https://files.data.gouv.fr/geo-dvf/latest/csv/2025/communes/86/86194.csv",
        "https://files.data.gouv.fr/geo-dvf/latest/csv/2024/communes/86/86194.csv",
        "https://files.data.gouv.fr/geo-dvf/latest/csv/2023/communes/86/86194.csv",
    ]
    dfs = []
    for u in urls:
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                sub = pd.read_csv(resp, low_memory=False)
                dfs.append(sub)
        except Exception:
            continue

    if dfs:
        dvf_all = pd.concat(dfs, ignore_index=True).drop_duplicates()
        dvf_all.to_csv(dvf_path, index=False)
        if verbose:
            print(f"Fichier DVF Notaires téléchargé ({len(dvf_all)} lignes) -> {dvf_path}")
        return dvf_all
    return None


def nettoyer_dvf(dvf: pd.DataFrame) -> pd.DataFrame:
    """Filtre les ventes notariées d'appartements et maisons à Poitiers sans ventes multi-lots complexes."""
    dvf = dvf[
        (dvf["nature_mutation"] == "Vente")
        & (dvf["type_local"].isin(["Appartement", "Maison"]))
    ].copy()
    # On garde les mutations à 1 ligne de bâti principal (évite les ventes d'immeubles entiers)
    dvf = dvf[dvf.groupby("id_mutation")["id_mutation"].transform("size") <= 2]
    dvf = dvf.dropna(subset=["valeur_fonciere", "surface_reelle_bati", "latitude", "longitude"])
    dvf = dvf[(dvf["surface_reelle_bati"] >= 14) & (dvf["surface_reelle_bati"] <= 400)]
    dvf["pm2"] = dvf["valeur_fonciere"] / dvf["surface_reelle_bati"]
    dvf = dvf[dvf["pm2"].between(700, 6500)].copy()
    return dvf


def prix_m2_voisinage(
    df: pd.DataFrame, dvf: pd.DataFrame, k: int = 25, rayon_m: int = 800
) -> pd.DataFrame:
    """
    Calcule pour chaque annonce :
    - `pm2_voisinage` : médiane du prix/m² des ventes réelles notariées (DVF) du MÊME TYPE
      (Appartement ou Maison) dans un rayon de `rayon_m` mètres.
    - `nb_ventes_dvf` : nombre de ventes notariées comparables trouvées autour du bien.
    """
    dvf_clean = nettoyer_dvf(dvf)
    out_pm2 = pd.Series(np.nan, index=df.index, name="pm2_voisinage")
    out_nb = pd.Series(0, index=df.index, name="nb_ventes_dvf")

    mapping_type = {"flat": "Appartement", "programme": "Appartement", "house": "Maison"}

    for code_annonce, type_dvf in mapping_type.items():
        sub_dvf = dvf_clean[dvf_clean["type_local"] == type_dvf]
        if len(sub_dvf) < 5:
            continue
        med_globale_type = float(sub_dvf["pm2"].median())

        mask_df = df["type_bien"] == code_annonce
        mask_gps = mask_df & df[["latitude", "longitude"]].notna().all(axis=1)

        if mask_gps.any():
            tree = BallTree(np.radians(sub_dvf[["latitude", "longitude"]].to_numpy()), metric="haversine")
            pts = np.radians(df.loc[mask_gps, ["latitude", "longitude"]].to_numpy())
            dist, idx = tree.query(pts, k=min(k, len(sub_dvf)))
            dans_rayon = (dist * 6_371_000) <= rayon_m
            pm2_voisins = np.where(dans_rayon, sub_dvf["pm2"].to_numpy()[idx], np.nan)

            with np.errstate(all="ignore"):
                med_locale = np.nanmedian(pm2_voisins, axis=1)
            nb_local = np.sum(dans_rayon, axis=1)

            med_locale = np.where(np.isnan(med_locale), med_globale_type, med_locale)
            out_pm2.loc[mask_gps] = np.round(med_locale, 0)
            out_nb.loc[mask_gps] = nb_local

        # Fallback médiane communale DVF pour les biens sans coordonnées GPS
        mask_sans_gps = mask_df & (~mask_gps)
        if mask_sans_gps.any():
            out_pm2.loc[mask_sans_gps] = round(med_globale_type, 0)

    return pd.DataFrame({"pm2_voisinage": out_pm2, "nb_ventes_dvf": out_nb})


def charger_et_preparer(csv_path: str = "annonces.csv", verbose: bool = True) -> pd.DataFrame:
    """Pipeline unique : Nettoyage -> Doublons -> Texte -> LLM/Heuristique -> Géo -> DVF Notaires."""
    df = pd.read_csv(csv_path)
    df = nettoyer_annonces(df, verbose=verbose)
    df = marquer_doublons(df, verbose=verbose)
    df = garder_un_doublon(df)
    df = features_texte(df)
    df = enrichir_avec_ollama(df, utiliser_api_ollama=True)
    df = features_geo(df)

    dvf = telecharger_ou_charger_dvf("dvf_poitiers.csv", verbose=verbose)
    if dvf is not None and len(dvf) > 0:
        dvf_feats = prix_m2_voisinage(df, dvf)
        df["pm2_voisinage"] = dvf_feats["pm2_voisinage"]
        df["nb_ventes_dvf"] = dvf_feats["nb_ventes_dvf"]
    return df


# --------------------------------------------------------------------------
# 4. Modèle, Prédictions, Comparaison DVF Notaires & Taux de Négociation
# --------------------------------------------------------------------------
ORDRE_DPE = {c: i for i, c in enumerate("ABCDEFG", start=1)}
CATEGORIELLES = ["quartier", "reseau", "type_bien"]


def construire_X_y(df: pd.DataFrame):
    df = df[df["prix"].notna() & (df["prix"] > 0)].copy()
    df["dpe_num"] = df["dpe"].map(ORDRE_DPE)
    df["ges_num"] = df["ges"].map(ORDRE_DPE)
    for c in CATEGORIELLES:
        if c not in df.columns:
            df[c] = "inconnu"
        else:
            df[c] = df[c].fillna("inconnu")

    numeriques = [
        "surface_m2", "pieces", "nbr_chambres", "etage", "annee_construction",
        "latitude", "longitude", "dist_centre_km", "dist_gare_km",
        "dpe_num", "ges_num", "txt_longueur", "surface_ext_m2", "charges_annuelles",
        "llm_etat_score", "llm_travaux_score", "llm_note_offre",
    ] + [c for c in df.columns if c.startswith("txt_") and c != "txt_longueur"]
    if "pm2_voisinage" in df.columns:
        numeriques.append("pm2_voisinage")

    numeriques = [c for c in numeriques if c in df.columns]
    X = df[numeriques + CATEGORIELLES].copy()
    X[numeriques] = X[numeriques].astype(float)
    return X, df["prix"].astype(float)


def construire_modele() -> TransformedTargetRegressor:
    cv_interne = KFold(n_splits=5, shuffle=True, random_state=0)
    prepa = ColumnTransformer(
        [("cat", TargetEncoder(target_type="continuous", cv=cv_interne), CATEGORIELLES)],
        remainder="passthrough",
    )
    modele = Pipeline([
        ("prepa", prepa),
        ("gbm", HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=8, random_state=0)),
    ])
    return TransformedTargetRegressor(regressor=modele, func=np.log, inverse_func=np.exp)


def estimer_negociation(df_res: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule pour chaque bien :
    - `prix_notaire_dvf` : valeur de marché basée sur les actes notariés DVF du voisinage
    - `ecart_vs_notaire_pct` : surcote/décote du prix affiché par rapport aux ventes notariées réelles
    - `taux_nego_estime_pct` : marge de négociation réaliste (entre 1.5% et 18%) tenant compte :
        1) du taux de base Poitiers (3.5% appartements, 5.5% maisons),
        2) de la surcote éventuelle vs IA et vs Notaires DVF,
        3) du malus DPE (passoire F/G : +3.5%, DPE E : +1.5%),
        4) du niveau de travaux détecté par le LLM (+1.5% à +4%).
    - `prix_cible_nego` : prix d'offre recommandé après négociation.
    """
    df_res = df_res.copy()

    if "pm2_voisinage" in df_res.columns and df_res["pm2_voisinage"].notna().any():
        df_res["pm2_notaire_dvf"] = df_res["pm2_voisinage"].fillna(df_res["pm2_predit"])
    else:
        # Si DVF non chargé, repli sur -6% sous le prix affiché médian (écart moyen affiché vs signé)
        df_res["pm2_notaire_dvf"] = np.round(df_res["pm2_predit"] * 0.94, 0)

    df_res["prix_notaire_dvf"] = np.round(df_res["pm2_notaire_dvf"] * df_res["surface_m2"], 0)
    df_res["ecart_vs_notaire_pct"] = np.round(
        (df_res["prix_reel"] - df_res["prix_notaire_dvf"]) / df_res["prix_notaire_dvf"] * 100, 1
    )

    # Taux de négociation de base selon le type de bien à Poitiers (observatoire 2024-2026)
    base_nego = np.where(df_res["type_label"] == "Maison", 5.5, 3.5)

    # Ajustement selon la surcote par rapport à l'estimation IA et au notaire DVF
    surcote_ia = np.clip(df_res["ecart_pct"], -5.0, 25.0) * 0.25
    surcote_dvf = np.clip(df_res["ecart_vs_notaire_pct"], -5.0, 25.0) * 0.15

    # Ajustement énergétique DPE (levier majeur de négociation)
    dpe_s = df_res["dpe"].fillna("").astype(str).str.upper()
    bonus_dpe = np.where(dpe_s.isin(["F", "G"]), 3.5, np.where(dpe_s == "E", 1.5, 0.0))

    # Ajustement travaux (score LLM 0 à 3)
    trav_s = df_res.get("llm_travaux_score", pd.Series(0, index=df_res.index)).fillna(0)
    bonus_travaux = trav_s * 1.3

    taux_nego = np.clip(base_nego + surcote_ia + surcote_dvf + bonus_dpe + bonus_travaux, 1.5, 18.0)
    df_res["taux_nego_estime_pct"] = np.round(taux_nego, 1)
    df_res["prix_cible_nego"] = np.round(df_res["prix_reel"] * (1.0 - df_res["taux_nego_estime_pct"] / 100.0), 0)
    df_res["economie_nego_euro"] = np.round(df_res["prix_reel"] - df_res["prix_cible_nego"], 0)
    return df_res


def generer_predictions_cv(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Calcule les prédictions hors-échantillon (CV 5 plis), les écarts DVF et la marge de négociation."""
    X, y = construire_X_y(df)
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
    X, y = construire_X_y(df)
    modele = construire_modele()
    modele.fit(X, y)
    perm = permutation_importance(
        modele, X, y, n_repeats=10, random_state=0, scoring="neg_mean_absolute_error"
    )
    imp = pd.DataFrame({
        "feature": X.columns,
        "label": [LABELS_FEATURES.get(c, c) for c in X.columns],
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

    # Indicateurs Notaires DVF & Négociation
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


def entrainer_modele(df: pd.DataFrame) -> TransformedTargetRegressor:
    X, y = construire_X_y(df)
    modele = construire_modele()
    modele.fit(X, y)
    return modele


def predire_bien(modele: TransformedTargetRegressor, bien: dict) -> float:
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
        df_bien["quartier"] = df_bien["quartier"].replace(QUARTIERS_HARMONISES)
    if "type_bien" in df_bien.columns:
        df_bien["type_bien"] = df_bien["type_bien"].replace(TYPES_BIEN_HARMONISES)
    df_bien = features_texte(df_bien)
    df_bien = enrichir_avec_ollama(df_bien, utiliser_api_ollama=False)
    df_bien = features_geo(df_bien)
    if df_bien["pm2_voisinage"].isna().all() and os.path.exists("dvf_poitiers.csv"):
        dvf = pd.read_csv("dvf_poitiers.csv", low_memory=False)
        df_bien["pm2_voisinage"] = prix_m2_voisinage(df_bien, dvf)["pm2_voisinage"]
    X_bien, _ = construire_X_y(df_bien)
    return float(modele.predict(X_bien)[0])


if __name__ == "__main__":
    df_propre = charger_et_preparer("annonces.csv")
    evaluer(df_propre)
