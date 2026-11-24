"""
Script 1/3 : Feature Engineering — IA Immobilier Poitiers.

Responsabilité unique :
- Lit les tables brutes (`offres_historique_brut`, `offres_recentes_brut`).
- Applique le nettoyage, le dédoublonnage, les regex texte, l'enrichissement LLM (Ollama),
  les distances GPS et le prix/m² des ventes notariées réelles (DVF Poitiers).
- Génère et sauvegarde les tables APRÈS Feature Engineering :
  • `offres_historique_features` (`csv/annonces_historique_features.csv`)
  • `offres_recentes_features`   (`csv/annonces_recentes_features.csv`)

Usage :
    python FeatureEngineering.py
"""

import json
import os
import re
import urllib.request
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

CHEMIN_DVF_CSV = os.path.join("csv", "dvf_grand_poitiers.csv")
CHEMIN_DVF_POITIERS_LEGACY = os.path.join("csv", "dvf_poitiers.csv")

# Codes INSEE des communes de Grand Poitiers
CODES_INSEE_GRAND_POITIERS = [
    "86194",  # Poitiers
    "86041",  # Buxerolles
    "86214",  # Saint-Benoît
    "86158",  # Migné-Auxances
    "86062",  # Chasseneuil-du-Poitou
    "86115",  # Jaunay-Marigny
    "86027",  # Biard
    "86297",  # Vouneuil-sous-Biard
    "86100",  # Fontaine-le-Comte
    "86163",  # Montamisé
    "86263",  # Smarves
    "86133",  # Ligugé
]

# Repères géographiques de Grand Poitiers
CENTRE = (46.5802, 0.3404)       # Hôtel de Ville de Poitiers
GARE = (46.5826, 0.3333)         # Gare de Poitiers
CAMPUS = (46.5695, 0.3827)       # Campus Universitaire (Beaulieu)
CHU = (46.5567, 0.3765)          # CHU de la Milétrie
FUTUROSCOPE = (46.6667, 0.3667)  # Technopole du Futuroscope

# Périmètres QPV (Quartiers Prioritaires de la Politique de la Ville - ANCT Grand Poitiers)
# (centre lat, lon, rayon approximatif en km)
QPV_GRAND_POITIERS = [
    ("Couronneries", 46.5935, 0.3555, 0.65),
    ("Trois Cités", 46.5638, 0.3332, 0.55),
    ("Bel Air", 46.5875, 0.3125, 0.45),
    ("Beaulieu", 46.5720, 0.3705, 0.55),
    ("Saint-Éloi", 46.5890, 0.3740, 0.50),
]

# Référentiel socio-économique & parc immobilier Grand Poitiers (INSEE FiLoSoFi / RP / RPLS)
# Colonnes : (revenu_median_uc_euro, taux_hlm_pct, part_proprietaires_pct, taux_vacance_pct)
STATS_COMMUNES_GRAND_POITIERS = {
    "86000": (22350, 26.5, 34.0, 9.8),   # Poitiers (moyenne)
    "86180": (24800, 17.2, 62.0, 5.4),   # Buxerolles
    "86280": (26900, 9.5, 67.5, 5.1),    # Saint-Benoît
    "86440": (25100, 8.2, 72.0, 4.9),    # Migné-Auxances
    "86360": (24400, 11.4, 58.0, 6.2),   # Chasseneuil-du-Poitou / Montamisé
    "86130": (24600, 7.8, 70.5, 5.5),    # Jaunay-Marigny
    "86580": (26200, 6.5, 71.0, 4.8),    # Biard / Vouneuil-sous-Biard
    "86240": (25500, 6.0, 74.0, 4.6),    # Fontaine-le-Comte / Smarves / Ligugé
}

# Affinage par quartier à Poitiers (86000) selon les IRIS INSEE FiLoSoFi / RPLS
STATS_QUARTIERS_POITIERS = {
    "Centre-ville": (24900, 8.5, 29.0, 11.5),
    "Saint-Eloi - Breuil - Mingot": (21200, 32.0, 44.0, 8.0),
    "Couronneries": (16200, 56.0, 22.0, 10.5),
    "Trois Cités": (16500, 54.0, 24.0, 10.0),
    "Beaulieu": (18400, 42.0, 28.0, 9.5),
    "Ouest": (21800, 24.0, 46.0, 8.5),
    "Montbernage - Pont Neuf": (23500, 14.0, 45.0, 8.0),
    "Gibauderie": (24200, 16.0, 52.0, 7.5),
    "Poitiers": (22350, 26.5, 34.0, 9.8),
}

LABELS_FEATURES = {
    "surface_m2": "Surface habitable (m²)",
    "log_surface": "Logarithme de la surface",
    "sqrt_surface": "Racine de la surface",
    "inv_surface": "Inverse surface (1 / m²)",
    "surface_par_piece": "Surface moyenne par pièce (m²/p)",
    "ratio_chambres": "Ratio Chambres / Pièces",
    "surface_relative_marche": "Surface relative au marché local",
    "pm2_voisinage": "Prix/m² Notaire DVF voisinage (800m)",
    "pm2_dvf_pondere_dist": "Prix/m² DVF pondéré par la distance",
    "pm2_dvf_taille_proche": "Prix/m² DVF biens de taille proche",
    "pm2_dvf_tranche_surface": "Prix/m² DVF par tranche de surface",
    "pm2_dvf_dispersion": "Dispersion €/m² DVF voisinage",
    "nb_ventes_dvf": "Nb ventes Notaires DVF (800m)",
    "tendance_dvf_pct": "Tendance prix DVF secteur (%)",
    "valeur_dvf_ajustee_taille": "Valeur DVF ajustée par taille (€)",
    "valeur_dvf_nette_travaux": "Valeur DVF nette de travaux (€)",
    "cout_travaux_estime_euro": "Coût travaux estimé (€)",
    "cout_travaux_m2": "Coût travaux estimé (€/m²)",
    "trav_rafraichissement": "Poste travaux : Rafraîchissement",
    "trav_cuisine_sdb": "Poste travaux : Cuisine / SDB",
    "trav_menuiseries": "Poste travaux : Menuiseries",
    "trav_elec_plomberie": "Poste travaux : Élec. / Plomberie",
    "trav_gros_oeuvre": "Poste travaux : Gros œuvre / Isol.",
    "type_bien": "Type de bien (Maison / Appart.)",
    "quartier": "Quartier / Commune",
    "dist_centre_km": "Distance Centre-ville (km)",
    "dist_gare_km": "Distance Gare (km)",
    "dist_campus_km": "Distance Campus Universitaire (km)",
    "dist_chu_km": "Distance CHU Milétrie (km)",
    "dist_futuroscope_km": "Distance Futuroscope (km)",
    "est_en_qpv": "Situé en zone QPV (0/1)",
    "dist_qpv_km": "Distance au QPV le plus proche (km)",
    "revenu_median_secteur": "Revenu médian du secteur (€/an)",
    "taux_hlm_secteur": "Taux de logements HLM secteur (%)",
    "part_proprietaires_secteur": "Part propriétaires occupants (%)",
    "taux_vacance_secteur": "Taux de logements vacants (%)",
    "dpe_num": "Note DPE (A à G)",
    "annee_construction": "Année de construction",
    "age_bien": "Ancienneté du bien (années)",
    "ges_num": "Note GES (A à G)",
    "nbr_chambres": "Nombre de chambres",
    "pieces": "Nombre de pièces",
    "etage": "Étage",
    "est_rdc": "Rez-de-chaussée (0/1)",
    "etage_eleve_sans_ascenseur": "Étage ≥ 3 sans ascenseur (0/1)",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "txt_longueur": "Longueur du descriptif",
    "reseau": "Agence / Réseau",
    "surface_ext_m2": "Surface terrain/jardin extraite (m²)",
    "charges_annuelles": "Charges annuelles extraites (€)",
    "llm_etat_score": "Note État du bien (LLM 1-5)",
    "llm_travaux_score": "Niveau de travaux (LLM 0-3)",
    "llm_note_offre": "Note Attractivité (LLM /10)",
    "vis_nb_photos": "Nombre de photos de l'annonce",
    "vis_luminosite": "Luminosité moyenne des photos",
    "vis_contraste": "Contraste moyen des photos",
    "vis_nettete": "Netteté moyenne des photos",
    "vis_saturation": "Saturation couleurs photos",
    "vis_etat_vlm": "Note état intérieur VLM (1-5)",
    "iris_code": "Îlot IRIS exact (GPS)",
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
    "txt_investissement": "Mot-clé : Investissement / Étudiant",
    "txt_pac": "Mot-clé : Pompe à chaleur / Clim",
    "txt_piscine": "Mot-clé : Piscine",
    "txt_immeuble": "Mot-clé : Immeuble de rapport / Lots",
    "txt_standing": "Mot-clé : Standing / Architecte",
}

CHEMIN_IRIS_GEOJSON = os.path.join("csv", "iris_grand_poitiers.geojson")
ORDRE_DPE = {c: i for i, c in enumerate("ABCDEFG", start=1)}
CATEGORIELLES = ["quartier", "iris_code", "type_bien"]

# Colonnes générées uniquement lors de la prédiction/évaluation (ne doivent JAMAIS être dans les tables *_features ni dans X)
COLONNES_PREDICTION = [
    "prix_reel",
    "prix_predit",
    "ecart_euro",
    "erreur_abs_euro",
    "ecart_pct",
    "erreur_abs_pct",
    "pm2_reel",
    "pm2_predit",
    "quartier_clean",
    "type_label",
    "titre_propre",
    "pm2_notaire_dvf",
    "prix_notaire_dvf",
    "ecart_vs_notaire_pct",
    "taux_nego_estime_pct",
    "prix_cible_nego",
    "economie_nego_euro",
    "prix_total_projet",
    "valeur_estimee_renovee",
    "gain_net_apres_travaux",
    "decote_nette_pct",
    "score_pouvoir_nego",
    "tranche_prix",
]


def supprimer_colonnes_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Garantit qu'aucune colonne de prédiction antérieure ne reste dans une table de features."""
    cols_presentes = [c for c in COLONNES_PREDICTION if c in df.columns]
    return df.drop(columns=cols_presentes) if cols_presentes else df.copy()

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


def nettoyer_annonces(df: pd.DataFrame, verbose: bool = True, exclure_atypiques: bool = True) -> pd.DataFrame:
    """
    Nettoie les annonces :
    - Garde uniquement les biens de la Vienne (`86xxx`, évite les homonymes type Saint-Benoît 974)
    - Supprime les prix/surfaces manquants ou aberrants
    - Pour l'investissement résidentiel Grand Poitiers (`exclure_atypiques=True`), écarte
      les biens d'exception hors marché (> 480 000 € ou > 240 m² ou châteaux/manoirs)
    """
    df = df.copy()
    avant = len(df)
    df = df.dropna(subset=["prix", "surface_m2"])
    df = df[(df["prix"] > 0) & (df["surface_m2"] > 0)].copy()

    if "code_postal" in df.columns:
        cp_str = df["code_postal"].astype(str).str.strip()
        df = df[cp_str.str.startswith("86")].copy()

    pm2 = df["prix"] / df["surface_m2"]
    df = df[pm2.between(300, 12000)].copy()

    if exclure_atypiques:
        txt_td = (df["titre"].fillna("") + " " + df["description"].fillna("")).str.lower()
        est_chateau = txt_td.str.contains(r"\bch[âa]teau\b|\bmanoir\b|\bpropri[ée]t[ée]\s+d'exception\b", regex=True)
        est_viager_encheres = txt_td.str.contains(
            r"\bviager\b|\bbouquet\b|\bnue[-\s]?propri[ée]t[ée]\b|\busufruit\b|\brente\s+viag[èe]re\b|\bench[èe]res\b|\bvente\s+[àa]\s+terme\b",
            regex=True,
        )
        pm2_cur = df["prix"] / df["surface_m2"]
        df = df[(df["prix"] <= 480_000) & (df["surface_m2"] <= 240) & (~est_chateau) & (~est_viager_encheres) & (pm2_cur >= 950)].copy()

    suppr = avant - len(df)
    if verbose and suppr:
        print(f"{suppr} annonces supprimées (hors 86, viager/enchères, < 950 €/m² ou luxe > 480k€ / > 240m²).")

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
    "investissement": r"investissement|locatif|rendement|rentabilit[ée]|colocation|[ée]tudiant|campus",
    "pac": r"pompe\s+[àa]\s+chaleur|\bpac\b|climatisation\s+r[ée]versible",
    "piscine": r"piscine",
    "immeuble": r"immeuble\s+de\s+rapport|immeuble\s+entier|monopropri[ée]t[ée]|plusieurs\s+lots",
    "standing": r"architecte|maison\s+de\s+ma[îi]tre|contemporaine|haut\s+de\s+gamme|d[ée]pendances",
}


def extraire_chiffres_texte(desc_serie: pd.Series) -> pd.DataFrame:
    """Extrait par regex la surface extérieure (terrain/jardin) et les charges annuelles."""
    s = desc_serie.fillna("").str.lower()

    terrain = s.str.extract(
        r"(?:terrain|jardin(?:\s+privatif)?|parcelle)\s+(?:clos\s+|arbor[ée]\s+)?(?:de\s+|d'environ\s+)?(\d{2,5})\s*m[²2]",
        expand=False,
    )
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
    df["dist_campus_km"] = haversine_km(df["latitude"], df["longitude"], *CAMPUS)
    df["dist_chu_km"] = haversine_km(df["latitude"], df["longitude"], *CHU)
    df["dist_futuroscope_km"] = haversine_km(df["latitude"], df["longitude"], *FUTUROSCOPE)
    return df


def features_open_data_grand_poitiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rattache par GPS et code postal/quartier les indicateurs Open Data Grand Poitiers :
    - QPV (Couronneries, Trois Cités, Bel Air, Beaulieu, Saint-Éloi) : `est_en_qpv`, `dist_qpv_km`
    - INSEE FiLoSoFi / RPLS / RP : `revenu_median_secteur`, `taux_hlm_secteur`,
      `part_proprietaires_secteur`, `taux_vacance_secteur`
    """
    df = df.copy()

    # 1. Distance au QPV le plus proche & appartenance QPV
    if len(df) > 0:
        dists_qpv = np.column_stack([
            haversine_km(df["latitude"], df["longitude"], lat_q, lon_q)
            for _, lat_q, lon_q, _ in QPV_GRAND_POITIERS
        ])
        rayons_qpv = np.array([r_q for _, _, _, r_q in QPV_GRAND_POITIERS])
        with np.errstate(all="ignore"):
            df["dist_qpv_km"] = np.round(np.nanmin(dists_qpv, axis=1), 3)
            en_qpv = np.nanmin(dists_qpv - rayons_qpv, axis=1) <= 0
        df["est_en_qpv"] = np.where(df["latitude"].notna() & en_qpv, 1, 0)
    else:
        df["dist_qpv_km"] = np.nan
        df["est_en_qpv"] = 0

    # 2. Revenus médians, HLM, propriétaires & vacance (INSEE FiLoSoFi / RPLS / RP)
    rev_l, hlm_l, prop_l, vac_l = [], [], [], []
    for _, row in df.iterrows():
        q = str(row.get("quartier") or "")
        cp = str(row.get("code_postal") or "")[:5]
        if q in STATS_QUARTIERS_POITIERS:
            stats = STATS_QUARTIERS_POITIERS[q]
        elif cp in STATS_COMMUNES_GRAND_POITIERS:
            stats = STATS_COMMUNES_GRAND_POITIERS[cp]
        else:
            stats = STATS_COMMUNES_GRAND_POITIERS["86000"]

        # Ajustement local si situé dans un QPV sans quartier renseigné
        if row.get("est_en_qpv", 0) == 1 and q not in STATS_QUARTIERS_POITIERS:
            stats = (16800, 48.0, 26.0, 10.0)

        rev_l.append(stats[0])
        hlm_l.append(stats[1])
        prop_l.append(stats[2])
        vac_l.append(stats[3])

    df["revenu_median_secteur"] = rev_l
    df["taux_hlm_secteur"] = hlm_l
    df["part_proprietaires_secteur"] = prop_l
    df["taux_vacance_secteur"] = vac_l

    # 3. Mobilité (arrêts Vitalis / gares) & Commerces de proximité (cache OpenStreetMap Grand Poitiers)
    lat_arr = pd.to_numeric(df["latitude"], errors="coerce").to_numpy()
    lon_arr = pd.to_numeric(df["longitude"], errors="coerce").to_numpy()

    def _calc_poi(csv_path: str, rayon_km: float, filtre_col: str | None = None, filtre_vals: set[str] | None = None):
        if not os.path.exists(csv_path) or len(df) == 0:
            return np.full(len(df), np.nan), np.zeros(len(df), dtype=int)
        poi = pd.read_csv(csv_path)
        if filtre_col and filtre_vals and filtre_col in poi.columns:
            poi = poi[poi[filtre_col].isin(filtre_vals)]
        poi = poi.dropna(subset=["latitude", "longitude"])
        if poi.empty:
            return np.full(len(df), np.nan), np.zeros(len(df), dtype=int)
        plat = poi["latitude"].to_numpy(dtype=float)
        plon = poi["longitude"].to_numpy(dtype=float)
        dist_min_m = np.full(len(df), np.nan)
        nb_dans_rayon = np.zeros(len(df), dtype=int)
        for i in range(len(df)):
            if np.isnan(lat_arr[i]) or np.isnan(lon_arr[i]):
                continue
            d_km = haversine_km(lat_arr[i], lon_arr[i], plat, plon)
            dist_min_m[i] = round(float(np.min(d_km) * 1000.0), 0)
            nb_dans_rayon[i] = int(np.sum(d_km <= rayon_km))
        return dist_min_m, nb_dans_rayon

    d_bus, nb_bus = _calc_poi(os.path.join("csv", "transports_grand_poitiers.csv"), 0.40)
    df["dist_arret_bus_m"] = d_bus
    df["nb_arrets_bus_400m"] = nb_bus

    d_alim, _ = _calc_poi(
        os.path.join("csv", "commerces_grand_poitiers.csv"),
        0.50,
        filtre_col="categorie",
        filtre_vals={"supermarket", "convenience", "bakery", "butcher", "greengrocer"},
    )
    _, nb_comm = _calc_poi(os.path.join("csv", "commerces_grand_poitiers.csv"), 0.50)
    df["dist_commerce_alim_m"] = d_alim
    df["nb_commerces_500m"] = nb_comm
    df["score_commodites_500m"] = np.round(np.clip(nb_comm * 1.2 + nb_bus * 0.8, 0, 20), 1)
    return df


def features_surface_et_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Capture l'effet dégressif non linéaire du prix/m² selon la surface et
    ajoute des ratios structurels clés.
    """
    df = df.copy()
    surf = pd.to_numeric(df["surface_m2"], errors="coerce").astype(float).clip(lower=9.0)
    pieces = pd.to_numeric(df["pieces"], errors="coerce").astype(float).clip(lower=1.0)
    chambres = pd.to_numeric(df["nbr_chambres"], errors="coerce").astype(float)
    etage = pd.to_numeric(df["etage"], errors="coerce").astype(float)
    annee = pd.to_numeric(df["annee_construction"], errors="coerce").astype(float)

    df["log_surface"] = np.round(np.log(surf), 4)
    df["sqrt_surface"] = np.round(np.sqrt(surf), 3)
    df["inv_surface"] = np.round(1.0 / surf, 5)
    df["surface_par_piece"] = np.round(surf / pieces.fillna(2.5), 1)
    df["ratio_chambres"] = np.round(chambres.fillna(1.0) / pieces.fillna(2.5), 2)
    df["age_bien"] = np.where(np.isfinite(annee), 2026.0 - annee, np.nan)
    df["est_rdc"] = np.where(etage == 0.0, 1, 0)
    asc = pd.to_numeric(df.get("txt_ascenseur", pd.Series(0, index=df.index)), errors="coerce").fillna(0.0).astype(float)
    df["etage_eleve_sans_ascenseur"] = np.where((etage >= 3.0) & (asc == 0.0), 1, 0)

    # Surface relative par rapport au marché du même type de bien
    med_par_type = df.groupby("type_bien")["surface_m2"].transform("median")
    df["surface_relative_marche"] = np.round(surf / med_par_type.replace(0, np.nan), 3)

    # Loi Carrez vs surface utile / au sol
    desc_l = df["description"].fillna("").astype(str).str.lower()
    m_carrez = desc_l.str.extract(r"(?:loi\s+)?carrez\s*[:de]*\s*(\d{2,3}(?:[.,]\d{1,2})?)\s*m", expand=False)
    m_utile = desc_l.str.extract(r"(?:surface\s+utile|au\s+sol)\s*[:de]*\s*(\d{2,3}(?:[.,]\d{1,2})?)\s*m", expand=False)
    surf_carrez = pd.to_numeric(m_carrez.str.replace(",", ".", regex=False), errors="coerce")
    surf_utile = pd.to_numeric(m_utile.str.replace(",", ".", regex=False), errors="coerce")
    est_mention_carrez = desc_l.str.contains(r"\bcarrez\b", regex=True).astype(int)

    df["est_mention_carrez"] = est_mention_carrez
    df["surface_carrez_m2"] = np.where(
        surf_carrez.notna() & (surf_carrez >= 9.0),
        surf_carrez,
        np.where((df["type_bien"].astype(str) == "flat") & (est_mention_carrez == 1), surf, np.nan),
    )
    df["surface_utile_m2"] = np.where(surf_utile.notna() & (surf_utile >= 9.0), surf_utile, surf)
    prix_num = pd.to_numeric(df["prix"], errors="coerce")
    df["prix_m2_carrez"] = np.round(prix_num / pd.to_numeric(df["surface_carrez_m2"], errors="coerce").replace(0, np.nan), 0)
    df["prix_m2_utile"] = np.round(prix_num / pd.to_numeric(df["surface_utile_m2"], errors="coerce").clip(lower=9.0), 0)

    # Flag communes peu représentées (Saint-Benoît, Biard, Vouneuil-sous-Biard, etc.)
    if "ville" in df.columns:
        ville_clean = df["ville"].fillna("Poitiers").astype(str).str.strip()
        counts_v = ville_clean.map(ville_clean.value_counts())
        df["commune_nb_annonces"] = counts_v.fillna(0).astype(int)
        df["commune_donnees_limitees"] = np.where(df["commune_nb_annonces"] < 15, 1, 0)
    else:
        df["commune_nb_annonces"] = len(df)
        df["commune_donnees_limitees"] = 0
    return df


def estimer_cout_travaux(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimateur de coût des travaux en 3 couches (utilisable directement comme feature dans X) :
    1. Détection par postes (Texte + DPE + Année + LLM) :
       - `trav_rafraichissement`, `trav_cuisine_sdb`, `trav_menuiseries`,
         `trav_elec_plomberie`, `trav_gros_oeuvre`
    2. Chiffrage paramétrique marché Poitiers/Vienne selon surface & type de bien
    3. Sorties numériques : `cout_travaux_min_euro`, `cout_travaux_max_euro`,
       `cout_travaux_estime_euro`, `cout_travaux_m2`, `postes_travaux`
    """
    df = df.copy()
    desc = df["description"].fillna("").str.lower()
    dpe = df["dpe"].fillna("").astype(str).str.upper()
    surf = pd.to_numeric(df["surface_m2"], errors="coerce").fillna(60.0)
    est_maison = df["type_bien"].astype(str) == "house"
    llm_trav = pd.to_numeric(df.get("llm_travaux_score", 0), errors="coerce").fillna(0)

    # Couche 1 : Détection des postes
    t_raf = (
        desc.str.contains(r"rafra[iî]chir|rafra[iî]chissement|quelques\s+travaux|peinture", regex=True)
        | (llm_trav == 1)
    ).astype(int)
    t_cui = desc.str.contains(
        r"cuisine\s+[àa]\s+(?:cr[ée]er|am[ée]nager|refaire)|salle\s+de\s+bain\s+[àa]\s+refaire",
        regex=True,
    ).astype(int)
    t_men = desc.str.contains(r"simple\s+vitrage|menuiserie[s]?\s+[àa]\s+changer", regex=True).astype(int)
    t_el = desc.str.contains(
        r"[ée]lectricit[ée]\s+[àa]\s+refaire|remettre\s+aux\s+normes|mise\s+aux\s+normes|plomberie",
        regex=True,
    ).astype(int)
    t_gros = (
        desc.str.contains(
            r"enti[èe]rement\s+[àa]\s+r[ée]nover|r[ée]novation\s+compl[èe]te|gros\s+travaux|toiture|charpente",
            regex=True,
        )
        | (llm_trav >= 3)
    ).astype(int)
    t_isol = dpe.isin(["F", "G"]).astype(int)
    t_toit_fac = desc.str.contains(r"toiture|charpente|ravalement|fa[çc]ade\s+[àa]\s+refaire", regex=True).astype(int)
    t_chauf = desc.str.contains(r"chaudi[èe]re\s+[àa]\s+changer|isolation|combles|humidit[ée]", regex=True).astype(int)
    annee_c = pd.to_numeric(df.get("annee_construction"), errors="coerce")

    # Si l'annonce dit "à rénover" / llm_trav >= 2 sans préciser le poste, on active rafraîchissement + élec/cuisine
    gen_renov = (
        desc.str.contains(r"[àa]\s+r[ée]nover|r[ée]novation\s+[àa]\s+pr[ée]voir", regex=True)
        | (llm_trav >= 2)
    )
    t_raf = np.where(gen_renov, 1, t_raf)

    # Couche 2 : Barème €/m² + forfaits Poitiers/Vienne
    cout = np.zeros(len(df), dtype=float)
    cout = np.where(t_raf == 1, cout + surf * 220.0, cout)
    cout = np.where(t_cui == 1, cout + 11000.0, cout)
    cout = np.where(t_men == 1, cout + surf * 130.0, cout)
    cout = np.where(t_el == 1, cout + surf * 140.0, cout)
    cout = np.where(t_isol == 1, cout + np.where(est_maison, surf * 240.0, surf * 140.0), cout)
    cout = np.where(t_toit_fac == 1, cout + np.where(est_maison, surf * 280.0, 5500.0), cout)
    cout = np.where(t_chauf == 1, cout + 7500.0, cout)
    # Règle d'ancienneté : bien d'avant 1975 classé E/F/G sans mention de rénovation
    ancien_passoire = (annee_c.notna()) & (annee_c < 1975) & dpe.isin(["E", "F", "G"]) & (cout < surf * 250.0)
    cout = np.where(ancien_passoire, np.maximum(cout, surf * 320.0), cout)
    # Rénovation lourde : remplace les petits postes pour éviter le double comptage
    cout = np.where(t_gros == 1, np.maximum(cout, surf * 1050.0), cout)
    # Si llm_trav == 2 sans gros œuvre, plancher à 420 €/m²
    cout = np.where((llm_trav == 2) & (t_gros == 0), np.maximum(cout, surf * 420.0), cout)
    # DPE E : léger poste énergétique
    cout = np.where((dpe == "E") & (cout == 0), surf * 80.0, cout)
    # Embeddings photos OpenCLIP : détection visuelle de travaux non mentionnés dans le texte
    clip_trav = pd.to_numeric(df.get("clip_score_travaux"), errors="coerce").fillna(0.0)
    clip_detect = (clip_trav >= 0.65) & (cout < surf * 280.0)
    cout = np.where(clip_detect, np.maximum(cout, surf * 280.0), cout)

    df["trav_rafraichissement"] = t_raf
    df["trav_cuisine_sdb"] = t_cui
    df["trav_menuiseries"] = t_men
    df["trav_elec_plomberie"] = t_el
    df["trav_gros_oeuvre"] = np.maximum(t_gros, t_isol)

    df["cout_travaux_estime_euro"] = np.round(cout, 0)
    df["cout_travaux_min_euro"] = np.round(cout * 0.78, 0)
    df["cout_travaux_max_euro"] = np.round(cout * 1.25, 0)
    df["cout_travaux_m2"] = np.round(cout / surf.clip(lower=1.0), 0)

    t_gros_arr = np.asarray(t_gros)
    t_raf_arr = np.asarray(t_raf)
    t_cui_arr = np.asarray(t_cui)
    t_men_arr = np.asarray(t_men)
    t_el_arr = np.asarray(t_el)
    t_isol_arr = np.asarray(t_isol)
    dpe_arr = dpe.to_numpy()

    libelles = []
    for i in range(len(df)):
        p = []
        if t_gros_arr[i]:
            p.append("Rénovation lourde")
        else:
            if t_raf_arr[i]:
                p.append("Rafraîchissement")
            if t_cui_arr[i]:
                p.append("Cuisine/SDB")
            if t_men_arr[i]:
                p.append("Menuiseries")
            if t_el_arr[i]:
                p.append("Élec./Normes")
        if t_isol_arr[i]:
            p.append(f"Isolation DPE {dpe_arr[i]}")
        elif dpe_arr[i] == "E" and not p:
            p.append("Optimisation DPE E")
        if bool(clip_detect.iloc[i]) if hasattr(clip_detect, "iloc") else bool(clip_detect[i]):
            p.append("Travaux visibles (CLIP photos)")
        libelles.append(" + ".join(p) if p else "Aucun travaux majeur")
    df["postes_travaux"] = libelles
    return df


def features_marche_historique(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule la durée en ligne et les baisses de prix pour le pouvoir de négociation."""
    df = df.copy()
    maintenant = pd.Timestamp.now(tz="UTC")
    if "date_premiere_vue" in df.columns:
        d_prem = pd.to_datetime(df["date_premiere_vue"], utc=True, errors="coerce")
        duree = (maintenant - d_prem).dt.total_seconds() / 86400.0
        df["duree_en_ligne_jours"] = np.round(duree.fillna(0.0).clip(lower=0.0), 1)
    else:
        df["duree_en_ligne_jours"] = 0.0

    if "evolution_prix_euro" in df.columns:
        evol = pd.to_numeric(df["evolution_prix_euro"], errors="coerce").fillna(0.0)
        prix_init = pd.to_numeric(df.get("prix_initial", df["prix"]), errors="coerce").replace(0, np.nan)
        df["a_baisse_prix"] = (evol < 0).astype(int)
        df["amplitude_baisse_pct"] = np.round(np.where(evol < 0, (-evol / prix_init) * 100.0, 0.0), 1)
    else:
        df["a_baisse_prix"] = 0
        df["amplitude_baisse_pct"] = 0.0
    return df


def telecharger_ou_charger_dvf(dvf_path: str = CHEMIN_DVF_CSV, verbose: bool = True) -> pd.DataFrame | None:
    """
    Charge le fichier local des ventes notariées DVF de Grand Poitiers (12 communes),
    ou le télécharge automatiquement depuis data.gouv.fr (Etalab geo-dvf 2023-2025).
    """
    os.makedirs(os.path.dirname(dvf_path) or "csv", exist_ok=True)
    if os.path.exists(dvf_path) and os.path.getsize(dvf_path) > 1000:
        try:
            return pd.read_csv(dvf_path, low_memory=False)
        except Exception:
            pass

    dfs = []
    for code_insee in CODES_INSEE_GRAND_POITIERS:
        for annee in ("2025", "2024", "2023"):
            u = f"https://files.data.gouv.fr/geo-dvf/latest/csv/{annee}/communes/86/{code_insee}.csv"
            try:
                req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=6) as resp:
                    sub = pd.read_csv(resp, low_memory=False)
                    dfs.append(sub)
            except Exception:
                continue

    if dfs:
        dvf_all = pd.concat(dfs, ignore_index=True).drop_duplicates()
        dvf_all.to_csv(dvf_path, index=False)
        if verbose:
            print(f"Fichier DVF Grand Poitiers téléchargé ({len(dvf_all)} lignes) -> {dvf_path}")
        return dvf_all

    # Repli local si hors-ligne sur l'ancien fichier Poitiers
    if os.path.exists(CHEMIN_DVF_POITIERS_LEGACY):
        return pd.read_csv(CHEMIN_DVF_POITIERS_LEGACY, low_memory=False)
    return None


def nettoyer_dvf(dvf: pd.DataFrame) -> pd.DataFrame:
    """Filtre les ventes notariées d'appartements et maisons sur Grand Poitiers."""
    dvf = dvf[
        (dvf["nature_mutation"] == "Vente")
        & (dvf["type_local"].isin(["Appartement", "Maison"]))
    ].copy()
    dvf = dvf[dvf.groupby("id_mutation")["id_mutation"].transform("size") <= 2]
    dvf = dvf.dropna(subset=["valeur_fonciere", "surface_reelle_bati", "latitude", "longitude"])
    dvf = dvf[(dvf["surface_reelle_bati"] >= 14) & (dvf["surface_reelle_bati"] <= 400)]
    dvf["pm2"] = dvf["valeur_fonciere"] / dvf["surface_reelle_bati"]
    dvf = dvf[dvf["pm2"].between(700, 6500)].copy()
    if "date_mutation" in dvf.columns:
        dvf["annee_mut"] = pd.to_datetime(dvf["date_mutation"], errors="coerce").dt.year
    else:
        dvf["annee_mut"] = np.nan
    return dvf


def _tranche_surface(s: pd.Series) -> pd.Series:
    return pd.cut(
        s,
        bins=[0, 35, 55, 80, 110, 1000],
        labels=["T1_lt35", "T2_35_55", "T3_55_80", "T4_80_110", "T5_gt110"],
    ).astype(str)


def prix_m2_voisinage(
    df: pd.DataFrame, dvf: pd.DataFrame, k: int = 40, rayon_m: int = 800
) -> pd.DataFrame:
    """
    Exploitation avancée des ventes notariées DVF :
    - `pm2_voisinage`           : médiane €/m² des ventes comparables dans les 800 m
    - `nb_ventes_dvf`           : nombre de ventes notariées comparables dans les 800 m
    - `pm2_dvf_pondere_dist`    : moyenne pondérée par 1 / (distance_m + 50)
    - `pm2_dvf_taille_proche`   : médiane €/m² des ventes de surface similaire (±30%)
    - `pm2_dvf_tranche_surface` : médiane €/m² DVF de la même tranche de surface (dégressivité réelle)
    - `pm2_dvf_dispersion`      : écart-type €/m² dans le voisinage (hétérogénéité du secteur)
    - `tendance_dvf_pct`        : évolution des prix/m² entre anciennes années et année récente
    """
    dvf_clean = nettoyer_dvf(dvf)
    dvf_clean["tranche_s"] = _tranche_surface(dvf_clean["surface_reelle_bati"])
    df_tranches = _tranche_surface(pd.to_numeric(df["surface_m2"], errors="coerce").fillna(60.0))

    cols_out = {
        "pm2_voisinage": pd.Series(np.nan, index=df.index),
        "nb_ventes_dvf": pd.Series(0, index=df.index),
        "pm2_dvf_pondere_dist": pd.Series(np.nan, index=df.index),
        "pm2_dvf_taille_proche": pd.Series(np.nan, index=df.index),
        "pm2_dvf_tranche_surface": pd.Series(np.nan, index=df.index),
        "pm2_dvf_dispersion": pd.Series(np.nan, index=df.index),
        "tendance_dvf_pct": pd.Series(0.0, index=df.index),
    }

    mapping_type = {"flat": "Appartement", "programme": "Appartement", "house": "Maison"}

    for code_annonce, type_dvf in mapping_type.items():
        sub_dvf = dvf_clean[dvf_clean["type_local"] == type_dvf].reset_index(drop=True)
        if len(sub_dvf) < 5:
            continue
        med_globale_type = float(sub_dvf["pm2"].median())
        med_par_tranche = sub_dvf.groupby("tranche_s")["pm2"].median().to_dict()

        mask_df = df["type_bien"] == code_annonce
        if not mask_df.any():
            continue

        # Attribution de la médiane DVF par tranche de surface (même sans GPS)
        cols_out["pm2_dvf_tranche_surface"].loc[mask_df] = (
            df_tranches.loc[mask_df].map(med_par_tranche).fillna(med_globale_type).round(0)
        )

        mask_gps = mask_df & df[["latitude", "longitude"]].notna().all(axis=1)
        if mask_gps.any():
            tree = BallTree(np.radians(sub_dvf[["latitude", "longitude"]].to_numpy()), metric="haversine")
            pts = np.radians(df.loc[mask_gps, ["latitude", "longitude"]].to_numpy())
            dist, idx = tree.query(pts, k=min(k, len(sub_dvf)))
            dist_m = dist * 6_371_000.0
            dans_rayon = dist_m <= rayon_m
            dans_rayon_elargi = dist_m <= 1500.0

            pm2_arr = sub_dvf["pm2"].to_numpy()[idx]
            surf_dvf_arr = sub_dvf["surface_reelle_bati"].to_numpy()[idx]
            annee_dvf_arr = sub_dvf["annee_mut"].to_numpy()[idx]

            pm2_voisins = np.where(dans_rayon, pm2_arr, np.nan)
            with np.errstate(all="ignore"):
                med_locale = np.nanmedian(pm2_voisins, axis=1)
                std_locale = np.nanstd(pm2_voisins, axis=1)
            nb_local = np.sum(dans_rayon, axis=1)

            # Moyenne pondérée par 1 / (d + 50m)
            poids = np.where(dans_rayon, 1.0 / (dist_m + 50.0), 0.0)
            somme_poids = np.sum(poids, axis=1)
            pm2_pond = np.where(
                somme_poids > 0,
                np.sum(np.where(dans_rayon, pm2_arr * poids, 0.0), axis=1) / np.maximum(somme_poids, 1e-9),
                med_globale_type,
            )

            # Médiane sur biens de surface proche (±30%)
            surf_annonces = pd.to_numeric(df.loc[mask_gps, "surface_m2"], errors="coerce").to_numpy()[:, None]
            proche_taille = dans_rayon_elargi & (np.abs(surf_dvf_arr - surf_annonces) <= 0.30 * surf_annonces)
            pm2_taille = np.where(proche_taille, pm2_arr, np.nan)
            with np.errstate(all="ignore"):
                med_taille = np.nanmedian(pm2_taille, axis=1)

            # Tendance temporelle DVF dans les 1500m (dernière année vs années précédentes)
            max_an = np.nanmax(sub_dvf["annee_mut"]) if sub_dvf["annee_mut"].notna().any() else np.nan
            tend_arr = np.zeros(len(surf_annonces))
            if pd.notna(max_an):
                rec_mask = dans_rayon_elargi & (annee_dvf_arr >= max_an)
                anc_mask = dans_rayon_elargi & (annee_dvf_arr < max_an)
                with np.errstate(all="ignore"):
                    med_rec = np.nanmedian(np.where(rec_mask, pm2_arr, np.nan), axis=1)
                    med_anc = np.nanmedian(np.where(anc_mask, pm2_arr, np.nan), axis=1)
                valid_t = np.isfinite(med_rec) & np.isfinite(med_anc) & (med_anc > 0)
                tend_arr = np.where(valid_t, np.clip((med_rec - med_anc) / med_anc * 100.0, -25.0, 25.0), 0.0)

            med_locale = np.where(np.isnan(med_locale), med_globale_type, med_locale)
            med_taille = np.where(np.isnan(med_taille), med_locale, med_taille)
            std_locale = np.where(np.isnan(std_locale), 300.0, std_locale)

            cols_out["pm2_voisinage"].loc[mask_gps] = np.round(med_locale, 0)
            cols_out["nb_ventes_dvf"].loc[mask_gps] = nb_local
            cols_out["pm2_dvf_pondere_dist"].loc[mask_gps] = np.round(pm2_pond, 0)
            cols_out["pm2_dvf_taille_proche"].loc[mask_gps] = np.round(med_taille, 0)
            cols_out["pm2_dvf_dispersion"].loc[mask_gps] = np.round(std_locale, 0)
            cols_out["tendance_dvf_pct"].loc[mask_gps] = np.round(tend_arr, 1)

        mask_sans_gps = mask_df & (~mask_gps)
        if mask_sans_gps.any():
            cols_out["pm2_voisinage"].loc[mask_sans_gps] = round(med_globale_type, 0)
            cols_out["pm2_dvf_pondere_dist"].loc[mask_sans_gps] = round(med_globale_type, 0)
            cols_out["pm2_dvf_taille_proche"].loc[mask_sans_gps] = round(med_globale_type, 0)

    return pd.DataFrame(cols_out)


def features_iris_spatial_join(df: pd.DataFrame, geojson_path: str = CHEMIN_IRIS_GEOJSON) -> pd.DataFrame:
    """
    Jointure spatiale GPS (Point-in-Polygon) sur les 67 polygones GeoJSON officiels
    des îlots IRIS de Grand Poitiers :
    - Attribue `iris_code` et `iris_name` à chaque annonce selon ses coordonnées `(latitude, longitude)`.
    """
    df = df.copy()
    df["iris_code"] = "inconnu"
    df["iris_name"] = "Inconnu"
    if not os.path.exists(geojson_path) or df.empty:
        return df

    try:
        from matplotlib.path import Path as MplPath
        with open(geojson_path, "r", encoding="utf-8") as f:
            gj = json.load(f)
    except Exception:
        return df

    polygones = []
    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        i_code = (props.get("iris_code") or ["inconnu"])[0]
        i_name = (props.get("iris_name") or ["Inconnu"])[0]
        geom = feat.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates") or []
        rings = []
        if gtype == "Polygon" and coords:
            rings.append(coords[0])
        elif gtype == "MultiPolygon":
            for poly in coords:
                if poly:
                    rings.append(poly[0])
        for ring in rings:
            if len(ring) >= 3:
                arr = np.asarray(ring, dtype=float)
                polygones.append((i_code, i_name, MplPath(arr[:, :2])))

    if not polygones:
        return df

    mask_gps = df[["latitude", "longitude"]].notna().all(axis=1)
    if not mask_gps.any():
        return df

    # Attention : GeoJSON est en (lon, lat)
    pts = df.loc[mask_gps, ["longitude", "latitude"]].to_numpy(dtype=float)
    codes_out = np.full(len(pts), "inconnu", dtype=object)
    names_out = np.full(len(pts), "Inconnu", dtype=object)
    non_assignes = np.ones(len(pts), dtype=bool)

    for i_code, i_name, path_obj in polygones:
        if not non_assignes.any():
            break
        dedans = path_obj.contains_points(pts[non_assignes])
        if dedans.any():
            idx_glob = np.where(non_assignes)[0][dedans]
            codes_out[idx_glob] = i_code
            names_out[idx_glob] = i_name
            non_assignes[idx_glob] = False

    df.loc[mask_gps, "iris_code"] = codes_out
    df.loc[mask_gps, "iris_name"] = names_out
    return df


# --------------------------------------------------------------------------
# 4. Pipeline Feature Engineering -> Tables PostgreSQL & CSV
# --------------------------------------------------------------------------
def preparer_df_brut(df_brut: pd.DataFrame, verbose: bool = True, avec_photos: bool = True) -> pd.DataFrame:
    """Applique tout le Feature Engineering (Nettoyage 86, Doublons, Texte, LLM, Géo, IRIS GPS, Open Data, Surface, Travaux, DVF, Photos)."""
    df = nettoyer_annonces(df_brut, verbose=verbose, exclure_atypiques=True)
    df = marquer_doublons(df, verbose=verbose)
    df = garder_un_doublon(df)
    df = features_texte(df)
    df = enrichir_avec_ollama(df, utiliser_api_ollama=True)
    df = features_geo(df)
    df = features_iris_spatial_join(df)
    df = features_open_data_grand_poitiers(df)
    df = features_surface_et_structure(df)
    if avec_photos:
        try:
            import Photos
            df = Photos.extraire_features_visuelles(df, utiliser_vlm=False)
        except Exception:
            pass
    df = estimer_cout_travaux(df)
    df = features_marche_historique(df)

    # Valeur locative estimée & rendement brut / net (référentiel loyers Grand Poitiers / ANIL)
    surf_loc = pd.to_numeric(df["surface_m2"], errors="coerce").clip(lower=12.0)
    est_m = df["type_bien"].astype(str) == "house"
    base_loyer_m2 = np.where(
        est_m,
        10.2,
        np.where(surf_loc <= 32, 14.2, np.where(surf_loc <= 55, 12.1, 10.6)),
    )
    rev_med = pd.to_numeric(df.get("revenu_median_secteur", 21500), errors="coerce").fillna(21500.0)
    coef_secteur = np.clip(rev_med / 21500.0, 0.88, 1.15)
    df["loyer_m2_estime"] = np.round(base_loyer_m2 * coef_secteur, 2)
    df["loyer_mensuel_estime"] = np.round(df["loyer_m2_estime"] * surf_loc, 0)
    prix_safe = pd.to_numeric(df["prix"], errors="coerce").replace(0, np.nan)
    trav_loc = pd.to_numeric(df["cout_travaux_estime_euro"], errors="coerce").fillna(0.0)
    df["rendement_brut_pct"] = np.round((df["loyer_mensuel_estime"] * 12.0) / prix_safe * 100.0, 2)
    df["rendement_net_estime_pct"] = np.round((df["loyer_mensuel_estime"] * 10.2) / (prix_safe + trav_loc) * 100.0, 2)

    dvf = telecharger_ou_charger_dvf(CHEMIN_DVF_CSV, verbose=verbose)
    if dvf is not None and len(dvf) > 0:
        dvf_feats = prix_m2_voisinage(df, dvf)
        for col in dvf_feats.columns:
            df[col] = dvf_feats[col]
        surf = pd.to_numeric(df["surface_m2"], errors="coerce").fillna(0.0)
        pm2_ref = df["pm2_dvf_tranche_surface"].fillna(df["pm2_voisinage"])
        df["valeur_dvf_ajustee_taille"] = np.round(surf * pm2_ref, 0)
        df["valeur_dvf_nette_travaux"] = np.round(df["valeur_dvf_ajustee_taille"] - df["cout_travaux_estime_euro"], 0)

    return df


def charger_et_preparer(csv_path: str = os.path.join("csv", "annonces.csv"), verbose: bool = True) -> pd.DataFrame:
    """Lit les offres récentes (PostgreSQL ou dossier csv/) et applique le Feature Engineering."""
    try:
        import DB
        df = DB.lire_table_ou_csv(DB.TABLE_RECENT_BRUT, csv_path)
        if df.empty:
            df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path)
    return preparer_df_brut(df, verbose=verbose)


def construire_X_y(df: pd.DataFrame):
    """
    Extrait la matrice X des variables explicatives (cœur propre sans bruit/redondance)
    et le vecteur cible y (prix) depuis une table après Feature Engineering.
    """
    df = supprimer_colonnes_prediction(df)
    df = df[df["prix"].notna() & (df["prix"] > 0)].copy()
    df["dpe_num"] = df["dpe"].map(ORDRE_DPE)
    for c in CATEGORIELLES:
        if c not in df.columns:
            df[c] = "inconnu"
        else:
            df[c] = df[c].fillna("inconnu").astype(str)

    numeriques = [
        # Structure & Surface (sans doublons monotones log/sqrt/inv)
        "surface_m2", "pieces", "surface_par_piece", "surface_relative_marche",
        "est_mention_carrez", "etage", "annee_construction", "age_bien", "dpe_num",
        # Localisation, Mobilité, Commerces & Open Data GPS (IRIS / QPV conservé / Pôles)
        "latitude", "dist_centre_km", "dist_gare_km", "dist_campus_km",
        "dist_arret_bus_m", "nb_arrets_bus_400m",
        "dist_commerce_alim_m", "nb_commerces_500m", "score_commodites_500m",
        "est_en_qpv",
        # Texte & LLM (sans llm_travaux_score / txt_standing bruit)
        "txt_longueur", "surface_ext_m2", "charges_annuelles",
        "llm_etat_score", "llm_note_offre",
        "txt_piscine", "txt_a_renover", "txt_jardin", "txt_terrasse",
        "txt_parking", "txt_cave", "txt_lumineux", "txt_pac",
        "txt_immeuble",
        # Postes travaux, Valeur locative & Notaires DVF
        "trav_rafraichissement", "trav_cuisine_sdb", "trav_menuiseries", "trav_elec_plomberie",
        "loyer_m2_estime",
        "pm2_voisinage", "pm2_dvf_tranche_surface", "pm2_dvf_pondere_dist",
        "valeur_dvf_ajustee_taille", "valeur_dvf_nette_travaux",
        # Photos : embeddings OpenCLIP (travaux/rénové) + embeddings visuels compacts sans VLM
        "vis_contraste", "vis_saturation", "vis_etat_vlm",
        "clip_score_travaux", "clip_score_renove",
        "vis_emb_0", "vis_emb_1", "vis_emb_2", "vis_emb_3", "vis_emb_4",
    ]

    numeriques = list(dict.fromkeys(
        c for c in numeriques
        if c in df.columns
        and c not in COLONNES_PREDICTION
        and c != "prix"
        and pd.to_numeric(df[c], errors="coerce").notna().sum() >= 2
    ))
    X = df[numeriques + CATEGORIELLES].copy()
    X[numeriques] = X[numeriques].astype(float)
    return X, df["prix"].astype(float)


def construire_tables_features(
    df_hist_brut: pd.DataFrame | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    ÉTAPE 1 — FEATURE ENGINEERING :
    - Lit la table brute historique (`offres_historique_brut`)
    - Applique le Feature Engineering complet (nettoyage, dédoublonnage, regex texte, LLM Ollama, géo, DVF Notaires)
    - Sauvegarde les deux tables APRÈS Feature Engineering (SANS aucune colonne de prédiction) :
        • `offres_historique_features` (`csv/annonces_historique_features.csv`)
        • `offres_recentes_features`   (`csv/annonces_recentes_features.csv`)
    """
    import DB

    if df_hist_brut is None:
        df_hist_brut = DB.lire_table_ou_csv(DB.TABLE_HIST_BRUT, DB.CSV_HIST_BRUT)
        if df_hist_brut.empty:
            df_hist_brut = DB.lire_table_ou_csv(DB.TABLE_RECENT_BRUT, DB.CSV_RECENT_BRUT)

    df_hist_feat = supprimer_colonnes_prediction(preparer_df_brut(df_hist_brut, verbose=verbose))
    if "est_active" in df_hist_feat.columns:
        df_recent_feat = df_hist_feat[df_hist_feat["est_active"] == True].copy()  # noqa: E712
    else:
        df_recent_feat = df_hist_feat.copy()

    DB.sauvegarder_tables_features(df_hist_feat, df_recent_feat)
    print(
        f"[Feature Engineering] Tables de features pures générées (sans colonnes de prédiction) :\n"
        f"  • '{DB.TABLE_HIST_FEAT}'   ({DB.CSV_HIST_FEAT}) : {len(df_hist_feat)} offres ({len(df_hist_feat.columns)} colonnes)\n"
        f"  • '{DB.TABLE_RECENT_FEAT}' ({DB.CSV_RECENT_FEAT}) : {len(df_recent_feat)} offres actives ({len(df_recent_feat.columns)} colonnes)"
    )
    return df_hist_feat, df_recent_feat


if __name__ == "__main__":
    construire_tables_features(verbose=True)
