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
    "log_surface": "Effet de taille (dégressivité m²)",
    "sqrt_surface": "Surface (échelle modérée)",
    "inv_surface": "Effet petite surface (1 / m²)",
    "surface_par_piece": "Espace moyen par pièce (m²/p)",
    "ratio_chambres": "Proportion de chambres",
    "surface_relative_marche": "Taille vs moyenne de la ville",
    "pm2_voisinage": "Prix/m² des ventes notaires proches",
    "pm2_dvf_pondere_dist": "Prix/m² notaire pondéré par la distance",
    "pm2_dvf_taille_proche": "Prix/m² notaire biens de taille proche",
    "pm2_dvf_tranche_surface": "Prix/m² notaire pour cette taille de bien",
    "pm2_dvf_dispersion": "Dispersion des prix notaires voisins (€/m²)",
    "nb_ventes_dvf": "Nombre de ventes notaires proches",
    "tendance_dvf_pct": "Tendance des prix notaires du secteur (%)",
    "valeur_dvf_ajustee_taille": "Valeur notaire ajustée à la taille (€)",
    "valeur_dvf_nette_travaux": "Valeur notaire nette de travaux (€)",
    "cout_travaux_estime_euro": "Budget travaux estimé (€)",
    "cout_travaux_m2": "Budget travaux estimé (€/m²)",
    "trav_rafraichissement": "Poste travaux : Rafraîchissement",
    "trav_cuisine_sdb": "Rénovation cuisine / salle de bain",
    "trav_menuiseries": "Rénovation fenêtres / isolation",
    "trav_elec_plomberie": "Mise aux normes électricité / plomberie",
    "trav_gros_oeuvre": "Travaux de gros œuvre / structure",
    "type_bien": "Type de bien (Maison / Appartement)",
    "ville": "Ville du bien",
    "quartier": "Quartier / Secteur",
    "ville_quartier": "Quartier dans la ville",
    "dist_centre_km": "Distance au centre-ville (km)",
    "dist_centre_norm": "Éloignement relatif au centre-ville",
    "dx_centre_km": "Position Est / Ouest vs centre-ville (km)",
    "dy_centre_km": "Position Nord / Sud vs centre-ville (km)",
    "dx_norm": "Position Est / Ouest normalisée",
    "dy_norm": "Position Nord / Sud normalisée",
    "angle_centre_sin": "Orientation Nord / Sud depuis le centre",
    "angle_centre_cos": "Orientation Est / Ouest depuis le centre",
    "tranche_surface": "Catégorie de surface (m²)",
    "tranche_surface_type": "Catégorie surface × type de bien",
    "typologie": "Typologie du bien (T1/T2/T3… × type)",
    "surface_x_maison": "Surface habitable (Maison)",
    "surface_x_appart": "Surface habitable (Appartement)",
    "ratio_terrain_bati": "Ratio surface extérieure / habitable",
    "dist_gare_km": "Proximité de la gare principale (km)",
    "dist_campus_km": "Proximité pôle universitaire (km)",
    "dist_chu_km": "Proximité pôle santé / CHU (km)",
    "dist_centre_vie_km": "Proximité pôle de vie local (km)",
    "score_centre_vie": "Score d'attractivité & vie de quartier",
    "est_en_qpv": "Situé en Quartier Prioritaire (QPV)",
    "dist_qpv_km": "Distance au Quartier Prioritaire (km)",
    "revenu_median_secteur": "Niveau de vie du micro-quartier (IRIS)",
    "taux_hlm_secteur": "Part de logements sociaux du secteur (%)",
    "part_proprietaires_secteur": "Part de propriétaires occupants (%)",
    "taux_vacance_secteur": "Taux de logements vacants (%)",
    "dpe_num": "Performance énergétique (DPE)",
    "annee_construction": "Année de construction",
    "age_bien": "Ancienneté du bien (années)",
    "ges_num": "Émissions GES (A à G)",
    "nbr_chambres": "Nombre de chambres",
    "pieces": "Nombre de pièces",
    "etage": "Étage du logement",
    "est_rdc": "Rez-de-chaussée (0/1)",
    "etage_eleve_sans_ascenseur": "Étage ≥ 3 sans ascenseur",
    "txt_longueur": "Richesse du descriptif de l'annonce",
    "reseau": "Agence / Réseau",
    "surface_ext_m2": "Surface extérieure (jardin/terrasse m²)",
    "charges_annuelles": "Charges annuelles de copropriété (€)",
    "llm_etat_score": "État général perçu (analyse texte)",
    "llm_travaux_score": "Ampleur des travaux détectés (texte)",
    "llm_note_offre": "Qualité globale de l'annonce",
    "vis_nb_photos": "Nombre de photos de l'annonce",
    "vis_luminosite": "Luminosité des photos",
    "vis_contraste": "Contraste des photos",
    "vis_nettete": "Netteté des photos",
    "vis_saturation": "Qualité visuelle des photos",
    "est_mention_carrez": "Surface certifiée Loi Carrez",
    "surface_carrez_m2": "Surface Loi Carrez (m²)",
    "surface_utile_m2": "Surface utile / au sol (m²)",
    "prix_m2_carrez": "Prix au m² Loi Carrez (€/m²)",
    "prix_m2_utile": "Prix au m² surface utile (€/m²)",
    "dist_arret_bus_m": "Distance arrêt de transport (m)",
    "nb_arrets_bus_400m": "Arrêts de transport à < 400 m",
    "dist_commerce_alim_m": "Distance commerce alimentaire (m)",
    "nb_commerces_500m": "Commerces de proximité à < 500 m",
    "score_commodites_500m": "Indice commodités & mobilité (/20)",
    "loyer_m2_estime": "Loyer de marché estimé (€/m²/mois)",
    "loyer_mensuel_estime": "Loyer mensuel estimé (€/mois)",
    "rendement_brut_pct": "Rendement locatif brut estimé (%)",
    "rendement_net_estime_pct": "Rendement net estimé après travaux (%)",
    "niveau_travaux_cat": "Catégorie de travaux (0 à 4)",
    "clip_etat_score": "État visuel sur photos (IA)",
    "clip_score_travaux": "Travaux visibles sur photos (IA)",
    "clip_score_renove": "Aspect rénové sur photos (IA)",
    "vis_emb_0": "Signature visuelle photos #1",
    "vis_emb_1": "Signature visuelle photos #2",
    "vis_emb_2": "Signature visuelle photos #3",
    "iris_code": "Îlot statistique INSEE (IRIS)",
    "commune_donnees_limitees": "Secteur avec peu de références",
    "txt_jardin": "Présence d'un jardin",
    "txt_a_renover": "Bien signalé à rénover",
    "txt_terrasse": "Présence d'une terrasse",
    "txt_parking": "Parking / garage inclus",
    "txt_balcon": "Présence d'un balcon",
    "txt_ascenseur": "Ascenseur dans l'immeuble",
    "txt_cave": "Cave / cellier",
    "txt_renove": "Bien rénové / refait à neuf",
    "txt_vendu_loue": "Vendu actuellement loué",
    "txt_dernier_etage": "Situé au dernier étage",
    "txt_lumineux": "Exposition lumineuse",
    "txt_calme": "Environnement calme",
    "txt_investissement": "Profil investissement / étudiant",
    "txt_pac": "Pompe à chaleur / climatisation",
    "txt_piscine": "Présence d'une piscine",
    "txt_immeuble": "Immeuble de rapport / multi-lots",
    "txt_standing": "Prestations haut de gamme / architecte",
}

DESCRIPTIONS_FEATURES = {
    "surface_m2": "Surface habitable totale : plus elle est grande, plus le prix total monte, avec un prix au m² généralement dégressif.",
    "surface_par_piece": "Surface moyenne par pièce : mesure le volume et le confort des espaces de vie.",
    "surface_relative_marche": "Compare la taille de ce bien à la médiane des biens similaires dans la même ville.",
    "pm2_voisinage": "Prix médian au m² des vraies ventes notaires (DVF) signées autour du bien dans un rayon de 800 m.",
    "pm2_dvf_pondere_dist": "Moyenne des ventes notaires proches, pondérée par la proximité géographique.",
    "pm2_dvf_tranche_surface": "Prix au m² notarial observé spécifiquement pour la même tranche de taille.",
    "dist_centre_km": "Distance en km jusqu'au centre-ville historique de la ville.",
    "dist_centre_norm": "Éloignement rapporté au rayon urbain de la ville (comparable d'une ville à l'autre).",
    "dx_centre_km": "Décalage Est (+) / Ouest (-) en km par rapport au centre-ville.",
    "dy_centre_km": "Décalage Nord (+) / Sud (-) en km par rapport au centre-ville.",
    "tranche_surface": "Segment de taille du bien (<20, 20-35, 35-60, 60-90, 90-130, >=130 m²).",
    "tranche_surface_type": "Croisement entre le segment de surface et le type de bien (Maison/Appartement).",
    "typologie": "Typologie du bien selon le nombre de pièces et le type (ex. Appartement T2, Maison T4).",
    "ratio_terrain_bati": "Rapport entre la surface extérieure (jardin/terrasse) et la surface habitable.",
    "dist_gare_km": "Distance jusqu'à la gare SNCF/TGV principale de la ville.",
    "dist_campus_km": "Distance jusqu'au pôle universitaire principal (influence la demande studio/T2).",
    "dist_chu_km": "Distance jusqu'au centre hospitalier principal (bassin d'emploi majeur).",
    "dist_centre_vie_km": "Distance au pôle de vie urbain le plus proche (centre-ville, gare ou pôle majeur).",
    "score_centre_vie": "Indice synthétique (0 à 100) combinant centralité, transports et commerces de proximité.",
    "dist_arret_bus_m": "Distance en mètres jusqu'à l'arrêt de transport en commun le plus proche.",
    "dist_commerce_alim_m": "Distance jusqu'au commerce alimentaire le plus proche.",
    "nb_commerces_500m": "Nombre de commerces et services accessibles à moins de 500 m à pied.",
    "dpe_num": "Classe énergétique (A = très économe à G = passoire thermique) : impacte fortement la valeur.",
    "etage": "Étage du logement (les étages élevés avec ascenseur sont valorisés ; le RDC peut décoter).",
    "annee_construction": "Année de construction : reflète l'architecture, l'isolation et les normes techniques.",
    "revenu_median_secteur": "Revenu médian des ménages du secteur INSEE (IRIS) : indicateur objectif du standing du micro-quartier.",
    "taux_hlm_secteur": "Proportion de logements sociaux dans le secteur.",
    "est_en_qpv": "Indique si le bien est situé dans un Quartier Prioritaire de la Politique de la Ville (ANCT).",
    "dist_qpv_km": "Distance en km au Quartier Prioritaire le plus proche.",
    "loyer_m2_estime": "Loyer mensuel estimé au m² pour ce secteur (Carte des loyers ANIL / marché local).",
    "clip_etat_score": "Score d'état intérieur calculé automatiquement par vision IA (OpenCLIP) sur les photos.",
    "clip_score_travaux": "Indice visuel de travaux détectés sur les photos.",
    "ville_quartier": "Effet propre du quartier dans sa ville (évite toute confusion entre quartiers homonymes).",
    "iris_code": "Micro-secteur statistique INSEE (~2 000 habitants), plus fin qu'un nom de quartier d'agence.",
    "type_bien": "Distingue une maison individuelle d'un appartement en copropriété.",
}

CHEMIN_IRIS_GEOJSON = os.path.join("csv", "iris_grand_poitiers.geojson")
ORDRE_DPE = {c: i for i, c in enumerate("ABCDEFG", start=1)}
CATEGORIELLES = ["ville", "ville_quartier", "iris_code", "type_bien", "tranche_surface", "tranche_surface_type", "typologie"]

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
    Nettoie les annonces multi-villes (12 villes-centres V1) :
    - Garde uniquement les biens des 12 villes-centres si `code_postal` / `ville` renseigné
    - Supprime les prix/surfaces manquants ou aberrants
    - Écarte les biens d'exception hors marché (> 750 000 € ou > 300 m² ou < 600 €/m² ou viager/château)
    """
    from src.ingestion.territoires import est_ville_centre, resoudre_territoire

    df = df.copy()
    avant = len(df)
    df = df.dropna(subset=["prix", "surface_m2"])
    df = df[(df["prix"] > 0) & (df["surface_m2"] > 0)].copy()

    if "code_postal" in df.columns or "ville" in df.columns:
        cps = df["code_postal"].tolist() if "code_postal" in df.columns else [None] * len(df)
        villes = df["ville"].tolist() if "ville" in df.columns else [None] * len(df)
        mask_vc = [est_ville_centre(cp, v) for cp, v in zip(cps, villes)]
        if any(mask_vc):
            df = df[mask_vc].copy()

    pm2 = df["prix"] / df["surface_m2"]
    df = df[pm2.between(300, 12000)].copy()

    if exclure_atypiques:
        txt_td = (df["titre"].fillna("") + " " + df["description"].fillna("")).str.lower()
        est_chateau = txt_td.str.contains(
            r"\bch[âa]teau\b|\bmanoir\b|\bpropri[ée]t[ée]\s+d'exception\b|\bhotel\s+particulier\b|\bh[ôo]tel\s+particulier\b",
            regex=True,
        )
        est_viager_encheres = txt_td.str.contains(
            r"\bviager\b|\bbouquet\b|\bnue[-\s]?propri[ée]t[ée]\b|\busufruit\b|\brente\s+viag[èe]re\b|\bench[èe]res\b|\bvente\s+[àa]\s+terme\b",
            regex=True,
        )
        est_atypique_pro = txt_td.str.contains(
            r"\bimmeuble\s+de\s+rapport\b|\bmonopropri[ée]t[ée]\b|\blocal\s+commercial\b|\bmurs\s+commerciaux\b|\bfonds\s+de\s+commerce\b|\bterrain\s+(?:nu|constructible|[àa]\s+b[âa]tir)\b|\b(?:parking|box|garage)\s+(?:seul|ferm[ée]\s+seul)\b|\bruine\b|\b[àa]\s+d[ée]molir\b|\bp[ée]niche\b|\bmobile[-\s]?home\b",
            regex=True,
        )
        pm2_cur = df["prix"] / df["surface_m2"]
        df = df[
            (df["prix"] >= 25_000)
            & (df["surface_m2"] >= 12)
            & (df["surface_m2"] <= 200)
            & (pm2_cur >= 500)
            & (~est_chateau)
            & (~est_viager_encheres)
            & (~est_atypique_pro)
        ].copy()

    if "quartier" in df.columns:
        df["quartier"] = df["quartier"].replace(QUARTIERS_HARMONISES)
    if "type_bien" in df.columns:
        df["type_bien"] = df["type_bien"].replace(TYPES_BIEN_HARMONISES)
    cps_2 = df["code_postal"].tolist() if "code_postal" in df.columns else [None] * len(df)
    villes_2 = df["ville"].tolist() if "ville" in df.columns else [None] * len(df)
    df["code_territoire"] = [resoudre_territoire(cp, v) or "poitiers" for cp, v in zip(cps_2, villes_2)]

    # Filtrage statistique adaptatif par ville (code_territoire, type_bien)
    if exclure_atypiques and len(df) >= 25:
        df["_pm2_tmp"] = df["prix"] / df["surface_m2"]
        keep_mask = pd.Series(True, index=df.index)
        for terr, grp_t in df.groupby("code_territoire"):
            for tb, grp in grp_t.groupby("type_bien"):
                ref = grp if len(grp) >= 25 else grp_t
                if len(ref) < 20:
                    continue
                for col_f, q_low, q_high, k_iqr in [
                    ("prix", 0.02, 0.98, 2.0),
                    ("surface_m2", 0.02, 0.98, 2.0),
                    ("_pm2_tmp", 0.02, 0.98, 2.0),
                ]:
                    s = pd.to_numeric(ref[col_f], errors="coerce").dropna()
                    if len(s) < 20:
                        continue
                    q02, q25, q75, q98 = s.quantile([q_low, 0.25, 0.75, q_high]).values
                    iqr = max(q75 - q25, 1e-6)
                    lo = max(q02, q25 - k_iqr * iqr)
                    hi = min(q98, q75 + k_iqr * iqr)
                    if col_f == "surface_m2":
                        hi = min(hi, 200.0)
                    vals = pd.to_numeric(grp[col_f], errors="coerce")
                    keep_mask.loc[grp.index] &= vals.between(lo, hi)
        df = df[keep_mask].drop(columns=["_pm2_tmp"]).copy()

    suppr = avant - len(df)
    if verbose and suppr:
        print(f"{suppr} annonces supprimées (hors 12 villes-centres, >200m², atypiques ou outliers adaptatifs par ville).")
    q_col = df["quartier"].fillna("Centre").astype(str).str.strip() if "quartier" in df.columns else pd.Series("Centre", index=df.index)
    v_col = df["ville"].fillna(df["code_territoire"]).astype(str).str.strip() if "ville" in df.columns else df["code_territoire"]
    df["ville_quartier"] = v_col + "__" + q_col
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
    """Calcule les distances aux pôles urbains locaux de la ville du bien (centre, gare, campus, CHU, pôle de vie)."""
    from src.ingestion.territoires import VILLES_CENTRES, resoudre_territoire, territoire_plus_proche

    df = df.copy()
    if len(df) == 0:
        for col in [
            "dist_centre_km", "dist_centre_norm", "dist_gare_km", "dist_campus_km",
            "dist_chu_km", "dist_centre_vie_km",
            "dx_centre_km", "dy_centre_km", "dx_norm", "dy_norm",
            "angle_centre_sin", "angle_centre_cos",
        ]:
            df[col] = np.nan
        return df

    lats = pd.to_numeric(df["latitude"], errors="coerce").to_numpy(dtype=float)
    lons = pd.to_numeric(df["longitude"], errors="coerce").to_numpy(dtype=float)
    cps = df["code_postal"].tolist() if "code_postal" in df.columns else [None] * len(df)
    villes = df["ville"].tolist() if "ville" in df.columns else [None] * len(df)
    terrs = df["code_territoire"].tolist() if "code_territoire" in df.columns else [None] * len(df)

    d_centre, d_norm, d_gare, d_campus, d_chu, d_vie, codes_out = [], [], [], [], [], [], []
    dx_l, dy_l, dx_n_l, dy_n_l, ang_s_l, ang_c_l = [], [], [], [], [], []
    for i in range(len(df)):
        code = terrs[i] if (terrs[i] and str(terrs[i]) in VILLES_CENTRES) else resoudre_territoire(cps[i], villes[i])
        if not code and np.isfinite(lats[i]) and np.isfinite(lons[i]):
            code = territoire_plus_proche(lats[i], lons[i], max_km=30.0)
        cfg = VILLES_CENTRES.get(code or "poitiers", VILLES_CENTRES["poitiers"])
        codes_out.append(cfg.code_territoire)

        if np.isfinite(lats[i]) and np.isfinite(lons[i]):
            dc = float(haversine_km(lats[i], lons[i], cfg.centre_lat, cfg.centre_lon))
            dg = float(haversine_km(lats[i], lons[i], cfg.gare_lat, cfg.gare_lon))
            du = float(haversine_km(lats[i], lons[i], cfg.campus_lat, cfg.campus_lon))
            dh = float(haversine_km(lats[i], lons[i], cfg.chu_lat, cfg.chu_lon))
            rayon = max(cfg.rayon_urbain_km, 1.0)
            dy_km = (lats[i] - cfg.centre_lat) * 111.32
            dx_km = (lons[i] - cfg.centre_lon) * 111.32 * float(np.cos(np.radians(cfg.centre_lat)))
            theta = float(np.arctan2(dy_km, dx_km))

            d_centre.append(round(dc, 3))
            d_norm.append(round(dc / rayon, 3))
            d_gare.append(round(dg, 3))
            d_campus.append(round(du, 3))
            d_chu.append(round(dh, 3))
            d_vie.append(round(min(dc, dg, du, dh), 3))
            dx_l.append(round(dx_km, 3))
            dy_l.append(round(dy_km, 3))
            dx_n_l.append(round(dx_km / rayon, 3))
            dy_n_l.append(round(dy_km / rayon, 3))
            ang_s_l.append(round(float(np.sin(theta)), 4))
            ang_c_l.append(round(float(np.cos(theta)), 4))
        else:
            d_centre.append(np.nan)
            d_norm.append(np.nan)
            d_gare.append(np.nan)
            d_campus.append(np.nan)
            d_chu.append(np.nan)
            d_vie.append(np.nan)
            dx_l.append(np.nan)
            dy_l.append(np.nan)
            dx_n_l.append(np.nan)
            dy_n_l.append(np.nan)
            ang_s_l.append(np.nan)
            ang_c_l.append(np.nan)

    df["code_territoire"] = codes_out
    df["dist_centre_km"] = d_centre
    df["dist_centre_norm"] = d_norm
    df["dist_gare_km"] = d_gare
    df["dist_campus_km"] = d_campus
    df["dist_chu_km"] = d_chu
    df["dist_centre_vie_km"] = d_vie
    df["dx_centre_km"] = dx_l
    df["dy_centre_km"] = dy_l
    df["dx_norm"] = dx_n_l
    df["dy_norm"] = dy_n_l
    df["angle_centre_sin"] = ang_s_l
    df["angle_centre_cos"] = ang_c_l
    return df


def features_open_data_grand_poitiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rattache par territoire & GPS les indicateurs Open Data multi-villes :
    - QPV : `est_en_qpv`, `dist_qpv_km` (Poitiers + table PostGIS `ref_qpv` quand chargée)
    - INSEE FiLoSoFi : `revenu_median_secteur`, `taux_hlm_secteur`, `part_proprietaires_secteur`, `taux_vacance_secteur`
    - Mobilité / Commerces (`ref_poi` ou CSV locaux) + `score_centre_vie` (0-100).
    """
    from src.ingestion.territoires import VILLES_CENTRES, resoudre_territoire

    df = df.copy()
    if len(df) == 0:
        return df

    # 1. Distance au QPV le plus proche & appartenance QPV (Poitiers connu + fallback propre)
    lats = pd.to_numeric(df["latitude"], errors="coerce").to_numpy(dtype=float)
    lons = pd.to_numeric(df["longitude"], errors="coerce").to_numpy(dtype=float)
    dists_qpv = np.column_stack([
        haversine_km(lats, lons, lat_q, lon_q)
        for _, lat_q, lon_q, _ in QPV_GRAND_POITIERS
    ])
    rayons_qpv = np.array([r_q for _, _, _, r_q in QPV_GRAND_POITIERS])
    with np.errstate(all="ignore"):
        d_qpv_min = np.nanmin(dists_qpv, axis=1)
        en_qpv = np.nanmin(dists_qpv - rayons_qpv, axis=1) <= 0
    # Hors Grand Poitiers (distance > 15 km), ne pas imputer un QPV Poitiers
    df["dist_qpv_km"] = np.where(np.isfinite(d_qpv_min) & (d_qpv_min <= 15.0), np.round(d_qpv_min, 3), np.nan)
    df["est_en_qpv"] = np.where(np.isfinite(d_qpv_min) & (d_qpv_min <= 15.0) & en_qpv, 1, 0)

    # 2. Revenus médians, HLM, propriétaires & vacance (sans fallback silencieux vers Poitiers 86000)
    rev_l, hlm_l, prop_l, vac_l = [], [], [], []
    for _, row in df.iterrows():
        q = str(row.get("quartier") or "")
        cp = str(row.get("code_postal") or "")[:5]
        terr = row.get("code_territoire") or resoudre_territoire(cp, row.get("ville")) or "poitiers"
        if terr == "poitiers" and q in STATS_QUARTIERS_POITIERS:
            stats = STATS_QUARTIERS_POITIERS[q]
        elif terr == "poitiers" and cp in STATS_COMMUNES_GRAND_POITIERS:
            stats = STATS_COMMUNES_GRAND_POITIERS[cp]
        else:
            cfg = VILLES_CENTRES.get(terr, VILLES_CENTRES["poitiers"])
            stats = (cfg.revenu_median_defaut, 22.0, 40.0, 8.5)

        if terr == "poitiers" and row.get("est_en_qpv", 0) == 1 and q not in STATS_QUARTIERS_POITIERS:
            stats = (16800, 48.0, 26.0, 10.0)

        rev_l.append(stats[0])
        hlm_l.append(stats[1])
        prop_l.append(stats[2])
        vac_l.append(stats[3])

    df["revenu_median_secteur"] = rev_l
    df["taux_hlm_secteur"] = hlm_l
    df["part_proprietaires_secteur"] = prop_l
    df["taux_vacance_secteur"] = vac_l

    # 3. Mobilité & Commerces (PostgreSQL `ref_poi` si peuplé, sinon CSV Poitiers pour le 86)
    lat_arr = lats
    lon_arr = lons

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
            d_m_min = float(np.min(d_km) * 1000.0)
            if d_m_min <= 15000.0:
                dist_min_m[i] = round(d_m_min, 0)
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

    d_norm = pd.to_numeric(df.get("dist_centre_norm", pd.Series(0.5, index=df.index)), errors="coerce").fillna(0.5)
    df["score_centre_vie"] = np.round(
        np.clip(100.0 * np.exp(-1.1 * d_norm) * 0.7 + df["score_commodites_500m"] * 1.5, 0.0, 100.0),
        1,
    )
    return df


features_open_data_territoires = features_open_data_grand_poitiers


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

    # Catégories de taille, typologie (T1/T2/...) et interactions surface × type
    tb_str = df["type_bien"].fillna("flat").astype(str) if "type_bien" in df.columns else pd.Series("flat", index=df.index)
    est_maison = (tb_str == "house").astype(float)
    df["surface_x_maison"] = np.round(surf * est_maison, 1)
    df["surface_x_appart"] = np.round(surf * (1.0 - est_maison), 1)
    surf_ext = pd.to_numeric(df.get("surface_ext_m2", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    df["ratio_terrain_bati"] = np.round(np.clip(surf_ext / surf, 0.0, 30.0), 3)

    tranche_s = pd.cut(
        surf,
        bins=[0.0, 20.0, 35.0, 60.0, 90.0, 130.0, float("inf")],
        labels=["<20", "20_35", "35_60", "60_90", "90_130", ">=130"],
        right=False,
    ).astype(str)
    df["tranche_surface"] = tranche_s
    df["tranche_surface_type"] = tranche_s + "__" + tb_str

    p_int = pieces.round(0).clip(lower=1, upper=5).fillna(2).astype(int)
    t_lbl = np.where(p_int >= 5, "T5+", "T" + p_int.astype(str))
    df["typologie"] = tb_str + "__" + pd.Series(t_lbl, index=df.index)
    df["age_bien"] = np.where(np.isfinite(annee), 2026.0 - annee, np.nan)
    df["est_rdc"] = np.where(etage == 0.0, 1, 0)
    asc = pd.to_numeric(df.get("txt_ascenseur", pd.Series(0, index=df.index)), errors="coerce").fillna(0.0).astype(float)
    df["etage_eleve_sans_ascenseur"] = np.where((etage >= 3.0) & (asc == 0.0), 1, 0)

    # Surface relative par rapport au marché du même type de bien et de la même ville
    grp_cols = ["ville", "type_bien"] if "ville" in df.columns else ["type_bien"]
    med_par_type = df.groupby(grp_cols)["surface_m2"].transform("median")
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

    # Flag communes peu représentées
    if "ville" in df.columns:
        ville_clean = df["ville"].fillna("inconnu").astype(str).str.strip()
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
    est_renove_recent = desc.str.contains(
        r"refait\s+[àa]\s+neuf|r[ée]novation\s+r[ée]cente|enti[èe]rement\s+r[ée]nov[ée]|aucun\s+travaux",
        regex=True,
    )
    annee_c = pd.to_numeric(df.get("annee_construction", pd.Series(np.nan, index=df.index)), errors="coerce")

    # Signaux OpenCLIP globaux (5 catégories -> état 1..5, travaux, rénové)
    clip_trav = pd.to_numeric(df.get("clip_score_travaux", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    clip_renove = pd.to_numeric(df.get("clip_score_renove", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    clip_etat = pd.to_numeric(df.get("clip_etat_score", pd.Series(3.0, index=df.index)), errors="coerce").fillna(3.0)
    llm_etat = pd.to_numeric(df.get("llm_etat_score", pd.Series(3.0, index=df.index)), errors="coerce").fillna(3.0)

    # Colonne unifiée `niveau_travaux_cat` (0=Aucun, 1=Légers 400€/m², 2=Rénovation 750€/m², 3=Remise à neuf 1200€/m², 4=Lourde 2000€/m²)
    est_remise_neuf = desc.str.contains(r"enti[èe]rement\s+[àa]\s+r[ée]nover|r[ée]novation\s+compl[èe]te|remise\s+[àa]\s+neuf", regex=True)
    est_lourde = desc.str.contains(r"gros\s+[œo]uvre|r[ée]novation\s+lourde|charpente\s+[àa]\s+refaire|ruine", regex=True)

    niv = np.zeros(len(df), dtype=int)
    cond_n1 = (t_raf == 1) | (llm_trav == 1) | (clip_trav >= 0.42)
    niv = np.where(cond_n1, np.maximum(niv, 1), niv)

    cond_n2 = (
        desc.str.contains(r"[àa]\s+r[ée]nover|r[ée]novation\s+[àa]\s+pr[ée]voir", regex=True)
        | (llm_trav == 2)
        | (clip_trav >= 0.55)
        | ((t_el == 1) & (t_men == 1))
        | ((annee_c.notna()) & (annee_c < 1975) & dpe.isin(["F", "G"]) & (~est_renove_recent) & (clip_renove < 0.60))
    )
    niv = np.where(cond_n2, np.maximum(niv, 2), niv)

    cond_n3 = est_remise_neuf | (llm_trav >= 3) | (clip_trav >= 0.68)
    niv = np.where(cond_n3, np.maximum(niv, 3), niv)

    cond_n4 = est_lourde | ((llm_trav >= 3) & (llm_etat <= 1.5)) | (clip_etat <= 1.6)
    niv = np.where(cond_n4, 4, niv)

    # Garde-fou : si explicitement refait à neuf ET CLIP confirme rénové (>= 0.60)
    niv = np.where(est_renove_recent & (clip_renove >= 0.60), 0, niv)

    tarif_m2 = np.select(
        [niv == 1, niv == 2, niv == 3, niv == 4],
        [400.0, 750.0, 1200.0, 2000.0],
        default=0.0,
    )
    cout_base = surf * tarif_m2

    # Isolation / DPE (Maison vs Appartement, réduit de 50% en Remise à neuf / Lourde)
    tarif_isol_m2 = np.where(
        dpe == "G", np.where(est_maison, 220.0, 130.0),
        np.where(dpe == "F", np.where(est_maison, 170.0, 100.0),
        np.where(dpe == "E", np.where(est_maison, 80.0, 50.0), 0.0))
    )
    coef_isol = np.where(niv >= 3, 0.5, 1.0)
    cout = cout_base + surf * tarif_isol_m2 * coef_isol

    df["niveau_travaux_cat"] = niv
    df["trav_rafraichissement"] = t_raf
    df["trav_cuisine_sdb"] = t_cui
    df["trav_menuiseries"] = t_men
    df["trav_elec_plomberie"] = t_el
    df["trav_gros_oeuvre"] = np.maximum((niv >= 3).astype(int), t_isol)

    df["cout_travaux_estime_euro"] = np.round(cout, 0)
    df["cout_travaux_min_euro"] = np.round(cout * 0.78, 0)
    df["cout_travaux_max_euro"] = np.round(cout * 1.25, 0)
    df["cout_travaux_m2"] = np.round(cout / surf.clip(lower=1.0), 0)

    noms_niv = {0: "", 1: "Travaux légers (400€/m²)", 2: "Rénovation (750€/m²)", 3: "Remise à neuf (1200€/m²)", 4: "Rénovation lourde (2000€/m²)"}
    dpe_arr = dpe.to_numpy()
    libelles = []
    for i in range(len(df)):
        p = []
        if niv[i] > 0:
            p.append(noms_niv[int(niv[i])])
        if dpe_arr[i] in {"E", "F", "G"}:
            p.append(f"Isolation DPE {dpe_arr[i]}")
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


CHEMIN_DVF_MULTI_CSV = os.path.join("csv", "dvf_multi_villes.csv")


def telecharger_ou_charger_dvf(dvf_path: str = CHEMIN_DVF_MULTI_CSV, verbose: bool = True) -> pd.DataFrame | None:
    """
    Charge les ventes notariées DVF des 12 villes-centres (Etalab geo-dvf 2023-2025),
    avec repli local sur `csv/dvf_grand_poitiers.csv` si hors-ligne.
    """
    from src.ingestion.territoires import VILLES_CENTRES

    os.makedirs(os.path.dirname(dvf_path) or "csv", exist_ok=True)
    if os.path.exists(dvf_path) and os.path.getsize(dvf_path) > 5000:
        try:
            return pd.read_csv(dvf_path, low_memory=False)
        except Exception:
            pass

    dfs = []
    for code, cfg in VILLES_CENTRES.items():
        dept = cfg.code_insee[:2]
        for annee in ("2025", "2024", "2023"):
            u = f"https://files.data.gouv.fr/geo-dvf/latest/csv/{annee}/communes/{dept}/{cfg.code_insee}.csv"
            try:
                req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    sub = pd.read_csv(resp, low_memory=False)
                    sub["code_territoire"] = code
                    dfs.append(sub)
            except Exception:
                continue

    if dfs:
        dvf_all = pd.concat(dfs, ignore_index=True).drop_duplicates()
        dvf_all.to_csv(dvf_path, index=False)
        if verbose:
            print(f"Fichier DVF 12 villes-centres téléchargé ({len(dvf_all)} lignes) -> {dvf_path}")
        return dvf_all

    for fallback in (CHEMIN_DVF_CSV, CHEMIN_DVF_POITIERS_LEGACY):
        if os.path.exists(fallback):
            return pd.read_csv(fallback, low_memory=False)
    return None


def nettoyer_dvf(dvf: pd.DataFrame) -> pd.DataFrame:
    """Filtre les ventes notariées d'appartements et maisons (bornes multi-villes élargies)."""
    dvf = dvf[
        (dvf["nature_mutation"] == "Vente")
        & (dvf["type_local"].isin(["Appartement", "Maison"]))
    ].copy()
    dvf = dvf[dvf.groupby("id_mutation")["id_mutation"].transform("size") <= 2]
    dvf = dvf.dropna(subset=["valeur_fonciere", "surface_reelle_bati", "latitude", "longitude"])
    dvf = dvf[(dvf["surface_reelle_bati"] >= 12) & (dvf["surface_reelle_bati"] <= 400)]
    dvf["pm2"] = dvf["valeur_fonciere"] / dvf["surface_reelle_bati"]
    dvf = dvf[dvf["pm2"].between(500, 7500)].copy()
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
            from src.ml import photos as Photos
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
        from src.ingestion import db as DB
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
        # Structure & Surface (incluant courbure non-linéaire et interactions type)
        "surface_m2", "log_surface", "sqrt_surface", "inv_surface",
        "surface_x_maison", "surface_x_appart", "ratio_terrain_bati",
        "pieces", "surface_par_piece", "surface_relative_marche",
        "est_mention_carrez", "etage", "annee_construction", "dpe_num",
        # Localisation relative aux pôles urbains locaux, repère centré (dx/dy/angle) & Open Data
        "dist_centre_km", "dist_centre_norm", "dist_gare_km", "dist_campus_km",
        "dist_chu_km", "dist_centre_vie_km", "score_centre_vie",
        "dx_centre_km", "dy_centre_km", "dx_norm", "dy_norm",
        "angle_centre_sin", "angle_centre_cos",
        "revenu_median_secteur", "taux_hlm_secteur", "est_en_qpv", "dist_qpv_km",
        "dist_arret_bus_m", "dist_commerce_alim_m", "nb_commerces_500m",
        # Texte & LLM
        "txt_longueur", "surface_ext_m2", "charges_annuelles",
        "llm_etat_score", "llm_travaux_score", "llm_note_offre",
        "txt_piscine", "txt_a_renover", "txt_jardin", "txt_terrasse",
        "txt_parking", "txt_cave", "txt_lumineux", "txt_immeuble",
        # Postes travaux utiles, Valeur locative & Notaires DVF
        "trav_cuisine_sdb", "trav_menuiseries", "trav_elec_plomberie",
        "loyer_m2_estime",
        "pm2_voisinage", "pm2_dvf_tranche_surface", "pm2_dvf_pondere_dist",
        "valeur_dvf_ajustee_taille", "valeur_dvf_nette_travaux",
        # Photos : embeddings OpenCLIP (5 catégories) + contraste/saturation + 3 composantes visuelles
        "vis_contraste", "vis_saturation",
        "clip_etat_score", "clip_score_travaux",
        "vis_emb_0", "vis_emb_1", "vis_emb_2",
    ]
    if "ville_quartier" not in df.columns:
        v_s = df["ville"].fillna("inconnu").astype(str).str.strip() if "ville" in df.columns else pd.Series("inconnu", index=df.index)
        q_s = df["quartier"].fillna("inconnu").astype(str).str.strip() if "quartier" in df.columns else pd.Series("inconnu", index=df.index)
        df["ville_quartier"] = v_s + "__" + q_s

    numeriques = list(dict.fromkeys(
        c for c in numeriques
        if c in df.columns
        and c not in COLONNES_PREDICTION
        and c != "prix"
        and pd.to_numeric(df[c], errors="coerce").notna().sum() >= 20
        and pd.to_numeric(df[c], errors="coerce").nunique(dropna=True) >= 2
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
    from src.ingestion import db as DB

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
