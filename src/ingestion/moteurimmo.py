"""
MoteurImmo.py — Capture interactive Playwright depuis moteurimmo.fr, export JSON et intégration PostgreSQL.

Usage :
  - Mode interactif (connexion manuelle + recherche Poitiers -> export moteurimmo_annonces.json + BDD) :
      python MoteurImmo.py
  - Importer un JSON déjà exporté dans la base PostgreSQL :
      python MoteurImmo.py --import-json moteurimmo_annonces.json
  - Importer un CSV déjà téléchargé (et le convertir aussi en moteurimmo_annonces.json) :
      python MoteurImmo.py --import-csv data/moteurimmo/poitiers.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd
from playwright.async_api import async_playwright

from src.ingestion import db as DB

DOWNLOAD_DIR = Path("data/moteurimmo")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

PROFILE_DIR = Path("moteurimmo_profile")
JSON_EXPORT_PATH = Path("moteurimmo_annonces.json")
CSV_EXPORT_PATH = DOWNLOAD_DIR / "moteurimmo_poitiers.csv"
LAST_URL_PATH = DOWNLOAD_DIR / "last_search_url.txt"

COLONNES_STANDARD = [
    "id",
    "titre",
    "ville",
    "code_postal",
    "prix",
    "surface_m2",
    "pieces",
    "nbr_chambres",
    "etage",
    "annee_construction",
    "dpe",
    "ges",
    "quartier",
    "latitude",
    "longitude",
    "type_bien",
    "description",
    "date_publication",
    "url",
    "photos_urls",
    "source",
]

COMMUNES_GRAND_POITIERS = {
    "poitiers",
    "buxerolles",
    "saint-benoit",
    "biard",
    "chasseneuil-du-poitou",
    "migne-auxances",
    "vouneuil-sous-biard",
    "montamise",
    "mignaloux-beauvoir",
    "fontaine-le-comte",
    "liguge",
    "smarves",
    "croutelle",
    "sevres-anxaumont",
    "jaunay-marigny",
    "saint-georges-les-baillargeaux",
    "nouaille-maupertuis",
    "roches-premarie-andille",
    "iteuil",
    "dissay",
    "lusignan",
    "chauvigny",
}

CENTROIDS_POITIERS = {
    "centre": (46.5802, 0.3404, "Centre-ville"),
    "plateau": (46.5802, 0.3404, "Centre-ville"),
    "couronneries": (46.5975, 0.3510, "Couronneries"),
    "trois cites": (46.5655, 0.3550, "Trois Cités"),
    "beaulieu": (46.5715, 0.3710, "Beaulieu"),
    "gibauderie": (46.5650, 0.3480, "La Gibauderie"),
    "ganterie": (46.5695, 0.3350, "Ouest"),
    "saint-eloi": (46.5840, 0.3770, "Saint-Éloi"),
    "saint-cyprien": (46.5665, 0.3410, "Saint-Cyprien"),
    "montbernage": (46.5795, 0.3545, "Montbernage"),
    "chilvert": (46.5775, 0.3260, "Chilvert"),
    "bel air": (46.5885, 0.3210, "Bel Air"),
    "buxerolles": (46.5985, 0.3515, "Buxerolles"),
    "saint-benoit": (46.5495, 0.3445, "Saint-Benoît"),
    "biard": (46.5775, 0.3055, "Biard"),
    "vouneuil": (46.5735, 0.2715, "Vouneuil-sous-Biard"),
    "migne": (46.6275, 0.3125, "Migné-Auxances"),
    "chasseneuil": (46.6515, 0.3725, "Chasseneuil-du-Poitou"),
    "fontaine-le-comte": (46.5285, 0.2625, "Fontaine-le-Comte"),
    "liguge": (46.5175, 0.3285, "Ligugé"),
    "montamise": (46.6135, 0.4235, "Montamisé"),
    "mignaloux": (46.5435, 0.4155, "Mignaloux-Beauvoir"),
    "jaunay": (46.6855, 0.3715, "Jaunay-Marigny"),
}


def _normaliser_nom_colonne(col: str) -> str:
    txt = unicodedata.normalize("NFKD", str(col or "")).encode("ascii", "ignore").decode("ascii")
    txt = re.sub(r"[^a-z0-9]+", "_", txt.lower()).strip("_")
    return txt


def _extraire_nombre(val, entier: bool = False):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return int(round(val)) if entier else float(val)
    s = str(val).strip()
    if not s or s.lower() in {"nan", "none", "null", "nc", "n/c", "-"}:
        return None
    s = s.replace("\u202f", "").replace("\xa0", "").replace(" ", "")
    s = re.sub(r"(?:€|eur|m²|m2)", "", s, flags=re.IGNORECASE)
    s = s.replace(",", ".")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        num = float(m.group(0))
        return int(round(num)) if entier else num
    except ValueError:
        return None


def _normaliser_dpe_ges(val) -> str | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, dict):
        val = val.get("letter") or val.get("classe") or val.get("value") or val.get("label")
    s = str(val or "").strip().upper()
    m = re.match(r"^([A-G])\b", s)
    if m:
        return m.group(1)
    if len(s) == 1 and s in "ABCDEFG":
        return s
    return None


def _normaliser_type_bien(val, titre: str = "", extra: str = "") -> str:
    txt = f"{val or ''} {titre or ''} {extra or ''}".lower()
    if any(k in txt for k in ("maison", "villa", "pavillon", "fermette", "longere", "longère", "plain-pied", "house")):
        return "house"
    return "flat"


def _normaliser_code_postal(val) -> str | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    m = re.search(r"\b(\d{5})\b", str(val))
    return m.group(1) if m else None


def _premier_val(d: dict, *cles: str):
    if not isinstance(d, dict):
        return None
    norm_lookup = {_normaliser_nom_colonne(k): v for k, v in d.items()}
    for c in cles:
        if c in norm_lookup and norm_lookup[c] not in (None, "", []):
            return norm_lookup[c]
    return None


def _extraire_photos(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    if isinstance(val, list):
        urls = []
        for item in val:
            if isinstance(item, str) and item.startswith("http"):
                urls.append(item)
            elif isinstance(item, dict):
                u = item.get("url") or item.get("src") or item.get("href")
                if isinstance(u, str) and u.startswith("http"):
                    urls.append(u)
        return json.dumps(urls, ensure_ascii=False)
    return str(val).strip()


def normaliser_items_moteurimmo(
    items: list[dict],
    filtrer_poitiers_86: bool = True,
) -> pd.DataFrame:
    """
    Normalise une liste de dictionnaires d'annonces (issus d'un JSON ou d'un CSV MoteurImmo)
    vers le schéma standard du projet.
    """
    if not items:
        return pd.DataFrame(columns=COLONNES_STANDARD)

    rows = []
    for item in items:
        if not isinstance(item, dict):
            continue

        loc = item.get("location") if isinstance(item.get("location"), dict) else {}
        coords = item.get("coordinates") if isinstance(item.get("coordinates"), dict) else {}
        if not coords and isinstance(loc.get("coordinates"), dict):
            coords = loc.get("coordinates")

        url_val = str(_premier_val(item, "url", "lien", "link", "annonce_url", "ad_url", "origin_url") or "").strip()
        if url_val.lower() in {"nan", "none"}:
            url_val = ""

        titre_val = str(_premier_val(item, "titre", "title", "nom", "libelle", "intitule") or "").strip()
        if titre_val.lower() in {"nan", "none"}:
            titre_val = ""

        ville_val = str(
            _premier_val(item, "ville", "city", "commune", "localite")
            or _premier_val(loc, "city", "ville", "commune")
            or ""
        ).strip()
        if ville_val.lower() in {"nan", "none"}:
            ville_val = ""

        cp_val = _normaliser_code_postal(
            _premier_val(item, "code_postal", "cp", "postal_code", "zip", "zipcode")
            or _premier_val(loc, "postal_code", "code_postal", "zip")
        )
        if not cp_val and ville_val:
            cp_val = _normaliser_code_postal(ville_val)
        if not cp_val and titre_val:
            cp_val = _normaliser_code_postal(titre_val)

        raw_prix = _premier_val(item, "prix", "price", "montant", "prix_fai", "ad_price", "amount")
        raw_surf = _premier_val(item, "surface_m2", "surface", "surface_habitable", "area", "superficie", "living_area", "habitable_surface", "size", "sqm")
        if isinstance(raw_prix, dict) or isinstance(raw_surf, dict):
            continue

        prix_val = _extraire_nombre(raw_prix, entier=False)
        surface_val = _extraire_nombre(raw_surf, entier=False)
        if not prix_val or prix_val < 5000 or not surface_val or surface_val < 5:
            continue

        if filtrer_poitiers_86:
            ville_norm = unicodedata.normalize("NFKD", ville_val).encode("ascii", "ignore").decode("ascii").lower()
            est_86 = bool(cp_val and cp_val.startswith("86"))
            est_grand_poitiers = any(c in ville_norm for c in COMMUNES_GRAND_POITIERS)
            if (cp_val or ville_val) and not (est_86 or est_grand_poitiers):
                continue

        raw_id = str(_premier_val(item, "id", "identifiant", "reference", "ref", "ad_id", "unique_id") or "").strip()
        if raw_id and raw_id.lower() not in {"nan", "none", "null"}:
            ad_id = raw_id if raw_id.startswith("mi_") else f"mi_{raw_id}"
        else:
            cle = url_val or f"{titre_val}|{ville_val}|{cp_val}|{prix_val}|{surface_val}"
            ad_id = f"mi_{hashlib.sha1(cle.encode('utf-8')).hexdigest()[:16]}"

        src_raw = str(_premier_val(item, "source", "origine", "site", "portail", "plateforme", "origin") or "").strip().lower()
        if not src_raw or src_raw in {"nan", "none", "null"}:
            src_val = DB.detecter_source(ad_id, url_val, "moteurimmo")
        else:
            src_val = f"moteurimmo_{src_raw}" if "moteurimmo" not in src_raw else src_raw

        lat_val = _extraire_nombre(_premier_val(item, "latitude", "lat") or coords.get("lat") or coords.get("latitude"), entier=False)
        lon_val = _extraire_nombre(_premier_val(item, "longitude", "lon", "lng", "long") or coords.get("lng") or coords.get("lon") or coords.get("longitude"), entier=False)

        desc_val = str(_premier_val(item, "description", "texte", "annonce", "commentaire") or "").strip()
        if desc_val.lower() in {"nan", "none"}:
            desc_val = ""

        quartier_val = _premier_val(item, "quartier", "secteur", "district", "neighborhood") or _premier_val(loc, "district", "quartier")
        if quartier_val is not None:
            quartier_val = str(quartier_val).strip()
            if quartier_val.lower() in {"nan", "none", ""}:
                quartier_val = None

        # Enrichissement quartier + GPS de repli si la carte DOM n'a pas de coordonnées exactes
        txt_search = unicodedata.normalize("NFKD", f"{ville_val} {titre_val} {desc_val}").encode("ascii", "ignore").decode("ascii").lower()
        for cle_q, (q_lat, q_lon, q_nom) in CENTROIDS_POITIERS.items():
            if cle_q in txt_search:
                if not quartier_val:
                    quartier_val = q_nom
                if lat_val is None or lon_val is None:
                    lat_val, lon_val = q_lat, q_lon
                break
        if lat_val is None or lon_val is None:
            lat_val, lon_val = 46.5802, 0.3404  # Centre Poitiers par défaut pour débloquer DVF/AVM/Mobilité

        dpe_val = _normaliser_dpe_ges(_premier_val(item, "dpe", "classe_energie", "energy_class", "dpe_lettre", "energy_rate"))
        if not dpe_val and desc_val:
            m_dpe = re.search(r"\bDPE\s*[:\-]?\s*([A-G])\b", desc_val, re.I)
            if m_dpe:
                dpe_val = m_dpe.group(1).upper()

        ges_val = _normaliser_dpe_ges(_premier_val(item, "ges", "classe_ges", "ghg", "emission_ges", "ges_rate"))
        if not ges_val and desc_val:
            m_ges = re.search(r"\bGES\s*[:\-]?\s*([A-G])\b", desc_val, re.I)
            if m_ges:
                ges_val = m_ges.group(1).upper()

        etage_val = _extraire_nombre(_premier_val(item, "etage", "floor"), entier=True)
        if etage_val is None and desc_val:
            if re.search(r"\b(?:rdc|rez-de-chauss[eé]e)\b", desc_val, re.I):
                etage_val = 0
            else:
                m_et = re.search(r"\b(\d{1,2})\s*(?:er|e|ème)\s*[ée]tage\b", desc_val, re.I)
                if m_et:
                    etage_val = int(m_et.group(1))

        annee_val = _extraire_nombre(_premier_val(item, "annee_construction", "annee", "construction_year", "year"), entier=True)
        if annee_val is None and desc_val:
            m_an = re.search(r"\b(?:construi[te]?|ann[ée]e)\s*(?:en\s*)?(19\d{2}|20[0-2]\d)\b", desc_val, re.I)
            if m_an:
                annee_val = int(m_an.group(1))

        date_val = str(
            _premier_val(item, "date_publication", "date", "publication_date", "created_at", "date_creation", "creation_date", "mise_en_ligne")
            or ""
        ).strip()
        if not date_val or date_val.lower() in {"nan", "none"}:
            date_val = datetime.now().strftime("%Y-%m-%d")

        rows.append(
            {
                "id": ad_id,
                "titre": titre_val or f"Bien {ville_val or 'Poitiers'} {surface_val:.0f}m²",
                "ville": ville_val or "Poitiers",
                "code_postal": cp_val or "86000",
                "prix": prix_val,
                "surface_m2": surface_val,
                "pieces": _extraire_nombre(_premier_val(item, "pieces", "nb_pieces", "nombre_pieces", "rooms", "rooms_quantity"), entier=True),
                "nbr_chambres": _extraire_nombre(_premier_val(item, "nbr_chambres", "chambres", "nb_chambres", "nombre_chambres", "bedrooms", "bedrooms_quantity"), entier=True),
                "etage": etage_val,
                "annee_construction": annee_val,
                "dpe": dpe_val,
                "ges": ges_val,
                "quartier": quartier_val,
                "latitude": lat_val,
                "longitude": lon_val,
                "type_bien": _normaliser_type_bien(_premier_val(item, "type_bien", "type", "categorie", "property_type", "bien"), titre_val, f"{desc_val} {url_val}"),
                "description": desc_val,
                "date_publication": date_val,
                "url": url_val,
                "photos_urls": _extraire_photos(_premier_val(item, "photos_urls", "photos", "pictures", "images", "image", "photo")),
                "source": src_val,
            }
        )

    if not rows:
        return pd.DataFrame(columns=COLONNES_STANDARD)

    df_norm = pd.DataFrame(rows, columns=COLONNES_STANDARD)
    df_norm = df_norm.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)
    try:
        from src.ingestion import enrichissement as EnrichissementAnnonces
        df_norm = EnrichissementAnnonces.enrichir_dataframe(df_norm, max_requetes_portail=800, verbose=True)
    except Exception as e:
        print(f"[MoteurImmo] Enrichissement ignoré : {e}")
    return df_norm


def _lire_csv_flexible(csv_path: Path) -> pd.DataFrame:
    encodings = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
    last_err = None
    for enc in encodings:
        try:
            with open(csv_path, "r", encoding=enc, errors="strict") as f:
                sample = f.read(4096)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=";,\t|")
                sep = dialect.delimiter
            except csv.Error:
                sep = ";" if sample.count(";") >= sample.count(",") else ","
            df = pd.read_csv(csv_path, sep=sep, encoding=enc, dtype=str)
            if not df.empty or len(df.columns) > 1:
                return df
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Impossible de lire le CSV {csv_path}: {last_err}")


def normaliser_csv_moteurimmo(csv_path: Path | str, filtrer_poitiers_86: bool = True) -> pd.DataFrame:
    """Lit un export CSV MoteurImmo et le convertit au schéma standard du projet."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier CSV introuvable : {path}")
    df_raw = _lire_csv_flexible(path)
    if df_raw.empty:
        return pd.DataFrame(columns=COLONNES_STANDARD)
    return normaliser_items_moteurimmo(df_raw.to_dict(orient="records"), filtrer_poitiers_86=filtrer_poitiers_86)


def exporter_json_moteurimmo(df_moteur: pd.DataFrame, json_path: Path | str = JSON_EXPORT_PATH) -> Path:
    """Sauvegarde les annonces normalisées dans un fichier JSON (fusionné sans doublon d'id)."""
    cible = Path(json_path)
    existants: list[dict] = []
    if cible.exists():
        try:
            data_old = json.loads(cible.read_text(encoding="utf-8"))
            if isinstance(data_old, list):
                existants = data_old
        except Exception:
            existants = []

    nouveaux = df_moteur.where(pd.notna(df_moteur), None).to_dict(orient="records")
    par_id: dict[str, dict] = {}
    for row in existants + nouveaux:
        if isinstance(row, dict) and row.get("id"):
            par_id[str(row["id"])] = row

    liste_finale = list(par_id.values())
    cible.write_text(json.dumps(liste_finale, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[MoteurImmo -> JSON] {len(liste_finale)} annonces enregistrées dans {cible}")
    return cible


def charger_tous_exports_moteurimmo() -> pd.DataFrame:
    """Charge moteurimmo_annonces.json et tous les éventuels CSV dans data/moteurimmo/."""
    dfs = []
    if JSON_EXPORT_PATH.exists():
        try:
            data = json.loads(JSON_EXPORT_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                df_json = normaliser_items_moteurimmo(data, filtrer_poitiers_86=False)
                if not df_json.empty:
                    dfs.append(df_json)
        except Exception as e:
            print(f"[MoteurImmo] Erreur lecture {JSON_EXPORT_PATH} : {e}")

    if DOWNLOAD_DIR.exists():
        for csv_file in sorted(DOWNLOAD_DIR.glob("*.csv")):
            try:
                df_part = normaliser_csv_moteurimmo(csv_file)
                if not df_part.empty:
                    dfs.append(df_part)
            except Exception as e:
                print(f"[MoteurImmo] Erreur lecture {csv_file.name} : {e}")

    if not dfs:
        return pd.DataFrame(columns=COLONNES_STANDARD)
    df_all = pd.concat(dfs, ignore_index=True)
    return df_all.drop_duplicates(subset=["id"], keep="last").reset_index(drop=True)


def synchroniser_df_vers_bdd(df_moteur: pd.DataFrame, lancer_pipeline_ia: bool = True) -> dict:
    """Exporte le DataFrame en CSV + JSON, le synchronise dans PostgreSQL et met à jour les prédictions IA."""
    if df_moteur.empty:
        print("[MoteurImmo] Aucune annonce valide trouvée.")
        return {"importees": 0}

    df_moteur.to_csv(CSV_EXPORT_PATH, index=False, encoding="utf-8-sig", sep=";")
    print(f"[MoteurImmo -> CSV] {len(df_moteur)} annonces enregistrées dans {CSV_EXPORT_PATH}")
    exporter_json_moteurimmo(df_moteur, JSON_EXPORT_PATH)

    sources = sorted(df_moteur["source"].dropna().unique().tolist())
    print(f"\n[MoteurImmo] {len(df_moteur)} annonces normalisées (sources : {sources}).")

    _, _, stats = DB.synchroniser_offres_brutes(
        df_moteur,
        sources_scannees=sources,
        marquer_absentes_expirees=False,
    )
    print(
        f"[MoteurImmo -> BDD] +{stats['nouvelles']} nouvelles | "
        f"{stats['mises_a_jour']} déjà connues | "
        f"{stats['baisses_prix']} baisses de prix | "
        f"Total actives en BDD : {stats['total_actives']}"
    )

    if lancer_pipeline_ia:
        try:
            from src.ml import features as FeatureEngineering

            print("\n[MoteurImmo -> IA] Recalcul du Feature Engineering...")
            FeatureEngineering.construire_tables_features(verbose=True)
        except Exception as e:
            print(f"[MoteurImmo] Import BDD OK, mais Feature Engineering interrompu : {e}")

    return stats


def importer_csv_vers_bdd(
    csv_path: Path | str,
    lancer_pipeline_ia: bool = True,
    filtrer_poitiers_86: bool = True,
) -> dict:
    df_moteur = normaliser_csv_moteurimmo(csv_path, filtrer_poitiers_86=filtrer_poitiers_86)
    return synchroniser_df_vers_bdd(df_moteur, lancer_pipeline_ia=lancer_pipeline_ia)


def importer_json_vers_bdd(
    json_path: Path | str,
    lancer_pipeline_ia: bool = True,
    filtrer_poitiers_86: bool = True,
) -> dict:
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier JSON introuvable : {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = _extraire_annonces_depuis_payload(raw)
    if not isinstance(raw, list):
        raise ValueError(f"Format JSON non reconnu dans {path}")
    df_moteur = normaliser_items_moteurimmo(raw, filtrer_poitiers_86=filtrer_poitiers_86)
    return synchroniser_df_vers_bdd(df_moteur, lancer_pipeline_ia=lancer_pipeline_ia)


def _est_objet_annonce(obj: dict) -> bool:
    if not isinstance(obj, dict):
        return False
    # Rejeter les schémas de formulaire / filtres (ex: {"price": {"min": ..., "max": ...}})
    prix_raw = _premier_val(obj, "prix", "price", "montant", "prix_fai", "ad_price", "amount")
    surf_raw = _premier_val(obj, "surface", "surface_m2", "surface_habitable", "area", "superficie", "living_area", "habitable_surface", "size", "sqm")
    if prix_raw is None or isinstance(prix_raw, dict) or isinstance(surf_raw, dict):
        return False
    prix_num = _extraire_nombre(prix_raw, entier=False)
    if not prix_num or prix_num < 5000:
        return False
    cles = {_normaliser_nom_colonne(k) for k in obj.keys()}
    has_surf = surf_raw is not None
    has_meta = bool(cles & {"url", "link", "titre", "title", "ville", "city", "code_postal", "postal_code", "id", "origin", "source"})
    return has_surf or has_meta


def _extraire_annonces_depuis_payload(payload) -> list[dict]:
    trouves: list[dict] = []

    def _walk(node):
        if isinstance(node, list):
            for el in node:
                if isinstance(el, dict) and _est_objet_annonce(el):
                    trouves.append(el)
                elif isinstance(el, (dict, list)):
                    _walk(el)
        elif isinstance(node, dict):
            if _est_objet_annonce(node):
                trouves.append(node)
            else:
                for val in node.values():
                    if isinstance(val, (dict, list)):
                        _walk(val)

    _walk(payload)
    return trouves


async def _extraire_panneau_detail(page) -> dict:
    """Lit le panneau latéral / modal ouvert après clic (React Fiber/Props + icônes SVG/aria + DOM)."""
    script = """
    () => {
        const panel =
            document.querySelector('[role="dialog"], aside, [class*="detail" i], [class*="drawer" i], [class*="panel" i], [class*="modal" i]')
            || document.body;
        const txt = (panel.innerText || '').trim();
        const html = panel.innerHTML || '';

        // 0. Extraction directe depuis les props/fiber React (récupère le JSON brut même si affiché en icônes)
        let reactData = {};
        const extractFromObj = (rootObj) => {
            const seen = new Set();
            const stack = [{ obj: rootObj, depth: 0 }];
            while (stack.length > 0) {
                const { obj, depth } = stack.pop();
                if (!obj || typeof obj !== 'object' || depth > 7 || seen.has(obj)) continue;
                seen.add(obj);
                if (!Array.isArray(obj)) {
                    const keys = Object.keys(obj);
                    const hasCoords = ('lat' in obj && ('lng' in obj || 'lon' in obj)) || ('latitude' in obj && 'longitude' in obj);
                    if (hasCoords && !reactData.latitude) {
                        const la = parseFloat(obj.lat ?? obj.latitude);
                        const lo = parseFloat(obj.lng ?? obj.lon ?? obj.longitude);
                        if (la > 45 && la < 48 && lo > -1 && lo < 2) {
                            reactData.latitude = la;
                            reactData.longitude = lo;
                        }
                    }
                    if (('energyGrade' in obj || 'dpe' in obj) && !reactData.dpe) {
                        reactData.dpe = obj.energyGrade || obj.dpe;
                    }
                    if (('gasGrade' in obj || 'ges' in obj) && !reactData.ges) {
                        reactData.ges = obj.gasGrade || obj.ges;
                    }
                    if (('constructionYear' in obj || 'anneeConstruction' in obj) && !reactData.annee_construction) {
                        reactData.annee_construction = obj.constructionYear || obj.anneeConstruction;
                    }
                    if (('floor' in obj || 'etage' in obj) && reactData.etage === undefined) {
                        reactData.etage = obj.floor ?? obj.etage;
                    }
                    if (('roomsQuantity' in obj || 'rooms' in obj || 'nbPieces' in obj) && !reactData.pieces) {
                        reactData.pieces = obj.roomsQuantity ?? obj.rooms ?? obj.nbPieces;
                    }
                    if (('bedroomsQuantity' in obj || 'bedrooms' in obj || 'nbChambres' in obj) && !reactData.nbr_chambres) {
                        reactData.nbr_chambres = obj.bedroomsQuantity ?? obj.bedrooms ?? obj.nbChambres;
                    }
                    if (('pictures' in obj || 'images' in obj) && (!reactData.photos || reactData.photos.length === 0)) {
                        const arr = obj.pictures || obj.images;
                        if (Array.isArray(arr) && arr.length > 0) {
                            reactData.photos = arr.map(p => typeof p === 'string' ? p : (p?.url || p?.src || '')).filter(u => u.startsWith('http'));
                        }
                    }
                    if (('originUrl' in obj || 'url' in obj) && !reactData.url) {
                        const u = String(obj.originUrl || obj.url || '');
                        if (u.startsWith('http') && !/moteurimmo\\.fr/i.test(u)) reactData.url = u;
                    }
                    for (const k of keys) {
                        if (k === 'return' || k === '_owner' || k === 'stateNode') continue;
                        const v = obj[k];
                        if (v && typeof v === 'object') stack.push({ obj: v, depth: depth + 1 });
                    }
                } else {
                    for (let i = 0; i < Math.min(obj.length, 15); i++) {
                        if (obj[i] && typeof obj[i] === 'object') stack.push({ obj: obj[i], depth: depth + 1 });
                    }
                }
            }
        };
        for (const el of [panel, ...Array.from(panel.querySelectorAll('div, section, article')).slice(0, 25)]) {
            for (const k of Object.keys(el)) {
                if (k.startsWith('__reactProps$') || k.startsWith('__reactFiber$')) {
                    try { extractFromObj(el[k]); } catch {}
                }
            }
        }

        // 1. GPS dans les liens Google Maps / OpenStreetMap / attributs data-lat
        let lat = reactData.latitude ?? null, lon = reactData.longitude ?? null;
        if (!lat || !lon) {
            const gpsMatch = html.match(/(?:@|q=|ll=|center=)(46\\.\\d{3,}),\\s*(0\\.\\d{3,}|-?\\d+\\.\\d{3,})/i)
                || html.match(/"lat(?:itude)?"\\s*:\\s*(46\\.\\d{3,})[^}]*"(?:lon|lng|longitude)"\\s*:\\s*(0\\.\\d{3,})/i);
            if (gpsMatch) {
                lat = parseFloat(gpsMatch[1]);
                lon = parseFloat(gpsMatch[2]);
            } else {
                const latEl = panel.querySelector('[data-lat], [data-latitude]');
                const lonEl = panel.querySelector('[data-lng], [data-lon], [data-longitude]');
                if (latEl && lonEl) {
                    lat = parseFloat(latEl.getAttribute('data-lat') || latEl.getAttribute('data-latitude'));
                    lon = parseFloat(lonEl.getAttribute('data-lng') || lonEl.getAttribute('data-lon') || lonEl.getAttribute('data-longitude'));
                }
            }
        }

        // 2. DPE / GES via texte + aria-label / title / classes sur icônes & badges
        let iconHints = [];
        for (const ic of panel.querySelectorAll('[aria-label], [title], [data-tooltip], svg, [class*="dpe" i], [class*="ges" i]')) {
            const h = [ic.getAttribute('aria-label'), ic.getAttribute('title'), ic.getAttribute('data-tooltip'), ic.className?.baseVal ?? ic.className, ic.textContent].filter(Boolean).join(' ');
            if (h) iconHints.push(h);
        }
        const fullTxt = txt + '\\n' + iconHints.join(' ');
        const dpeM = fullTxt.match(/(?:DPE|Classe\\s+[ée]nergie|Consommation\\s+[ée]nerg[ée]tique|dpe[-_\\s]*)[:\\-\\n\\s]?([A-G])\\b/i);
        const gesM = fullTxt.match(/(?:GES|Gaz\\s+[àa]\\s+effet\\s+de\\s+serre|Émissions|ges[-_\\s]*)[:\\-\\n\\s]?([A-G])\\b/i);
        const anneeM = fullTxt.match(/(?:Ann[ée]e\\s+de\\s+construction|Construite?\\s+en)\\s*[:\\-]?\\s*(1[6-9]\\d{2}|20[0-2]\\d)/i);
        const etageM = fullTxt.match(/(?:[ÉE]tage|Au)\\s*[:\\-]?\\s*(\\d{1,2})\\s*(?:er|e|ème)?/i);
        const piecesM = fullTxt.match(/(\\d+)\\s*pi[eè]ces?/i) || txt.match(/m[²2]\\s+(\\d{1,2})\\b/);
        const chM = fullTxt.match(/(\\d+)\\s*chambres?/i) || txt.match(/m[²2]\\s+\\d{1,2}\\s+(\\d{1,2})\\b/);

        // 3. Lien direct portail (hors Google Maps / MoteurImmo interne)
        const links = Array.from(panel.querySelectorAll('a[href]')).map(a => a.href || '').filter(Boolean);
        const adUrl = reactData.url || links.find(u => !/google\\.|maps\\.|openstreetmap|moteurimmo\\.fr/i.test(u)) || '';

        // 4. Photos réelles du bien (img src + data-src + background-image)
        const imgs = Array.from(panel.querySelectorAll('img'))
            .flatMap(i => [i.src, i.getAttribute('data-src')])
            .filter(src => src && src.startsWith('http') && !/moteurimmo\\.fr\\/images\\//i.test(src) && !/logo|icon|avatar|map|tile|openstreetmap/i.test(src));
        for (const bgEl of panel.querySelectorAll('[style*="background-image"]')) {
            const m = (bgEl.style.backgroundImage || '').match(/url\\(["']?(https?:[^"')]+)["']?\\)/i);
            if (m && !/moteurimmo\\.fr\\/images\\//i.test(m[1])) imgs.push(m[1]);
        }
        const finalPhotos = (reactData.photos && reactData.photos.length > 0) ? reactData.photos : Array.from(new Set(imgs)).slice(0, 6);

        return {
            latitude: lat,
            longitude: lon,
            dpe: reactData.dpe || (dpeM ? dpeM[1].toUpperCase() : null),
            ges: reactData.ges || (gesM ? gesM[1].toUpperCase() : null),
            annee_construction: reactData.annee_construction || (anneeM ? anneeM[1] : null),
            etage: reactData.etage ?? (etageM ? etageM[1] : null),
            pieces: reactData.pieces ?? (piecesM ? piecesM[1] : null),
            nbr_chambres: reactData.nbr_chambres ?? (chM ? chM[1] : null),
            url: adUrl,
            photos: finalPhotos,
            description: txt.replace(/\\s+/g, ' '),
        };
    }
    """
    try:
        return await page.evaluate(script)
    except Exception:
        return {}


async def _cliquer_et_extraire_cartes_visibles(page, vus_keys: set[str], annonces_capturees: list[dict]) -> int:
    """Identifie les nouvelles cartes visibles, clique sur chacune pour charger sa fiche complète et fusionne."""
    marque_script = """
    () => {
        const priceRe = /(\\d[\\d\\s\\u202f\\u00a0.]{1,12})\\s*€/;
        const surfRe = /(\\d+(?:[.,]\\d+)?)\\s*m[²2]/i;
        const candidates = document.querySelectorAll('article, li, [class*="card" i], [class*="ad" i], [class*="listing" i], [class*="result" i], div');
        let idx = 0;
        const cartes = [];
        for (const el of candidates) {
            if (el.getAttribute('data-mi-done') === '1') continue;
            const rect = el.getBoundingClientRect();
            if (rect.width < 150 || rect.height < 60 || rect.bottom < 0 || rect.top > window.innerHeight) continue;
            const txt = (el.innerText || '').trim();
            if (!txt || txt.length < 15 || txt.length > 2200) continue;
            if (!priceRe.test(txt) || !surfRe.test(txt)) continue;
            const euroCount = (txt.match(/€/g) || []).length;
            if (euroCount > 6) continue;
            // Éviter de marquer à la fois parent et enfant
            if (el.querySelector('[data-mi-cid]')) continue;
            idx += 1;
            const cid = 'mi_c_' + Date.now() + '_' + idx;
            el.setAttribute('data-mi-cid', cid);
            el.setAttribute('data-mi-done', '1');
            const prixM = txt.match(priceRe);
            const surfM = txt.match(surfRe);
            const piecesM = txt.match(/(\\d+)\\s*p(?:i[eè]ces?)?\\b/i);
            const chM = txt.match(/(\\d+)\\s*ch(?:ambres?)?\\b/i);
            const cpM = txt.match(/\\b(86\\d{3})\\b/);
            const allLinks = Array.from(el.querySelectorAll('a[href]')).map(a => a.href || '').filter(Boolean);
            const adLink = allLinks.find(u => !/google\\.|maps\\.|openstreetmap/i.test(u)) || '';
            const imgs = Array.from(el.querySelectorAll('img')).map(i => i.src).filter(s => s && !/moteurimmo\\.fr\\/images\\//i.test(s) && !/logo|icon|avatar/i.test(s));
            cartes.push({
                cid,
                base: {
                    titre: txt.split('\\n').map(l => l.trim()).filter(Boolean).slice(0, 2).join(' - ').slice(0, 140),
                    prix: prixM ? prixM[1] : null,
                    surface: surfM ? surfM[1] : null,
                    pieces: piecesM ? piecesM[1] : null,
                    nbr_chambres: chM ? chM[1] : null,
                    code_postal: cpM ? cpM[1] : '86000',
                    ville: 'Poitiers',
                    url: adLink,
                    photos: imgs,
                    description: txt.replace(/\\s+/g, ' ').slice(0, 500),
                    source: 'moteurimmo'
                }
            });
        }
        return cartes;
    }
    """
    try:
        cartes = await page.evaluate(marque_script)
    except Exception:
        return 0

    nouveaux = 0
    for c in cartes:
        base = c["base"]
        cle_it = f"{base.get('url', '')}|{base.get('prix', '')}|{base.get('surface', '')}|{base.get('titre', '')[:40]}"
        if cle_it in vus_keys:
            continue
        vus_keys.add(cle_it)

        # Cliquer sur la carte (en évitant les liens externes) et attendre que le panneau change réellement
        try:
            loc = page.locator(f"[data-mi-cid='{c['cid']}']").first
            if await loc.count() > 0 and await loc.is_visible():
                prev_panel_txt = await page.evaluate(
                    "() => (document.querySelector('[role=\"dialog\"], aside, [class*=\"detail\" i]')?.innerText || '').slice(0, 160)"
                )
                await loc.click(position={"x": 40, "y": 25}, timeout=2000)
                for _ in range(8):
                    await page.wait_for_timeout(180)
                    cur_panel_txt = await page.evaluate(
                        "() => (document.querySelector('[role=\"dialog\"], aside, [class*=\"detail\" i]')?.innerText || '').slice(0, 160)"
                    )
                    if cur_panel_txt and cur_panel_txt != prev_panel_txt:
                        break
                detail = await _extraire_panneau_detail(page)
                # Si un onglet "Adresse" est présent (Pos. GPS connue), cliquer dessus pour extraire les coords GPS
                try:
                    addr_tab = page.locator("button:text-is('Adresse'), a:text-is('Adresse'), [role='tab']:has-text('Adresse')").first
                    if await addr_tab.count() > 0 and await addr_tab.is_visible():
                        await addr_tab.click(timeout=1200)
                        await page.wait_for_timeout(300)
                        detail_addr = await _extraire_panneau_detail(page)
                        if detail_addr.get("latitude") and detail_addr.get("longitude"):
                            detail["latitude"] = detail_addr["latitude"]
                            detail["longitude"] = detail_addr["longitude"]
                except Exception:
                    pass
                for k, v in detail.items():
                    if v not in (None, "", []):
                        if k == "description" and len(str(v)) <= len(str(base.get("description", ""))):
                            continue
                        base[k] = v
                try:
                    await page.keyboard.press("Escape")
                except Exception:
                    pass
        except Exception:
            pass

        if _est_objet_annonce(base):
            annonces_capturees.append(base)
            nouveaux += 1

    return nouveaux


async def _extraire_annonces_dom(page) -> list[dict]:
    """Extraction depuis le DOM (toutes frames + cartes avec ou sans <a>) et JSON-LD."""
    script = """
    () => {
        const out = [];
        // 1. JSON-LD éventuels
        for (const s of document.querySelectorAll('script[type="application/ld+json"]')) {
            try { out.push(JSON.parse(s.textContent)); } catch {}
        }
        // 2. Cartes d'annonces (avec ou sans <a>) contenant un prix (€) et une surface (m² ou m2)
        const seen = new Set();
        const priceRe = /(\\d[\\d\\s\\u202f\\u00a0.]{1,12})\\s*€/;
        const surfRe = /(\\d+(?:[.,]\\d+)?)\\s*m[²2]/i;
        const candidates = document.querySelectorAll('article, li, [class*="card" i], [class*="ad" i], [class*="listing" i], [class*="result" i], a[href], div');
        for (const el of candidates) {
            const txt = (el.innerText || '').trim();
            if (!txt || txt.length < 15 || txt.length > 2200) continue;
            if (!priceRe.test(txt) || !surfRe.test(txt)) continue;
            // Éviter les gros conteneurs parents qui englobent plusieurs cartes (€ apparaît de nombreuses fois)
            const euroCount = (txt.match(/€/g) || []).length;
            if (euroCount > 6) continue;
            const allLinks = Array.from(el.querySelectorAll('a[href]')).map(a => a.href || '').filter(Boolean);
            const adLink = allLinks.find(u => !/google\\.|maps\\.|openstreetmap/i.test(u)) || (el.matches('a[href]') ? (el.href || '') : (allLinks[0] || ''));
            const href = adLink;
            const key = href || txt.slice(0, 140);
            if (seen.has(key)) continue;
            seen.add(key);
            const prixM = txt.match(priceRe);
            const surfM = txt.match(surfRe);
            const piecesM = txt.match(/(\\d+)\\s*p(?:i[eè]ces?)?\\b/i);
            const chM = txt.match(/(\\d+)\\s*ch(?:ambres?)?\\b/i);
            const cpM = txt.match(/\\b(86\\d{3})\\b/);
            const imgs = Array.from(el.querySelectorAll('img'))
                .map(i => i.src)
                .filter(src => src && !/moteurimmo\\.fr\\/images\\//i.test(src) && !/logo|icon|avatar/i.test(src));
            out.push({
                titre: txt.split('\\n').map(l => l.trim()).filter(Boolean).slice(0, 2).join(' - ').slice(0, 140),
                prix: prixM ? prixM[1] : null,
                surface: surfM ? surfM[1] : null,
                pieces: piecesM ? piecesM[1] : null,
                nbr_chambres: chM ? chM[1] : null,
                code_postal: cpM ? cpM[1] : '86000',
                ville: 'Poitiers',
                url: href,
                photos: imgs,
                description: txt.replace(/\\s+/g, ' ').slice(0, 500),
                source: 'moteurimmo'
            });
        }
        return out;
    }
    """
    resultats = []
    for frame in page.frames:
        try:
            raw = await frame.evaluate(script)
            if not isinstance(raw, list):
                continue
            for el in raw:
                if isinstance(el, dict) and _est_objet_annonce(el):
                    resultats.append(el)
                elif isinstance(el, (dict, list)):
                    resultats.extend(_extraire_annonces_depuis_payload(el))
        except Exception:
            continue
    return resultats


async def exporter_moteurimmo_interactif(
    lancer_pipeline_ia: bool = True,
    filtrer_poitiers_86: bool = True,
    max_scrolls_override: int | None = None,
) -> Path | None:
    """
    Ouvre MoteurImmo avec profil persistant, capture automatiquement les annonces JSON
    chargées pendant ta navigation/recherche sur Poitiers (et en secours le DOM),
    puis exporte tout dans moteurimmo_annonces.json et synchronise PostgreSQL.
    """
    annonces_capturees: list[dict] = []

    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=False,
            accept_downloads=True,
        )

        page = context.pages[0] if context.pages else await context.new_page()

        async def _attacher_listeners(p_page):
            async def _on_response(response):
                try:
                    u_low = response.url.lower()
                    if any(skip in u_low for skip in ("google", "facebook", "sentry", "intercom", "stripe", "mapbox", "openstreetmap", ".png", ".jpg", ".css", ".js")):
                        return
                    ctype = (response.headers.get("content-type") or "").lower()
                    if "json" not in ctype and "text/plain" not in ctype:
                        return
                    data = await response.json()
                    trouves = _extraire_annonces_depuis_payload(data)
                    if trouves:
                        annonces_capturees.extend(trouves)
                        print(f"[MoteurImmo] +{len(trouves)} annonces API capturées (brut cumulé : {len(annonces_capturees)})")
                except Exception:
                    pass

            p_page.on("response", _on_response)

        await _attacher_listeners(page)
        context.on("page", lambda new_p: asyncio.create_task(_attacher_listeners(new_p)))

        url_depart = "https://moteurimmo.fr/"
        if LAST_URL_PATH.exists():
            saved_url = LAST_URL_PATH.read_text(encoding="utf-8").strip()
            if saved_url.startswith("https://moteurimmo.fr"):
                url_depart = saved_url
                print(f"[MoteurImmo] Reprise de ta dernière recherche enregistrée : {url_depart}")

        await page.goto(url_depart, wait_until="domcontentloaded")

        print("\n1. Connecte-toi manuellement à MoteurImmo dans Chrome (ta session est conservée).")
        print("2. Affiche ta recherche immobilière sur Poitiers (page 1 des résultats).")

        mode = input("\nAppuie sur Entrée pour tout scraper automatiquement (toutes les pages) ou tape '1' pour cette page seule : ").strip()

        # Choisir l'onglet contenant les résultats (éviter /connexion si un autre onglet est ouvert)
        active_page = page
        best_score = -1
        for p_cand in context.pages:
            try:
                u = p_cand.url.lower()
                nb_eur = await p_cand.evaluate("() => (document.body?.innerText.match(/€/g) || []).length")
                score = nb_eur + (0 if "/connexion" in u else 50)
                if score > best_score:
                    best_score = score
                    active_page = p_cand
            except Exception:
                continue

        print("\nURL actuelle :", active_page.url)
        print("Titre :", await active_page.title())
        if "moteurimmo.fr" in active_page.url and "/connexion" not in active_page.url:
            LAST_URL_PATH.write_text(active_page.url, encoding="utf-8")

        max_steps = max_scrolls_override if max_scrolls_override else (1 if mode == "1" else 250)
        box = await active_page.evaluate("() => ({w: window.innerWidth, h: window.innerHeight})")
        cx, cy = int(box["w"] * 0.5), int(box["h"] * 0.5)
        await active_page.mouse.move(cx, cy)

        vus_keys: set[str] = set()
        stagnation = 0
        for step in range(1, max_steps + 1):
            await active_page.wait_for_timeout(650)
            nouveaux_step = await _cliquer_et_extraire_cartes_visibles(active_page, vus_keys, annonces_capturees)
            if nouveaux_step == 0:
                dom_items = await _extraire_annonces_dom(active_page)
                for it in dom_items:
                    cle_it = f"{it.get('url', '')}|{it.get('prix', '')}|{it.get('surface', '')}|{it.get('titre', '')[:40]}"
                    if cle_it not in vus_keys:
                        vus_keys.add(cle_it)
                        annonces_capturees.append(it)
                        nouveaux_step += 1

            if nouveaux_step > 0:
                stagnation = 0
                print(f"[MoteurImmo] Scroll {step} : +{nouveaux_step} nouvelles fiches détaillées (total uniques : {len(vus_keys)})")
            else:
                stagnation += 1

            if step >= max_steps or stagnation >= 5:
                break

            # Cliquer sur un éventuel bouton "Charger plus" / "Suivant" s'il est visible, sinon scroller à la molette
            load_more = active_page.locator(
                "button:has-text('Charger plus'), button:has-text('Voir plus'), "
                "a[rel='next'], button:has-text('Suivant'), a:has-text('Suivant')"
            ).first
            try:
                if await load_more.count() > 0 and await load_more.is_visible() and not await load_more.is_disabled():
                    await load_more.click()
                    await active_page.wait_for_timeout(1200)
                else:
                    await active_page.mouse.wheel(0, 950)
            except Exception:
                await active_page.mouse.wheel(0, 950)

        if not annonces_capturees:
            try:
                debug_txt = await active_page.evaluate("() => document.body ? document.body.innerText.slice(0, 4000) : ''")
                debug_path = DOWNLOAD_DIR / "debug_dom.txt"
                debug_path.write_text(debug_txt or "", encoding="utf-8")
                print(f"[MoteurImmo] Extrait DOM de diagnostic sauvegardé dans {debug_path}")
            except Exception:
                pass

        await context.close()

    if not annonces_capturees:
        print("[MoteurImmo] Aucune annonce capturée. Vérifie que les résultats étaient bien affichés.")
        return None

    raw_backup = DOWNLOAD_DIR / f"capture_brute_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    raw_backup.write_text(json.dumps(annonces_capturees, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[MoteurImmo] Capture brute sauvegardée : {raw_backup}")

    df_moteur = normaliser_items_moteurimmo(annonces_capturees, filtrer_poitiers_86=filtrer_poitiers_86)
    synchroniser_df_vers_bdd(df_moteur, lancer_pipeline_ia=lancer_pipeline_ia)
    return JSON_EXPORT_PATH


def main():
    parser = argparse.ArgumentParser(
        description="Capture MoteurImmo (Poitiers) -> Export JSON -> Synchronisation PostgreSQL"
    )
    parser.add_argument(
        "--import-json",
        type=str,
        default=None,
        help="Importer directement un fichier JSON MoteurImmo.",
    )
    parser.add_argument(
        "--import-csv",
        type=str,
        default=None,
        help="Importer directement un fichier CSV MoteurImmo (et l'exporter aussi en JSON).",
    )
    parser.add_argument(
        "--no-pipeline",
        action="store_true",
        help="Ne pas relancer FeatureEngineering et Prediction après l'import BDD.",
    )
    parser.add_argument(
        "--sans-filtre-poitiers",
        action="store_true",
        help="Importer toutes les annonces sans filtrer sur le 86 / Grand Poitiers.",
    )
    parser.add_argument(
        "--max-scrolls",
        type=int,
        default=None,
        help="Limiter le nombre de scrolls (ex: --max-scrolls 3 pour tester rapidement).",
    )
    args = parser.parse_args()

    lancer_ia = not args.no_pipeline
    filtrer = not args.sans_filtre_poitiers

    if args.import_json:
        importer_json_vers_bdd(
            args.import_json,
            lancer_pipeline_ia=lancer_ia,
            filtrer_poitiers_86=filtrer,
        )
    elif args.import_csv:
        importer_csv_vers_bdd(
            args.import_csv,
            lancer_pipeline_ia=lancer_ia,
            filtrer_poitiers_86=filtrer,
        )
    else:
        asyncio.run(
            exporter_moteurimmo_interactif(
                lancer_pipeline_ia=lancer_ia,
                filtrer_poitiers_86=filtrer,
                max_scrolls_override=args.max_scrolls,
            )
        )


if __name__ == "__main__":
    main()
