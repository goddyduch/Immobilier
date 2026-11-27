
"""
Récupération des annonces Bien'ici avec requests (API JSON interne, non officielle).

Usage (sortie JSON par défaut, CSV si --sortie finit par .csv) :
    python Scrapping_v2.py --ville Poitiers --prix-max 350000 --surface-min 40
    python Scrapping_v2.py --ville Poitiers --sortie resultats.csv
    python Scrapping_v2.py --ville Paris --type location --prix-max 1500 --max-annonces 200
    python Scrapping_v2.py --ville Poitiers --debug   # affiche les clés d'une annonce brute
"""

import argparse
import csv
import json
import random
import time

import requests

BASE_URL = "https://www.bienici.com"
SUGGEST_URL = "https://res.bienici.com/suggest.json"
ADS_URL = f"{BASE_URL}/realEstateAds.json"

GRAND_POITIERS_COMMUNES = [
    "Poitiers 86000",
    "Buxerolles 86180",
    "Saint-Benoît 86280",
    "Migné-Auxances 86440",
    "Chasseneuil-du-Poitou 86360",
    "Jaunay-Marigny 86130",
    "Biard 86580",
    "Vouneuil-sous-Biard 86580",
    "Fontaine-le-Comte 86240",
    "Montamisé 86360",
    "Smarves 86240",
    "Ligugé 86240",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "fr-FR,fr;q=0.9",
    "Referer": BASE_URL + "/",
}

session = requests.Session()
session.headers.update(HEADERS)


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def en_nombre(valeur):
    """
    Convertit une valeur de l'API en nombre (float) ou None.

    L'API renvoie parfois une liste (ex. [45.0, 62.5] pour un programme neuf
    qui regroupe plusieurs lots) : dans ce cas on garde la plus petite valeur.
    """
    if valeur is None:
        return None
    if isinstance(valeur, (list, tuple)):
        nombres = [n for n in (en_nombre(v) for v in valeur) if n is not None]
        return min(nombres) if nombres else None
    try:
        return float(valeur)
    except (TypeError, ValueError):
        return None


def en_entier(valeur):
    """Comme en_nombre, mais renvoie un int (ou None)."""
    n = en_nombre(valeur)
    return int(n) if n is not None else None


def premier_non_nul(a, *cles):
    """Retourne la première valeur non nulle parmi plusieurs clés possibles."""
    for cle in cles:
        v = a.get(cle)
        if v is not None and v != "":
            return v
    return None


def lettre_classe(valeur):
    """Normalise une classe énergétique (DPE/GES) en lettre A-G, sinon None."""
    if isinstance(valeur, str) and valeur.strip().upper() in list("ABCDEFG"):
        return valeur.strip().upper()
    return None


def extraire_position(a):
    """
    Retourne (latitude, longitude). Bien'ici fournit soit 'position', soit
    'blurInfo.position' (position floutée) quand l'adresse exacte est masquée.
    """
    pos = a.get("position") or (a.get("blurInfo") or {}).get("position") or {}
    lat = en_nombre(pos.get("lat") if "lat" in pos else pos.get("latitude"))
    lon = en_nombre(pos.get("lon") if "lon" in pos else pos.get("longitude"))
    return lat, lon


def extraire_quartier(a):
    """Quartier : 'district' est un objet (libelle/name) ou parfois une chaîne."""
    d = a.get("district")
    if isinstance(d, dict):
        return d.get("libelle") or d.get("name")
    if isinstance(d, str):
        return d
    return a.get("neighborhood")


class ScrapingBlockedError(RuntimeError):
    """Levée lorsque le site bloque les requêtes (403/429 persistant ou CAPTCHA)."""


def get_json(url, params=None, essais=3):
    """GET avec tentatives et détection de blocage (403 / 429 persistant)."""
    for i in range(essais):
        try:
            r = session.get(url, params=params, timeout=15)
            if r.status_code in (403, 429):
                if i == essais - 1:
                    raise ScrapingBlockedError(f"HTTP {r.status_code} sur {url}")
                time.sleep(5 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except ScrapingBlockedError:
            raise
        except requests.RequestException as e:
            if i == essais - 1:
                raise
            print(f"Erreur ({e}), nouvelle tentative...")
            time.sleep(2 * (i + 1))


# ---------------------------------------------------------------------------
# Collecte
# ---------------------------------------------------------------------------

def trouver_zone(ville):
    """Retourne les zoneIds correspondant à une ville (via la route de suggestion)."""
    suggestions = get_json(SUGGEST_URL, params={"q": ville})
    if not suggestions:
        raise ValueError(f"Aucune zone trouvée pour « {ville} »")
    return suggestions[0]["zoneIds"]


def trouver_zone_territoire(code_territoire: str) -> list[str]:
    """Retourne les zoneIds d'une ville-centre (cache PostgreSQL puis `suggest.json`)."""
    from src.ingestion import db as DB
    from src.ingestion.territoires import get_territoire

    cfg = get_territoire(code_territoire)
    if not cfg:
        raise ValueError(f"Territoire inconnu : {code_territoire}")

    if DB.db_disponible():
        try:
            with DB.engine.connect() as conn:
                cached = conn.execute(
                    DB.text("SELECT bienici_zone_ids FROM ref_territoires WHERE code_territoire = :c"),
                    {"c": cfg.code_territoire},
                ).scalar()
                if cached and isinstance(cached, list) and len(cached) > 0:
                    return cached
        except Exception:
            pass

    zone_ids = trouver_zone(f"{cfg.nom} {cfg.codes_postaux[0]}")
    if DB.db_disponible():
        try:
            with DB.engine.begin() as conn:
                conn.execute(
                    DB.text("UPDATE ref_territoires SET bienici_zone_ids = CAST(:z AS JSONB), updated_at = NOW() WHERE code_territoire = :c"),
                    {"z": json.dumps(zone_ids), "c": cfg.code_territoire},
                )
        except Exception:
            pass
    return zone_ids


def recuperer_annonces(ville=None, type_operation="buy", prix_max=None, prix_min=None,
                       surface_min=None, surface_max=None, max_annonces=500,
                       nouvelles_seulement=False, zone_ids=None, code_territoire=None):
    zone_ids = zone_ids or trouver_zone(ville)
    taille_page = 50
    annonces, vues = [], set()
    debut = 0

    ids_connus = set()
    if nouvelles_seulement:
        try:
            from src.ingestion.db import get_ids_connus
            ids_connus = get_ids_connus(source="bienici", code_territoire=code_territoire)
            print(f"{len(ids_connus)} annonces Bien'ici déjà connues en base ({code_territoire or 'global'}).")
        except Exception:
            ids_connus = set()

    while len(annonces) < max_annonces:
        filtres = {
            "size": taille_page,
            "from": debut,
            "showAllModels": False,
            "filterType": type_operation,  # "buy" ou "rent"
            "propertyType": ["flat", "house"],
            "page": debut // taille_page + 1,
            "sortBy": "publicationDate",
            "sortOrder": "desc",
            "onTheMarket": [True],
            "zoneIdsByTypes": {"zoneIds": zone_ids},
        }
        if prix_max:
            filtres["maxPrice"] = prix_max
        if prix_min:
            filtres["minPrice"] = prix_min
        if surface_min:
            filtres["minArea"] = surface_min
        if surface_max:
            filtres["maxArea"] = surface_max

        data = get_json(ADS_URL, params={"filters": json.dumps(filtres)})
        page = data.get("realEstateAds", [])
        if not page:
            break

        nouvelles_sur_page = 0
        for a in page:
            ad_id = str(a.get("id"))
            if ad_id in vues:  # dédoublonnage
                continue
            vues.add(ad_id)
            if nouvelles_seulement and ad_id in ids_connus:
                continue
            annonces.append(a)
            nouvelles_sur_page += 1

        total = data.get("total", 0)
        debut += taille_page
        print(f"{len(annonces)} annonces retenues (+{nouvelles_sur_page} sur cette page / total marché : {total})")
        if nouvelles_seulement and nouvelles_sur_page == 0:
            print("Toutes les annonces de cette page sont déjà en base -> arrêt anticipé.")
            break
        if debut >= total:
            break
        time.sleep(random.uniform(2, 4))  # pause pour ne pas surcharger le site

    return annonces[:max_annonces]


def recuperer_annonces_territoire(
    code_territoire: str,
    type_operation: str = "buy",
    max_annonces: int = 800,
    nouvelles_seulement: bool = False,
) -> tuple[list[dict], bool]:
    """
    Récupère les annonces d'une des 12 villes-centres uniquement (avec circuit-breaker 3h).
    Retourne `(annonces, scrape_complet)`.
    """
    from src.ingestion import db as DB
    from src.ingestion.territoires import get_territoire

    cfg = get_territoire(code_territoire)
    if not cfg:
        raise ValueError(f"Territoire inconnu : {code_territoire}")

    if DB.est_en_cooldown("bienici", cfg.code_territoire):
        print(f"[BienIci] {cfg.nom} en cooldown (3h) -> passage ignoré.")
        return [], False

    try:
        zone_ids = trouver_zone_territoire(cfg.code_territoire)
        lot = recuperer_annonces(
            ville=cfg.nom,
            type_operation=type_operation,
            max_annonces=max_annonces,
            nouvelles_seulement=nouvelles_seulement,
            zone_ids=zone_ids,
            code_territoire=cfg.code_territoire,
        )
        cps_valides = set(cfg.codes_postaux)
        toutes, vues = [], set()
        for a in lot:
            cp = str(a.get("postalCode") or "").strip()[:5]
            if cp and cp not in cps_valides:
                continue
            ad_id = str(a.get("id"))
            if ad_id not in vues:
                vues.add(ad_id)
                toutes.append(a)
        scrape_complet = not nouvelles_seulement
        DB.enregistrer_succes_scrape("bienici", cfg.code_territoire, len(toutes), scrape_complet=scrape_complet)
        return toutes[:max_annonces], scrape_complet
    except ScrapingBlockedError as e:
        DB.enregistrer_blocage_scrape("bienici", cfg.code_territoire, str(e), cooldown_hours=3)
        return [], False
    except Exception as e:
        print(f"[BienIci {cfg.nom}] Erreur réseau/API : {e}")
        return [], False


def recuperer_annonces_grand_poitiers(
    type_operation="buy",
    max_par_commune=250,
    max_total=800,
    nouvelles_seulement=False,
):
    """Compatibilité V1 : récupère uniquement Poitiers ville-centre (`86000`)."""
    annonces, _ = recuperer_annonces_territoire(
        code_territoire="poitiers",
        type_operation=type_operation,
        max_annonces=max_total,
        nouvelles_seulement=nouvelles_seulement,
    )
    return annonces


def extraire_photos_urls(a, max_photos=5):
    """Extrait jusqu'à `max_photos` URLs de photos depuis une annonce brute Bien'ici."""
    photos = a.get("photos") or []
    urls = []
    for p in photos:
        u = p.get("url") or p.get("url_photo") if isinstance(p, dict) else (p if isinstance(p, str) else None)
        if u and isinstance(u, str) and u.startswith("http"):
            urls.append(u)
        if len(urls) >= max_photos:
            break
    return "|".join(urls)


def simplifier(a):
    """Garde uniquement les champs utiles, avec prix et surface en nombres."""
    lat, lon = extraire_position(a)
    return {
        "id": a.get("id"),
        "titre": a.get("title"),
        "ville": a.get("city"),
        "code_postal": a.get("postalCode"),
        "prix": en_nombre(a.get("price")),
        "surface_m2": en_nombre(a.get("surfaceArea")),
        "pieces": en_nombre(a.get("roomsQuantity")),
        "nbr_chambres": en_entier(a.get("bedroomsQuantity")),
        "etage": en_entier(a.get("floor")),
        "annee_construction": en_entier(
            premier_non_nul(a, "yearOfConstruction", "buildingYear", "constructionYear")
        ),
        "dpe": lettre_classe(a.get("energyClassification")),
        "ges": lettre_classe(a.get("greenhouseGazClassification")),
        "quartier": extraire_quartier(a),
        "latitude": lat,
        "longitude": lon,
        "type_bien": a.get("propertyType"),
        "description": (a.get("description") or "").replace("\n", " ").strip(),
        "date_publication": a.get("publicationDate"),
        "url": f"{BASE_URL}/annonce/{a.get('id')}" if a.get("id") else None,
        "photos_urls": extraire_photos_urls(a, max_photos=5),
    }


# ---------------------------------------------------------------------------
# Filtrage local
# ---------------------------------------------------------------------------

def filtrer(annonces, ville=None, surface_min=None, surface_max=None,
            prix_min=None, prix_max=None):
    """
    Filtre les annonces simplifiées. Si un filtre est demandé et que la
    valeur de l'annonce est inconnue (None), l'annonce est écartée.
    """
    resultats = []
    for a in annonces:
        if ville and (a["ville"] or "").lower() != ville.lower():
            continue

        surface, prix = a["surface_m2"], a["prix"]

        if surface_min is not None and (surface is None or surface < surface_min):
            continue
        if surface_max is not None and (surface is None or surface > surface_max):
            continue
        if prix_min is not None and (prix is None or prix < prix_min):
            continue
        if prix_max is not None and (prix is None or prix > prix_max):
            continue

        resultats.append(a)
    return resultats


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def exporter_csv(annonces, chemin):
    if not annonces:
        print("Rien à exporter.")
        return
    import os
    os.makedirs("csv", exist_ok=True)
    if not chemin.startswith("csv/") and "/" not in chemin:
        chemin = os.path.join("csv", chemin)
    with open(chemin, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(annonces[0].keys()), delimiter=";")
        writer.writeheader()
        writer.writerows(annonces)
    print(f"{len(annonces)} annonces enregistrées dans {chemin}")


def exporter_json(annonces, chemin):
    """Écrit les annonces dans un fichier JSON (liste d'objets, accents conservés)."""
    if not annonces:
        print("Rien à exporter.")
        return
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(annonces, f, ensure_ascii=False, indent=2)
    print(f"{len(annonces)} annonces enregistrées dans {chemin}")


def exporter(annonces, chemin):
    """Choisit le format selon l'extension du fichier de sortie (.json ou .csv)."""
    if chemin.lower().endswith(".csv"):
        exporter_csv(annonces, chemin)
    else:
        exporter_json(annonces, chemin)


def main():
    p = argparse.ArgumentParser(description="Annonces Bien'ici")
    p.add_argument("--ville", required=True)
    p.add_argument("--type", choices=["achat", "location"], default="achat")
    p.add_argument("--prix-min", type=int)
    p.add_argument("--prix-max", type=int)
    p.add_argument("--surface-min", type=float)
    p.add_argument("--surface-max", type=float)
    p.add_argument("--max-annonces", type=int, default=200)
    p.add_argument("--sortie", default="annonces_bienici.json",
                   help="fichier de sortie : .json (défaut) ou .csv")
    p.add_argument("--debug", action="store_true",
                   help="affiche les clés de la 1re annonce brute (pour vérifier les noms de champs)")
    args = p.parse_args()

    brutes = recuperer_annonces(
        ville=args.ville,
        type_operation="buy" if args.type == "achat" else "rent",
        prix_min=args.prix_min,
        prix_max=args.prix_max,
        surface_min=args.surface_min,
        surface_max=args.surface_max,
        max_annonces=args.max_annonces,
    )

    if args.debug and brutes:
        print("Clés de la 1re annonce brute :", sorted(brutes[0].keys()))

    annonces = [simplifier(a) for a in brutes]
    annonces = filtrer(
        annonces,
        surface_min=args.surface_min, surface_max=args.surface_max,
        prix_min=args.prix_min, prix_max=args.prix_max,
    )

    for a in annonces[:5]:
        print(f"- {a['ville']} | {a['prix']} € | {a['surface_m2']} m² | "
              f"DPE {a['dpe']} | étage {a['etage']} | {a['quartier']} | {a['titre']}")

    exporter(annonces, args.sortie)


if __name__ == "__main__":
    main()