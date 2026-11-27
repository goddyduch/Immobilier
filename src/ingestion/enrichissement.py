"""
EnrichissementAnnonces.py — Enrichissement multi-méthodes 100% gratuit (Pages publiques + API BAN / Géoplateforme).

Règles strictes :
  - Aucune utilisation de l'API payante MoteurImmo.
  - Aucun contournement de CAPTCHA / protection anti-bot (si HTTP 403/429, on passe proprement).
  - Distinction stricte de la localisation :
      * "exacte"          : GPS issu du portail/JSON-LD/JS ou géocodage BAN d'une rue/adresse explicite (score >= 0.55)
      * "approximative"   : quartier, secteur ou repère géocodé (ex: Gibauderie, Gare, CHU)
      * "non_disponible"  : seule la commune (Poitiers 86000) est connue, sans rue ni quartier ni GPS
"""

from __future__ import annotations

import html as html_lib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE_PATH = Path("data/enrichissement_cache.json")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9",
}

DOMAINES_ANTI_BOT = {"leboncoin.fr", "seloger.com", "immobilier.lefigaro.fr", "logic-immo.com", "pap.fr"}

SECTEURS_POITIERS = {
    "centre-ville": (46.5802, 0.3404, "Centre-ville"),
    "centre ville": (46.5802, 0.3404, "Centre-ville"),
    "plateau": (46.5802, 0.3404, "Centre-ville"),
    "cathedrale": (46.5804, 0.3489, "Cathédrale"),
    "sainte radegonde": (46.5801, 0.3518, "Sainte-Radegonde"),
    "blossac": (46.5735, 0.3365, "Blossac"),
    "gare": (46.5822, 0.3331, "Gare"),
    "porteau": (46.5978, 0.3358, "Le Porteau"),
    "couronneries": (46.5975, 0.3510, "Couronneries"),
    "trois cites": (46.5655, 0.3550, "Trois Cités"),
    "beaulieu": (46.5715, 0.3710, "Beaulieu"),
    "gibauderie": (46.5650, 0.3480, "La Gibauderie"),
    "ganterie": (46.5695, 0.3350, "La Ganterie"),
    "blaiserie": (46.5795, 0.3210, "La Blaiserie"),
    "saint-eloi": (46.5840, 0.3770, "Saint-Éloi"),
    "saint-cyprien": (46.5665, 0.3410, "Saint-Cyprien"),
    "montbernage": (46.5795, 0.3545, "Montbernage"),
    "chilvert": (46.5775, 0.3260, "Chilvert"),
    "bel air": (46.5885, 0.3210, "Bel Air"),
    "chu": (46.5646, 0.3738, "CHU / Milétrie"),
    "miletrie": (46.5646, 0.3738, "CHU / Milétrie"),
    "campus": (46.5688, 0.3828, "Campus"),
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


def _charger_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _sauver_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _strip_tags(raw_html: str) -> str:
    txt = re.sub(r"<script[^>]*>.*?</script>", " ", raw_html, flags=re.S | re.I)
    txt = re.sub(r"<style[^>]*>.*?</style>", " ", txt, flags=re.S | re.I)
    txt = re.sub(r"<[^>]+>", " ", txt)
    return re.sub(r"\s+", " ", html_lib.unescape(txt)).strip()


def extraire_champs_texte_structure(texte: str) -> dict:
    """
    Extrait DPE, GES, année de construction, étage, pièces, chambres et quartier
    à partir du texte brut ou d'un en-tête MoteurImmo + description.
    """
    if not texte:
        return {}
    t = str(texte)
    out: dict = {}

    # DPE / GES
    m_dpe = re.search(
        r"(?:\bDPE\b|classe\s+[ée]nergie|[ée]tiquette\s+[ée]nergie|consommation\s+[ée]nerg[ée]tique)\s*[:\-]?\s*([A-G])\b",
        t,
        re.I,
    )
    if m_dpe:
        out["dpe"] = m_dpe.group(1).upper()

    m_ges = re.search(
        r"(?:\bGES\b|classe\s+climat|[ée]tiquette\s+climat|gaz\s+[àa]\s+effet\s+de\s+serre)\s*[:\-]?\s*([A-G])\b",
        t,
        re.I,
    )
    if m_ges:
        out["ges"] = m_ges.group(1).upper()

    # Année de construction : libellée OU badge autonome MoteurImmo (ex: "Pro 2012", "250 m² 1970", "Pro 1960")
    m_an = (
        re.search(r"(?:ann[ée]e\s*(?:de\s*)?construction|construite?\s+en|b[âa]tie?\s+en)\s*[:\-]?\s*(1[6-9]\d{2}|20[0-2]\d)\b", t, re.I)
        or re.search(r"\b(?:Pro|Part\.?|m[²2])\s+(1[6-9]\d{2}|20[0-2]\d)\b", t)
        or re.search(r"\b(?:demeure|maison|immeuble|r[ée]sidence)\s+(?:de|des\s+ann[ée]es)\s+(1[6-9]\d{2}|20[0-2]\d)\b", t, re.I)
    )
    if m_an:
        an = int(m_an.group(1))
        if 1600 <= an <= 2030:
            out["annee_construction"] = an

    # Étage : distinguer "4e étage" / "1er/2" / "RDC" de "5 étages" (hauteur totale de l'immeuble)
    if re.search(r"\b(?:rdc|rez[\-\s]de[\-\s]chauss[ée]e|plain[\-\s]pied)\b", t, re.I):
        out["etage"] = 0
    else:
        m_et = (
            re.search(r"\b(\d{1,2})\s*(?:er|ère|e|ème)\s*/\s*\d{1,2}\b", t, re.I)
            or re.search(r"\b(?:au|situ[ée]\s+au)?\s*(\d{1,2})\s*(?:er|ère|e|ème)\s*[ée]tage\b", t, re.I)
            or re.search(r"\b[ée]tage\s*[:\-]?\s*(\d{1,2})\b(?!\s*[ée]tages?\b)", t, re.I)
        )
        if m_et:
            et = int(m_et.group(1))
            if 0 <= et <= 30:
                out["etage"] = et

    # Pièces & Chambres
    m_p = (
        re.search(r"\b(\d{1,2})\s*pi[eè]ces?\b", t, re.I)
        or re.search(r"\b[TF](\d{1,2})\b", t)
        or re.search(r"\bm[²2]\s+(\d{1,2})\s+\d{1,2}\b", t, re.I)
    )
    if m_p:
        p_val = int(m_p.group(1))
        if 1 <= p_val <= 20:
            out["pieces"] = p_val
    elif re.search(r"\bstudio\b", t, re.I):
        out["pieces"] = 1

    m_ch = (
        re.search(r"\b(\d{1,2})\s*chambres?\b", t, re.I)
        or re.search(r"\bm[²2]\s+\d{1,2}\s+(\d{1,2})\b", t, re.I)
    )
    if m_ch:
        ch_val = int(m_ch.group(1))
        if 0 <= ch_val <= 15:
            out["nbr_chambres"] = ch_val

    return out


def nettoyer_description_moteurimmo(desc: str) -> str:
    """Retire le bruit de navigation MoteurImmo tout en conservant le texte complet de l'annonce."""
    if not desc:
        return ""
    s = str(desc).strip()
    # Couper le préfixe d'onglets MoteurImmo si le vrai corps suit après "Pro" / "Part."
    m_body = re.search(r"(?:Ajout/Correction\s+Description.*?)(?:Pro|Part\.?)\s+(?:Pro|Part\.?\s+)?(.+)$", s, re.I)
    if m_body and len(m_body.group(1)) > 60:
        s = m_body.group(1).strip()
    return s


def _extraire_adresse_explicite(texte: str) -> str | None:
    """Extrait une rue, avenue, boulevard, place ou résidence explicitement citée dans le texte."""
    if not texte:
        return None
    patterns = [
        r"\b((?:rue|avenue|av\.?|boulevard|bd\.?|place|all[ée]e|impasse|chemin|quai|square|r[ée]sidence)\s+(?:d[eu']\s+|de\s+la\s+|des\s+|du\s+)?[A-ZÉÈÊÀÂÎÔÙÇa-zéèêàâîôùç'\-]{2,}(?:\s+[A-ZÉÈÊÀÂÎÔÙÇa-zéèêàâîôùç'\-]{2,}){0,4})\b",
    ]
    for pat in patterns:
        m = re.search(pat, texte, re.I)
        if m:
            cand = re.sub(r"\s+", " ", m.group(1)).strip(" ,.;:-")
            if len(cand) >= 7 and not re.search(r"\b(?:commerces|écoles|bus|calme|sans\s+issue|passante)\b", cand, re.I):
                return cand
    return None


def geocoder_adresse_etat(requete: str, code_postal: str = "") -> dict | None:
    """
    Interroge l'API publique gratuite Base Adresse Nationale (api-adresse.data.gouv.fr),
    avec repli sur l'API Géoplateforme (data.geopf.fr), sur toute la France métropolitaine.
    """
    cp = str(code_postal or "").strip()[:5]
    params_base = {"q": requete, "limit": 1}
    if cp and cp.isdigit() and len(cp) == 5:
        params_base["postcode"] = cp

    endpoints = [
        ("ban", "https://api-adresse.data.gouv.fr/search/"),
        ("geoplateforme", "https://data.geopf.fr/geocodage/search"),
    ]
    for src_name, url in endpoints:
        try:
            r = requests.get(
                url,
                params=params_base,
                timeout=6,
            )
            if r.status_code != 200:
                continue
            feats = r.json().get("features", [])
            if not feats:
                continue
            f0 = feats[0]
            props = f0.get("properties", {})
            coords = f0.get("geometry", {}).get("coordinates", [])
            score = float(props.get("score", 0.0) or 0.0)
            ftype = str(props.get("type", "")).lower()
            if len(coords) == 2 and score >= 0.42:
                lon, lat = float(coords[0]), float(coords[1])
                if 41.0 <= lat <= 51.5 and -5.5 <= lon <= 10.0:
                    return {
                        "latitude": round(lat, 6),
                        "longitude": round(lon, 6),
                        "score": round(score, 3),
                        "type": ftype,
                        "label": props.get("label", ""),
                        "source": src_name,
                    }
        except Exception:
            continue
    return None


def enrichir_depuis_portail(url: str, cache: dict | None = None) -> dict:
    """
    Récupère les données structurées publiques depuis l'URL réelle de l'annonce :
    - Bien'ici : JSON public `/realEstateAd.json?id=...`
    - Portails agences publics (Foncia, IAD, Efficity, etc.) : JSON-LD + HTML + JS embarqué
    - Respecte strictement les blocages anti-bot (aucun contournement).
    """
    if not url or not str(url).startswith("http"):
        return {}

    u = str(url).strip()
    if cache is not None and u in cache:
        return cache[u]

    u_low = u.lower()
    if any(dom in u_low for dom in DOMAINES_ANTI_BOT):
        return {}

    res: dict = {}
    try:
        if "bienici.com/annonce/" in u_low:
            ad_id = u.rstrip("/").split("/")[-1].split("?")[0]
            r = requests.get(f"https://www.bienici.com/realEstateAd.json?id={ad_id}", headers=HEADERS, timeout=8)
            if r.status_code == 200:
                d = r.json()
                pos = d.get("blurInfo", {}).get("position") or {}
                if pos.get("lat") and pos.get("lon"):
                    res["latitude"] = float(pos["lat"])
                    res["longitude"] = float(pos["lon"])
                    res["localisation_statut"] = "exacte"
                    res["localisation_source"] = "bienici_json"
                for k_src, k_dst in [
                    ("energyClassification", "dpe"),
                    ("greenhouseGazClassification", "ges"),
                    ("yearOfConstruction", "annee_construction"),
                    ("floor", "etage"),
                    ("roomsQuantity", "pieces"),
                    ("bedroomsQuantity", "nbr_chambres"),
                    ("description", "description"),
                    ("title", "titre"),
                ]:
                    if d.get(k_src) not in (None, ""):
                        res[k_dst] = d[k_src]
                if isinstance(d.get("district"), dict) and d["district"].get("name"):
                    res["quartier"] = d["district"]["name"]
                if isinstance(d.get("photos"), list):
                    urls_p = [p.get("url") or p.get("url_photo") for p in d["photos"] if isinstance(p, dict)]
                    urls_p = [p for p in urls_p if p]
                    if urls_p:
                        res["photos_urls"] = "|".join(urls_p[:6])
        else:
            r = requests.get(u, headers=HEADERS, timeout=8)
            if r.status_code == 200 and len(r.text) > 1000:
                html = r.text
                # 1. JSON-LD
                for ld_raw in re.findall(r"<script[^>]+ld\+json[^>]*>(.*?)</script>", html, re.S | re.I):
                    try:
                        ld = json.loads(ld_raw.strip())
                    except Exception:
                        continue
                    nodes = ld if isinstance(ld, list) else (ld.get("@graph", [ld]) if isinstance(ld, dict) else [])
                    for node in nodes:
                        if not isinstance(node, dict):
                            continue
                        geo = node.get("geo")
                        if isinstance(geo, dict) and geo.get("latitude") and geo.get("longitude"):
                            res["latitude"] = float(geo["latitude"])
                            res["longitude"] = float(geo["longitude"])
                            res["localisation_statut"] = "exacte"
                            res["localisation_source"] = "json_ld"
                        desc_ld = node.get("description")
                        if isinstance(desc_ld, str) and len(desc_ld) > len(str(res.get("description", ""))):
                            if not any(g in desc_ld.lower() for g in ("administration de biens", "filiale d'emeria")):
                                res["description"] = desc_ld.strip()
                        if node.get("yearBuilt") and not res.get("annee_construction"):
                            res["annee_construction"] = int(node["yearBuilt"])
                        if node.get("numberOfRooms") and not res.get("pieces"):
                            res["pieces"] = int(node["numberOfRooms"])

                # 2. Coordonnées JS embarquées ("lat": 46.xxx, "lng": 0.xxx)
                if "latitude" not in res:
                    m_geo = re.search(
                        r'"lat(?:itude)?"\s*:\s*(46\.\d{3,})[^}]{0,120}"(?:lon|lng|longitude)"\s*:\s*(0\.\d{3,})',
                        html,
                    )
                    if m_geo:
                        res["latitude"] = float(m_geo.group(1))
                        res["longitude"] = float(m_geo.group(2))
                        res["localisation_statut"] = "exacte"
                        res["localisation_source"] = "html_js"

                # 3. Extraction DPE / GES / année / étage / description depuis HTML structuré & JS embarqué
                # (Couper avant les blocs "biens similaires" si présents, ex: foncia-card-dpe)
                html_main = html.split("foncia-card-dpe")[0]
                for pat_dpe in (
                    r'"(?:dpe|energy(?:Class|Classification|Grade|Letter))"\s*:\s*"([A-G])"',
                    r'class="[^"]*dpe-class-([A-G])[^"]*"',
                    r'(?:dpe|classe\s+[ée]nergie)[^<]{0,50}>\s*([A-G])\s*<',
                ):
                    m_d = re.search(pat_dpe, html_main, re.I)
                    if m_d and "dpe" not in res:
                        res["dpe"] = m_d.group(1).upper()
                        break

                for pat_ges in (
                    r'"(?:ges|ghg(?:Class|Classification|Grade|Letter)|climat)"\s*:\s*"([A-G])"',
                    r'class="[^"]*ges-class-([A-G])[^"]*"',
                    r'(?:ges|climat|effet\s+de\s+serre)[^<]{0,50}>\s*([A-G])\s*<',
                ):
                    m_g = re.search(pat_ges, html_main, re.I)
                    if m_g and "ges" not in res:
                        res["ges"] = m_g.group(1).upper()
                        break

                if "ges" not in res:
                    m_ges_num = re.search(r'"ges"\s*:\s*(\d+(?:\.\d+)?)', html_main, re.I)
                    if m_ges_num:
                        v_ges = float(m_ges_num.group(1))
                        res["ges"] = "A" if v_ges <= 6 else ("B" if v_ges <= 11 else ("C" if v_ges <= 30 else ("D" if v_ges <= 50 else ("E" if v_ges <= 70 else ("F" if v_ges <= 100 else "G")))))

                txt_page = _strip_tags(html_main)
                champs = extraire_champs_texte_structure(txt_page)
                for k, v in champs.items():
                    if k not in res and v not in (None, ""):
                        res[k] = v

                # 4. Fallback og:description si plus longue
                if len(str(res.get("description", ""))) < 250:
                    m_og = re.search(r'<meta[^>]+(?:name|property)=["\'](?:og:description|description)["\'][^>]+content=["\']([^"\']{80,})["\']', html, re.I)
                    if m_og:
                        og_txt = html_lib.unescape(m_og.group(1)).strip()
                        if len(og_txt) > len(str(res.get("description", ""))):
                            res["description"] = og_txt
    except Exception:
        pass

    if cache is not None and res:
        cache[u] = res
    return res


def determiner_localisation(row: dict, cache_geo: dict | None = None) -> dict:
    """
    Détermine la meilleure localisation possible tout en distinguant strictement :
      - exacte          : GPS portail/JSON-LD ou rue explicite géocodée via BAN (score >= 0.55)
      - approximative   : quartier / repère connu ou géocodé
      - non_disponible  : aucune adresse/quartier/GPS (garde None ou centre commune explicitement marqué)
    """
    lat = pd.to_numeric(row.get("latitude"), errors="coerce")
    lon = pd.to_numeric(row.get("longitude"), errors="coerce")
    statut_existant = str(row.get("localisation_statut") or "")

    # 1. GPS déjà confirmé (hors centre par défaut 46.5802 / 0.3404)
    if pd.notna(lat) and pd.notna(lon) and (round(float(lat), 4) != 46.5802 or round(float(lon), 4) != 0.3404):
        return {
            "latitude": float(lat),
            "longitude": float(lon),
            "localisation_statut": statut_existant if statut_existant in {"exacte", "approximative"} else "exacte",
            "localisation_source": str(row.get("localisation_source") or "portail_gps"),
        }

    texte_complet = f"{row.get('titre', '')} {row.get('description', '')}"
    cp = str(row.get("code_postal") or "86000")[:5]

    # 2. Rue / adresse explicite géocodée via API Adresse État (BAN / Géoplateforme)
    rue = _extraire_adresse_explicite(texte_complet)
    if rue:
        cle_c = f"ban::{cp}::{rue.lower()}"
        geo = cache_geo.get(cle_c) if cache_geo is not None else None
        if geo is None:
            geo = geocoder_adresse_etat(rue, cp)
            if cache_geo is not None and geo:
                cache_geo[cle_c] = geo
        if geo and geo["score"] >= 0.55 and geo["type"] in {"housenumber", "street"}:
            return {
                "latitude": geo["latitude"],
                "longitude": geo["longitude"],
                "localisation_statut": "exacte",
                "localisation_source": f"{geo['source']}_rue",
            }
        if geo:
            return {
                "latitude": geo["latitude"],
                "longitude": geo["longitude"],
                "localisation_statut": "approximative",
                "localisation_source": f"{geo['source']}_approx",
            }

    # 3. Quartier / repère explicite
    txt_norm = unicodedata.normalize("NFKD", texte_complet).encode("ascii", "ignore").decode("ascii").lower()
    for cle_q, (q_lat, q_lon, q_nom) in SECTEURS_POITIERS.items():
        if cle_q in txt_norm:
            return {
                "latitude": q_lat,
                "longitude": q_lon,
                "quartier": row.get("quartier") or q_nom,
                "localisation_statut": "approximative",
                "localisation_source": "secteur_texte",
            }

    # 4. Aucune localisation précise disponible
    return {
        "latitude": np.nan,
        "longitude": np.nan,
        "localisation_statut": "non_disponible",
        "localisation_source": "aucune",
    }


def enrichir_dataframe(df: pd.DataFrame, max_requetes_portail: int = 200, verbose: bool = True) -> pd.DataFrame:
    """
    Enrichit un DataFrame d'annonces (prioritairement celles à qui il manque DPE/GES/année/GPS)
    sans modifier les champs déjà valides.
    """
    if df.empty:
        return df

    d = df.copy()
    cache = _charger_cache()
    modifies_cache = 0
    reqs = 0

    for col_new in ("localisation_statut", "localisation_source"):
        if col_new not in d.columns:
            d[col_new] = None

    for idx in d.index:
        row = d.loc[idx].to_dict()
        desc = str(row.get("description") or "")
        titre = str(row.get("titre") or "")

        # A. Parsing texte structuré sur la fiche déjà présente
        champs_txt = extraire_champs_texte_structure(f"{titre} {desc}")
        for k, v in champs_txt.items():
            cur = row.get(k)
            if cur is None or (isinstance(cur, float) and pd.isna(cur)) or str(cur).strip() in {"", "nan", "None"}:
                d.at[idx, k] = v
                row[k] = v

        # B. Enrichissement portail public si un champ critique manque encore
        lat = pd.to_numeric(row.get("latitude"), errors="coerce")
        gps_defaut = pd.isna(lat) or round(float(lat), 4) == 46.5802
        manque_critique = any(
            row.get(c) is None or (isinstance(row.get(c), float) and pd.isna(row.get(c))) or str(row.get(c)).strip() in {"", "nan", "None"}
            for c in ("dpe", "ges", "annee_construction", "etage")
        ) or gps_defaut

        url = str(row.get("url") or "")
        if manque_critique and url.startswith("http") and reqs < max_requetes_portail:
            avant_len = len(cache)
            ext = enrichir_depuis_portail(url, cache=cache)
            if len(cache) > avant_len:
                modifies_cache += 1
            if url not in cache:
                reqs += 1
            for k, v in ext.items():
                if v in (None, "", []):
                    continue
                if k == "description":
                    if len(str(v)) > len(str(row.get("description") or "")):
                        d.at[idx, k] = v
                        row[k] = v
                elif k in {"latitude", "longitude"} and gps_defaut:
                    d.at[idx, k] = v
                    row[k] = v
                else:
                    cur = row.get(k)
                    if cur is None or (isinstance(cur, float) and pd.isna(cur)) or str(cur).strip() in {"", "nan", "None"}:
                        d.at[idx, k] = v
                        row[k] = v

        # C. Nettoyer le préfixe de description MoteurImmo si besoin
        desc_clean = nettoyer_description_moteurimmo(str(row.get("description") or ""))
        if desc_clean and len(desc_clean) >= 40:
            d.at[idx, "description"] = desc_clean
            row["description"] = desc_clean

        # D. Localisation vérifiée (exacte vs approximative vs non_disponible)
        avant_geo = len(cache)
        loc = determiner_localisation(row, cache_geo=cache)
        if len(cache) > avant_geo:
            modifies_cache += 1
        d.at[idx, "latitude"] = loc["latitude"]
        d.at[idx, "longitude"] = loc["longitude"]
        d.at[idx, "localisation_statut"] = loc["localisation_statut"]
        d.at[idx, "localisation_source"] = loc["localisation_source"]
        if loc.get("quartier") and (row.get("quartier") is None or pd.isna(row.get("quartier"))):
            d.at[idx, "quartier"] = loc["quartier"]

    if modifies_cache > 0:
        _sauver_cache(cache)

    if verbose:
        st_counts = d["localisation_statut"].value_counts(dropna=False).to_dict()
        print(f"[Enrichissement] {len(d)} annonces traitées | Localisation : {st_counts}")

    return d
