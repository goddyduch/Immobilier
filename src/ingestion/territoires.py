"""
Référentiel centralisé des 12 villes-centres V1 (hors La Rochelle et hors couronnes périurbaines).

Chaque territoire contient :
- code_territoire, nom, departement, code_insee, codes_postaux (commune-centre uniquement)
- repères urbains (centre, gare, campus, chu) et rayon_urbain_km pour normalisation des distances
- utilitaires de résolution déterministe (code_insee / code_postal / nom de ville)
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TerritoireConfig:
    code_territoire: str
    nom: str
    departement: str
    code_insee: str
    codes_postaux: tuple[str, ...]
    rayon_urbain_km: float
    centre_lat: float
    centre_lon: float
    gare_lat: float
    gare_lon: float
    campus_lat: float
    campus_lon: float
    chu_lat: float
    chu_lon: float
    loyer_m2_base: float
    revenu_median_defaut: float


VILLES_CENTRES: dict[str, TerritoireConfig] = {
    "poitiers": TerritoireConfig(
        code_territoire="poitiers",
        nom="Poitiers",
        departement="86",
        code_insee="86194",
        codes_postaux=("86000",),
        rayon_urbain_km=4.5,
        centre_lat=46.5802, centre_lon=0.3404,
        gare_lat=46.5822, gare_lon=0.3331,
        campus_lat=46.5698, campus_lon=0.3827,
        chu_lat=46.5579, chu_lon=0.3771,
        loyer_m2_base=11.6,
        revenu_median_defaut=21500.0,
    ),
    "tours": TerritoireConfig(
        code_territoire="tours",
        nom="Tours",
        departement="37",
        code_insee="37261",
        codes_postaux=("37000", "37100", "37200"),
        rayon_urbain_km=6.0,
        centre_lat=47.3941, centre_lon=0.6848,
        gare_lat=47.3898, gare_lon=0.6939,
        campus_lat=47.3586, campus_lon=0.7031,
        chu_lat=47.3879, chu_lon=0.6716,
        loyer_m2_base=13.2,
        revenu_median_defaut=22400.0,
    ),
    "angers": TerritoireConfig(
        code_territoire="angers",
        nom="Angers",
        departement="49",
        code_insee="49007",
        codes_postaux=("49000", "49100"),
        rayon_urbain_km=6.0,
        centre_lat=47.4784, centre_lon=-0.5632,
        gare_lat=47.4646, gare_lon=-0.5568,
        campus_lat=47.4806, campus_lon=-0.5973,
        chu_lat=47.4893, chu_lon=-0.5566,
        loyer_m2_base=13.4,
        revenu_median_defaut=22600.0,
    ),
    "niort": TerritoireConfig(
        code_territoire="niort",
        nom="Niort",
        departement="79",
        code_insee="79191",
        codes_postaux=("79000",),
        rayon_urbain_km=4.5,
        centre_lat=46.3237, centre_lon=-0.4647,
        gare_lat=46.3193, gare_lon=-0.4547,
        campus_lat=46.3292, campus_lon=-0.4744,
        chu_lat=46.3275, chu_lon=-0.4754,
        loyer_m2_base=10.8,
        revenu_median_defaut=22200.0,
    ),
    "le_mans": TerritoireConfig(
        code_territoire="le_mans",
        nom="Le Mans",
        departement="72",
        code_insee="72181",
        codes_postaux=("72000", "72100"),
        rayon_urbain_km=6.0,
        centre_lat=48.0061, centre_lon=0.1996,
        gare_lat=47.9956, gare_lon=0.1927,
        campus_lat=48.0172, campus_lon=0.1610,
        chu_lat=48.0132, chu_lon=0.1912,
        loyer_m2_base=11.2,
        revenu_median_defaut=21200.0,
    ),
    "orleans": TerritoireConfig(
        code_territoire="orleans",
        nom="Orléans",
        departement="45",
        code_insee="45234",
        codes_postaux=("45000", "45100"),
        rayon_urbain_km=6.0,
        centre_lat=47.9029, centre_lon=1.9093,
        gare_lat=47.9078, gare_lon=1.9045,
        campus_lat=47.8442, campus_lon=1.9383,
        chu_lat=47.8390, chu_lon=1.9195,
        loyer_m2_base=12.6,
        revenu_median_defaut=22300.0,
    ),
    "clermont_ferrand": TerritoireConfig(
        code_territoire="clermont_ferrand",
        nom="Clermont-Ferrand",
        departement="63",
        code_insee="63113",
        codes_postaux=("63000", "63100"),
        rayon_urbain_km=6.0,
        centre_lat=45.7772, centre_lon=3.0870,
        gare_lat=45.7789, gare_lon=3.1008,
        campus_lat=45.7606, campus_lon=3.1128,
        chu_lat=45.7603, chu_lon=3.0906,
        loyer_m2_base=11.9,
        revenu_median_defaut=21400.0,
    ),
    "reims": TerritoireConfig(
        code_territoire="reims",
        nom="Reims",
        departement="51",
        code_insee="51454",
        codes_postaux=("51100",),
        rayon_urbain_km=6.0,
        centre_lat=49.2583, centre_lon=4.0317,
        gare_lat=49.2590, gare_lon=4.0243,
        campus_lat=49.2374, campus_lon=4.0630,
        chu_lat=49.2323, chu_lon=4.0211,
        loyer_m2_base=12.5,
        revenu_median_defaut=21300.0,
    ),
    "dijon": TerritoireConfig(
        code_territoire="dijon",
        nom="Dijon",
        departement="21",
        code_insee="21231",
        codes_postaux=("21000",),
        rayon_urbain_km=5.5,
        centre_lat=47.3220, centre_lon=5.0415,
        gare_lat=47.3233, gare_lon=5.0273,
        campus_lat=47.3118, campus_lon=5.0715,
        chu_lat=47.3212, chu_lon=5.0695,
        loyer_m2_base=12.8,
        revenu_median_defaut=22900.0,
    ),
    "caen": TerritoireConfig(
        code_territoire="caen",
        nom="Caen",
        departement="14",
        code_insee="14118",
        codes_postaux=("14000",),
        rayon_urbain_km=5.5,
        centre_lat=49.1829, centre_lon=-0.3707,
        gare_lat=49.1763, gare_lon=-0.3477,
        campus_lat=49.1904, campus_lon=-0.3652,
        chu_lat=49.2036, chu_lon=-0.3591,
        loyer_m2_base=12.3,
        revenu_median_defaut=21900.0,
    ),
    "limoges": TerritoireConfig(
        code_territoire="limoges",
        nom="Limoges",
        departement="87",
        code_insee="87085",
        codes_postaux=("87000", "87100", "87280"),
        rayon_urbain_km=5.5,
        centre_lat=45.8336, centre_lon=1.2611,
        gare_lat=45.8362, gare_lon=1.2673,
        campus_lat=45.8195, campus_lon=1.2421,
        chu_lat=45.8206, chu_lon=1.2392,
        loyer_m2_base=10.3,
        revenu_median_defaut=20500.0,
    ),
    "besancon": TerritoireConfig(
        code_territoire="besancon",
        nom="Besançon",
        departement="25",
        code_insee="25056",
        codes_postaux=("25000",),
        rayon_urbain_km=5.0,
        centre_lat=47.2378, centre_lon=6.0241,
        gare_lat=47.2464, gare_lon=6.0218,
        campus_lat=47.2455, campus_lon=5.9866,
        chu_lat=47.2248, chu_lon=5.9636,
        loyer_m2_base=11.4,
        revenu_median_defaut=21600.0,
    ),
}

# Index rapides de résolution
_CP_TO_TERRITOIRE: dict[str, str] = {
    cp: code
    for code, cfg in VILLES_CENTRES.items()
    for cp in cfg.codes_postaux
}
_INSEE_TO_TERRITOIRE: dict[str, str] = {
    cfg.code_insee: code for code, cfg in VILLES_CENTRES.items()
}


def normaliser_texte(val: Any) -> str:
    if val is None:
        return ""
    s = unicodedata.normalize("NFKD", str(val)).encode("ascii", "ignore").decode("ascii")
    return " ".join(s.lower().replace("-", " ").replace("_", " ").split())


_NOM_TO_TERRITOIRE: dict[str, str] = {
    normaliser_texte(cfg.nom): code for code, cfg in VILLES_CENTRES.items()
}


def get_territoire(code_territoire: str) -> TerritoireConfig | None:
    if not code_territoire:
        return None
    norm = normaliser_texte(code_territoire).replace(" ", "_")
    return VILLES_CENTRES.get(code_territoire) or VILLES_CENTRES.get(norm)


def lister_codes_territoires() -> list[str]:
    return list(VILLES_CENTRES.keys())


def lister_noms_villes() -> list[str]:
    return [cfg.nom for cfg in VILLES_CENTRES.values()]


def lister_tous_codes_postaux() -> tuple[str, ...]:
    return tuple(_CP_TO_TERRITOIRE.keys())


def resoudre_territoire(
    code_postal: Any = None,
    ville: Any = None,
    code_insee: Any = None,
) -> str | None:
    """Résout de façon déterministe le `code_territoire` d'une ville-centre (sinon None)."""
    if code_insee:
        ci = str(code_insee).strip()
        if ci in _INSEE_TO_TERRITOIRE:
            return _INSEE_TO_TERRITOIRE[ci]

    if code_postal:
        cp = str(code_postal).strip()[:5]
        if cp in _CP_TO_TERRITOIRE:
            return _CP_TO_TERRITOIRE[cp]

    if ville:
        vn = normaliser_texte(ville)
        if vn in _NOM_TO_TERRITOIRE:
            return _NOM_TO_TERRITOIRE[vn]
    return None


def est_ville_centre(
    code_postal: Any = None,
    ville: Any = None,
    code_insee: Any = None,
) -> bool:
    """True uniquement si le bien appartient à l'une des 12 villes-centres V1."""
    if code_insee and str(code_insee).strip() in _INSEE_TO_TERRITOIRE:
        return True
    if code_postal and str(code_postal).strip()[:5] in _CP_TO_TERRITOIRE:
        return True
    if ville and normaliser_texte(ville) in _NOM_TO_TERRITOIRE:
        return True
    return False


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance géodésique en km entre deux points WGS84."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2.0 * r * math.asin(min(1.0, math.sqrt(a)))


def territoire_plus_proche(lat: float, lon: float, max_km: float = 15.0) -> str | None:
    """Fallback géographique si code postal absent : rattache à la ville-centre < max_km."""
    meilleur_code = None
    meilleure_dist = max_km
    for code, cfg in VILLES_CENTRES.items():
        d = haversine_km(lat, lon, cfg.centre_lat, cfg.centre_lon)
        if d <= meilleure_dist:
            meilleure_dist = d
            meilleur_code = code
    return meilleur_code
