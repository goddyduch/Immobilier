""" traitement des données """

import json
import pandas as pd 
from DB import *

#bien ici 
with open("annonces_bienici.json", "r", encoding="utf-8") as f:
    data = json.load(f)
df_bienIci= pd.DataFrame(data)



print(df_bienIci.columns)
#print(df_bienIci.head(5))


#jinka 
import json
import re
import pandas as pd


def extract_listing(ad: dict) -> dict:
    """Transforme une annonce Jinka (dict) en ligne plate."""
    ld = (ad.get("json_ld") or [{}])[0]
    entity = ld.get("mainEntity", {}) or {}
    address = entity.get("address", {}) or {}
    offers = ld.get("offers", {}) or {}
    floor = entity.get("floorSize", {}) or {}
    ext = ad.get("liens_externes") or []
    geo = entity.get("geo", {}) or {}
    carte = ad.get("carte") or {}
    texte = ad.get("texte_complet") or ""

    # DPE / GES : lettre A-G qui suit le libellé dans le texte brut
    m_dpe = re.search(r"\(DPE\)\s*([A-G])\b", texte)
    m_ges = re.search(r"\(GES\)\s*([A-G])\b", texte)
    dpe = m_dpe.group(1) if m_dpe else carte.get("dpe")
    ges = m_ges.group(1) if m_ges else None

    # Quartier : "Poitiers - Centre ville" -> "Centre ville" (None si absent)
    loc = carte.get("localisation") or ""
    quartier = loc.split(" - ", 1)[1].strip() if " - " in loc else None

    # NB : "carte.prix" est corrompu (ex. "768600" pour 68 600 €),
    # on utilise donc json_ld.offers.price.
    return {
        "id": ad.get("id"),
        "titre": ld.get("name") or ad.get("titre"),
        "ville": address.get("addressLocality"),
        "code_postal": address.get("postalCode"),
        "prix": offers.get("price"),
        "surface_m2": floor.get("value"),
        "pieces": entity.get("numberOfRooms"),
        "nbr_chambres": entity.get("numberOfBedrooms"),
        "etage": entity.get("floorLevel"),
        "annee_construction": entity.get("yearBuilt"),
        "dpe": dpe,
        "ges": ges,
        "quartier": quartier,
        "latitude": geo.get("latitude"),
        "longitude": geo.get("longitude"),
        "type_bien": entity.get("@type"),
        "description": ld.get("description"),
        "date_publication": ld.get("datePosted"),
        # 1er lien externe = annonce d'origine (bienici, citya...), sinon URL Jinka
        "url": ext[0] if ext else ad.get("url"),
    }


def build_dataframe(ads: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([extract_listing(a) for a in ads])
    df["prix"] = pd.to_numeric(df["prix"], errors="coerce")
    df["surface_m2"] = pd.to_numeric(df["surface_m2"], errors="coerce")
    df["pieces"] = pd.to_numeric(df["pieces"], errors="coerce").astype("Int64")
    for col in ["nbr_chambres", "etage", "annee_construction"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ["latitude", "longitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date_publication"] = pd.to_datetime(df["date_publication"], utc=True, errors="coerce")
    return df


if __name__ == "__main__":
    with open("annonces_bienici.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    df_bienIci= pd.DataFrame(data)


    with open("jinka_annonces.json", encoding="utf-8") as f:
        ads = json.load(f)
    df_jinka = build_dataframe(ads)
    pd.set_option("display.width", 200, "display.max_columns", None)
    print(df_jinka.drop(columns=["description"]).head())
    df = df_jinka.drop_duplicates(subset="id", keep="first")


    df = pd.concat([df_jinka, df_bienIci], ignore_index=True)
    df.to_csv("annonces.csv", index=False)
    creer_table_depuis_df(df)
    


