"""
Pipeline d'intégration incrémentale et de synchronisation PostgreSQL.

1. Extrait et fusionne les annonces issues de Bien'ici (`annonces_bienici.json`) et Jinka (`jinka_annonces.json`).
2. Synchronise les offres brutes dans PostgreSQL (`DB.py`) :
   - `offres_historique_brut`     : toutes les offres (actives + plus à jour) brutes
   - `offres_recentes_brut`       : uniquement les offres encore à jour brutes
3. Applique le Feature Engineering + Prédictions IA (`FeatureEngineering.py`) et sauvegarde :
   - `offres_historique_features` : toutes les offres (actives + plus à jour) après Feature Engineering
   - `offres_recentes_features`   : uniquement les offres encore à jour après Feature Engineering + Prédictions

Usage :
    python Merge.py                   # exécute la pipeline complète
    python Merge.py --verifier-http   # vérifie en plus par requête HTTP directe si les annonces actives sont toujours en ligne
"""

import argparse
import json
import os
import re
import pandas as pd

import DB
import Entrainement as tr
import FeatureEngineering as fe
import Prediction as pr


def extract_listing_jinka(ad: dict) -> dict:
    """Transforme une annonce Jinka (dict) en ligne plate propre."""
    ld = (ad.get("json_ld") or [{}])[0]
    entity = ld.get("mainEntity", {}) or {}
    address = entity.get("address", {}) or {}
    offers = ld.get("offers", {}) or {}
    floor = entity.get("floorSize", {}) or {}
    ext = ad.get("liens_externes") or []
    geo = entity.get("geo", {}) or {}
    carte = ad.get("carte") or {}
    texte = ad.get("texte_complet") or ""

    m_dpe = re.search(r"\(DPE\)\s*([A-G])\b", texte)
    m_ges = re.search(r"\(GES\)\s*([A-G])\b", texte)
    dpe = m_dpe.group(1) if m_dpe else carte.get("dpe")
    ges = m_ges.group(1) if m_ges else None

    loc = carte.get("localisation") or ""
    quartier = loc.split(" - ", 1)[1].strip() if " - " in loc else None

    # Prix JSON-LD officiel (évite les concaténations brutes de carte)
    prix = offers.get("price")

    # On écarte les liens PDF d'honoraires ou Géorisques pour garder le vrai lien de l'annonce
    liens_valides = [
        u for u in ext
        if "bareme_honoraires" not in u.lower() and "georisques.gouv.fr" not in u.lower()
    ]
    ad_id = ad.get("id")
    url_propre = (
        liens_valides[0]
        if liens_valides
        else (ld.get("url") or ad.get("url") or (f"https://www.jinka.fr/ad/{ad_id}" if ad_id else None))
    )

    # Extraction des 5 premières photos réelles (en excluant les logos/icônes SVG)
    imgs_raw = ad.get("images") or []
    imgs_valides = [
        u for u in imgs_raw
        if isinstance(u, str)
        and u.startswith("http")
        and not u.lower().endswith(".svg")
        and "logo" not in u.lower()
        and "icon" not in u.lower()
    ][:5]

    return {
        "id": ad_id,
        "titre": ld.get("name") or ad.get("titre"),
        "ville": address.get("addressLocality"),
        "code_postal": address.get("postalCode"),
        "prix": prix,
        "surface_m2": floor.get("value") or carte.get("surface_m2"),
        "pieces": entity.get("numberOfRooms") or carte.get("pieces"),
        "nbr_chambres": entity.get("numberOfBedrooms") or carte.get("chambres"),
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
        "url": url_propre,
        "photos_urls": "|".join(imgs_valides),
    }


def build_dataframe_jinka(ads: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([extract_listing_jinka(a) for a in ads])
    if df.empty:
        return df
    df["prix"] = pd.to_numeric(df["prix"], errors="coerce")
    df["surface_m2"] = pd.to_numeric(df["surface_m2"], errors="coerce")
    for col in ["pieces", "nbr_chambres", "etage", "annee_construction"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ["latitude", "longitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date_publication"] = pd.to_datetime(df["date_publication"], utc=True, errors="coerce")
    return df


def charger_sources_locales() -> pd.DataFrame:
    """Charge et fusionne les fichiers JSON/CSV issus des scrapers Bien'ici, Jinka et MoteurImmo."""
    dfs = []
    if os.path.exists("jinka_annonces.json"):
        with open("jinka_annonces.json", encoding="utf-8") as f:
            ads_jinka = json.load(f)
        df_jinka = build_dataframe_jinka(ads_jinka).drop_duplicates(subset="id", keep="first")
        df_jinka["source"] = "jinka"
        dfs.append(df_jinka)

    if os.path.exists("annonces_bienici.json"):
        with open("annonces_bienici.json", "r", encoding="utf-8") as f:
            ads_bienici = json.load(f)
        df_bienici = pd.DataFrame(ads_bienici).drop_duplicates(subset="id", keep="first")
        df_bienici["source"] = "bienici"
        dfs.append(df_bienici)

    try:
        import MoteurImmo

        df_moteur = MoteurImmo.charger_tous_exports_moteurimmo()
        if not df_moteur.empty:
            dfs.append(df_moteur)
    except Exception as e:
        print(f"[Merge] Lecture MoteurImmo ignorée : {e}")

    if not dfs:
        raise FileNotFoundError("Aucun fichier annonces_bienici.json, jinka_annonces.json ou export MoteurImmo trouvé.")
    return pd.concat(dfs, ignore_index=True)


def rafraichir_scan_bienici_en_direct(max_annonces: int = 800) -> pd.DataFrame | None:
    """
    Récupère en direct les offres actuellement en ligne sur Bien'ici Grand Poitiers :
    - Permet d'ajouter les NOUVELLES offres sur Poitiers et ses communes voisines
    - Permet de détecter les anciennes offres Bien'ici qui ne sont plus en ligne (vendues/expirées)
    """
    try:
        import BienIci

        print("\n[Scan en direct Bien'ici Grand Poitiers (nouvelles offres + offres expirées)...]")
        brutes = BienIci.recuperer_annonces_grand_poitiers(
            type_operation="buy",
            max_par_commune=250,
            max_total=max_annonces,
            nouvelles_seulement=False,
        )
        if not brutes:
            return None
        annonces = [BienIci.simplifier(a) for a in brutes]
        BienIci.exporter_json(annonces, "annonces_bienici.json")
        BienIci.exporter_csv(annonces, "csv/annonces_bienici.csv")
        df_live = pd.DataFrame(annonces).drop_duplicates(subset="id", keep="first")
        df_live["source"] = "bienici"
        return df_live
    except Exception as e:
        print(f"[Info Scan Bien'ici] Scan en direct ignoré ({e.__class__.__name__}), utilisation du cache JSON local.")
        return None


def executer_pipeline(
    verifier_http: bool = False,
    limite_verif_http: int = 30,
    reentrainer: bool = True,
    rafraichir_bienici: bool = True,
):
    print("=" * 78)
    print("  PIPELINE INCRÉMENTALE : BRUT -> HISTORIQUE / RÉCENT -> TRAIN & PREDICT")
    print("=" * 78)

    # 1. Lecture de la base locale initiale (Jinka + Bien'ici) filtrée sur le département 86
    df_scan_initial = charger_sources_locales()
    if "code_postal" in df_scan_initial.columns:
        cp_s = df_scan_initial["code_postal"].astype(str).str.strip()
        df_scan_initial = df_scan_initial[cp_s.str.startswith("86") | (cp_s == "nan") | (cp_s == "")].copy()
    df_hist_brut, df_recent_brut, stats = DB.synchroniser_offres_brutes(
        df_scan_initial,
        marquer_absentes_expirees=False,
    )

    # 2. Scan en direct Bien'ici Grand Poitiers pour récupérer les nouvelles offres et marquer les disparues
    if rafraichir_bienici:
        df_live_bienici = rafraichir_scan_bienici_en_direct()
        if df_live_bienici is not None and not df_live_bienici.empty:
            df_hist_brut, df_recent_brut, stats = DB.synchroniser_offres_brutes(
                df_live_bienici,
                sources_scannees=["bienici"],
                marquer_absentes_expirees=True,
            )

    # Nettoyage strict des éventuelles anciennes annonces hors Vienne (ex: 97470 Saint-Benoît Réunion)
    if "code_postal" in df_hist_brut.columns:
        cp_h = df_hist_brut["code_postal"].astype(str).str.strip()
        mask_86 = cp_h.str.startswith("86")
        if not mask_86.all():
            nb_hors = int((~mask_86).sum())
            df_hist_brut = df_hist_brut[mask_86].copy()
            df_recent_brut = df_hist_brut[df_hist_brut["est_active"] == True].copy()  # noqa: E712
            DB.ecrire_table_sql(df_hist_brut, DB.TABLE_HIST_BRUT)
            DB.ecrire_table_sql(df_recent_brut, DB.TABLE_RECENT_BRUT)
            df_hist_brut.to_csv(DB.CSV_HIST_BRUT, index=False)
            df_recent_brut.to_csv(DB.CSV_RECENT_BRUT, index=False)
            print(f"[Nettoyage 86] {nb_hors} annonces hors Vienne (ex: 974/02) supprimées de PostgreSQL.")

    print(
        f"[Bilan Synchronisation Brute]\n"
        f"  • Nouvelles offres ajoutées     : {stats['nouvelles']}\n"
        f"  • Offres connues mises à jour   : {stats['mises_a_jour']} (dont {stats['baisses_prix']} baisses de prix)\n"
        f"  • Offres passées en expirées    : {stats['expirees']} (absentes du scan)\n"
        f"  • Total table '{DB.TABLE_HIST_BRUT}'   : {stats['total_historique']} lignes (actives + expirées)\n"
        f"  • Total table '{DB.TABLE_RECENT_BRUT}' : {stats['total_actives']} lignes (actives uniquement)"
    )

    # Option : Vérification directe HTTP sur un échantillon d'annonces actives
    if verifier_http and len(df_recent_brut) > 0:
        print(f"\n[Vérification HTTP en direct sur {min(limite_verif_http, len(df_recent_brut))} offres actives...]")
        expirees_http = []
        for _, r in df_recent_brut.head(limite_verif_http).iterrows():
            en_ligne = DB.verifier_annonce_en_ligne_http(str(r["id"]), str(r.get("url", "")))
            if not en_ligne:
                expirees_http.append(str(r["id"]))
        if expirees_http:
            print(f"  -> {len(expirees_http)} annonces détectées hors-ligne par HTTP : marquées expirées.")
            maintenant = pd.Timestamp.now(tz="UTC").floor("s")
            mask = df_hist_brut["id"].astype(str).isin(expirees_http)
            df_hist_brut.loc[mask, "est_active"] = False
            df_hist_brut.loc[mask, "statut"] = "expiree_http"
            df_hist_brut.loc[mask, "date_desactivation"] = maintenant
            df_recent_brut = df_hist_brut[df_hist_brut["est_active"] == True].copy()  # noqa: E712
            DB.ecrire_table_sql(df_hist_brut, DB.TABLE_HIST_BRUT)
            DB.ecrire_table_sql(df_recent_brut, DB.TABLE_RECENT_BRUT)
        else:
            print("  -> Toutes les annonces vérifiées sont toujours actives en ligne.")

    # 3. Synchronisation des miniatures photos (avant Feature Engineering)
    try:
        import Photos
        print("\n[Photos] Synchronisation des miniatures compressées (384x384)...")
        df_p = Photos.synchroniser_photos(df_hist_brut, max_annonces=400, telecharger=True)
        print(f"  -> {len(df_p)} miniatures indexées dans '{DB.TABLE_PHOTOS}'.")
    except Exception as e:
        print(f"[Info Photos] Synchronisation ignorée ({e}).")

    # 4. Étape 1 : Génération des tables APRÈS Feature Engineering (FeatureEngineering.py)
    print("\n[Étape 1/3 : Construction des tables après Feature Engineering (FeatureEngineering.py)...]")
    df_hist_feat, df_recent_feat = fe.construire_tables_features(df_hist_brut, verbose=False)

    # 4. Étape 2 : Entraînement du modèle directement sur `offres_historique_features` (Entrainement.py)
    if reentrainer or not os.path.exists(tr.CHEMIN_MODELE):
        print("\n[Étape 2/3 : Entraînement du modèle sur 'offres_historique_features' (Entrainement.py)...]")
        tr.entrainer_et_sauvegarder_modele(df_hist_feat, afficher=False)

    # 5. Étape 3 : Prédiction directement sur `offres_recentes_features` -> `offres_predictions` (Prediction.py)
    print("\n[Étape 3/3 : Prédiction sur 'offres_recentes_features' -> 'offres_predictions' (Prediction.py)...]")
    df_predictions = pr.predire_sur_offres_recentes(df_recent_feat)
    df_hist_feat = DB.lire_table_ou_csv(DB.TABLE_HIST_FEAT, DB.CSV_HIST_FEAT)

    print(
        f"\n[Bilan Tables Features & Prédictions]\n"
        f"  • Table '{DB.TABLE_HIST_FEAT}'   : {len(df_hist_feat)} offres ({len(df_hist_feat.columns)} colonnes de features pures, 0 prédiction)\n"
        f"  • Table '{DB.TABLE_RECENT_FEAT}' : {len(df_recent_feat)} offres ({len(df_recent_feat.columns)} colonnes de features pures, 0 prédiction)\n"
        f"  • Table '{DB.TABLE_PREDICTIONS}' : {len(df_predictions)} offres récentes avec leurs prédictions IA & négo"
    )
    DB.afficher_etat_postgres()
    print("=" * 78)
    return df_hist_brut, df_recent_brut, df_hist_feat, df_recent_feat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline incrémentale Immobilier")
    parser.add_argument(
        "--verifier-http",
        action="store_true",
        help="Vérifie par requête HTTP directe si les annonces actives sont toujours en ligne",
    )
    args = parser.parse_args()
    executer_pipeline(verifier_http=args.verifier_http)
