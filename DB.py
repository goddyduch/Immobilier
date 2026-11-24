"""
Gestion PostgreSQL et cycle de vie des annonces immobilières.

Gère les 4 tables demandées dans la base PostgreSQL `immobilier` :
1. `offres_historique_brut`     : Toutes les offres (actives + expirées) brutes + suivi (dates, statut, évolution prix)
2. `offres_historique_features` : Toutes les offres (actives + expirées) après Feature Engineering (base d'entraînement ML)
3. `offres_recentes_brut`       : Uniquement les offres encore actives/récentes brutes
4. `offres_recentes_features`   : Uniquement les offres encore actives/récentes après Feature Engineering + Prédictions IA
"""

import getpass
import os
import pandas as pd
import requests
from sqlalchemy import create_engine, text

USER = getpass.getuser()
DB_NAME = "immobilier"
ENGINE_URL = f"postgresql+psycopg2://{USER}@localhost:5432/{DB_NAME}"

# Noms officiels des tables PostgreSQL
TABLE_HIST_BRUT = "offres_historique_brut"
TABLE_HIST_FEAT = "offres_historique_features"
TABLE_RECENT_BRUT = "offres_recentes_brut"
TABLE_RECENT_FEAT = "offres_recentes_features"
TABLE_PREDICTIONS = "offres_predictions"
TABLE_PHOTOS = "annonces_photos"

# Dossier central regroupant tous les fichiers CSV
DOSSIER_CSV = "csv"
os.makedirs(DOSSIER_CSV, exist_ok=True)

# Fichiers CSV miroirs locaux (dans le dossier csv/)
CSV_HIST_BRUT = os.path.join(DOSSIER_CSV, "annonces_historique_brut.csv")
CSV_HIST_FEAT = os.path.join(DOSSIER_CSV, "annonces_historique_features.csv")
CSV_RECENT_BRUT = os.path.join(DOSSIER_CSV, "annonces.csv")
CSV_RECENT_FEAT = os.path.join(DOSSIER_CSV, "annonces_recentes_features.csv")
CSV_PREDICTIONS = os.path.join(DOSSIER_CSV, "annonces_predictions.csv")
CSV_PHOTOS = os.path.join(DOSSIER_CSV, "annonces_photos.csv")
CSV_AUDIT_FEATURES = os.path.join(DOSSIER_CSV, "audit_features_histgb.csv")
CSV_AUDIT_ABLATION = os.path.join(DOSSIER_CSV, "audit_ablation_histgb.csv")
CSV_DVF = os.path.join(DOSSIER_CSV, "dvf_grand_poitiers.csv")

engine = create_engine(ENGINE_URL)


def db_disponible() -> bool:
    """Vérifie si le serveur PostgreSQL local est joignable."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"[Info DB] PostgreSQL non joignable ({e.__class__.__name__}), utilisation des CSV miroirs.")
        return False


def detecter_source(ad_id: str, url: str = "", source_hint: str = "") -> str:
    """Identifie la source ('jinka', 'bienici', 'moteurimmo', etc.) d'une annonce."""
    hint = str(source_hint or "").strip().lower()
    if hint and hint not in {"nan", "none", "null"}:
        return hint
    u = str(url or "").lower()
    i = str(ad_id or "").lower()
    if "moteurimmo" in u or i.startswith("mi_"):
        return "moteurimmo"
    if "leboncoin.fr" in u:
        return "leboncoin"
    if "seloger.com" in u:
        return "seloger"
    if "figaro" in u:
        return "figaro"
    if "pap.fr" in u:
        return "pap"
    if "jinka.fr" in u or (len(i) == 36 and i.count("-") == 4):
        return "jinka"
    return "bienici"


def normaliser_types_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise les types de colonnes avant écriture SQL."""
    df = df.copy()
    for col_date in ["date_publication", "date_premiere_vue", "date_derniere_vue", "date_desactivation"]:
        if col_date in df.columns:
            df[col_date] = pd.to_datetime(df[col_date], utc=True, errors="coerce")
    for col_int in ["pieces", "nbr_chambres", "etage", "annee_construction"]:
        if col_int in df.columns:
            df[col_int] = pd.to_numeric(df[col_int], errors="coerce").astype("Int64")
    for col_flt in ["prix", "prix_initial", "evolution_prix_euro", "surface_m2", "latitude", "longitude"]:
        if col_flt in df.columns:
            df[col_flt] = pd.to_numeric(df[col_flt], errors="coerce")
    return df


def nettoyer_anciennes_tables_doublons():
    """Supprime les anciennes tables obsolètes (`Brute_annonces`, `brute_annonces`, `annonces`)."""
    if not db_disponible():
        return
    try:
        with engine.begin() as conn:
            conn.execute(text('DROP TABLE IF EXISTS "Brute_annonces" CASCADE'))
            conn.execute(text('DROP TABLE IF EXISTS "brute_annonces" CASCADE'))
            conn.execute(text('DROP TABLE IF EXISTS "annonces" CASCADE'))
    except Exception:
        pass


def ecrire_table_sql(df: pd.DataFrame, nom_table: str):
    """Écrit un DataFrame dans PostgreSQL avec `id` en clé primaire."""
    df_sql = normaliser_types_df(df).drop_duplicates(subset="id", keep="first")
    if not db_disponible():
        return False
    df_sql.to_sql(nom_table, engine, if_exists="replace", index=False)
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {nom_table} ADD PRIMARY KEY (id)"))
    print(f"[PostgreSQL] Table '{nom_table}' écrite dans PostgreSQL ({len(df_sql)} lignes, {len(df_sql.columns)} colonnes).")
    return True


def lire_table_ou_csv(nom_table: str, csv_fallback: str) -> pd.DataFrame:
    """Lit une table depuis PostgreSQL, ou depuis le fichier CSV miroir si la table n'existe pas encore."""
    if db_disponible():
        try:
            with engine.connect() as conn:
                existe = conn.execute(
                    text("SELECT EXISTS (SELECT 1 FROM pg_tables WHERE schemaname='public' AND tablename=:t)"),
                    {"t": nom_table},
                ).scalar()
            if existe:
                return pd.read_sql(text(f"SELECT * FROM {nom_table}"), engine)
        except Exception:
            pass
    if os.path.exists(csv_fallback):
        return pd.read_csv(csv_fallback)
    return pd.DataFrame()


def get_ids_connus(source: str | None = None) -> set[str]:
    """
    Retourne l'ensemble des `id` d'annonces déjà présents dans l'historique.
    Permet aux scrapers (Jinka / BienIci) de ne récupérer en détail QUE les nouvelles offres !
    """
    df_hist = lire_table_ou_csv(TABLE_HIST_BRUT, CSV_HIST_BRUT)
    if df_hist.empty and os.path.exists(CSV_RECENT_BRUT):
        df_hist = pd.read_csv(CSV_RECENT_BRUT)
    if df_hist.empty or "id" not in df_hist.columns:
        return set()
    if source and "source" in df_hist.columns:
        df_hist = df_hist[df_hist["source"] == source]
    return set(df_hist["id"].dropna().astype(str).tolist())


def verifier_annonce_en_ligne_http(ad_id: str, url: str = "") -> bool:
    """
    Vérifie directement par requête HTTP si une annonce est toujours active en ligne :
    - Sur BienIci : interroge l'endpoint JSON `realEstateAd.json?id=<id>` (404 si vendue/retirée)
    - Sur Jinka : vérifie que la page ne contient pas 'annonce n'est plus disponible' / 'expirée'
    """
    source = detecter_source(ad_id, url)
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    try:
        if source == "bienici":
            r = requests.get(
                "https://www.bienici.com/realEstateAd.json",
                params={"id": ad_id},
                headers=headers,
                timeout=8,
            )
            if r.status_code == 404:
                return False
            if r.status_code == 200:
                data = r.json()
                return bool(data and data.get("id"))
        else:
            cible = url if url.startswith("http") else f"https://www.jinka.fr/ad/{ad_id}"
            r = requests.get(cible, headers=headers, timeout=8, allow_redirects=True)
            if r.status_code in (404, 410):
                return False
            txt = r.text.lower()
            if "plus disponible" in txt or "annonce expirée" in txt or "introuvable" in txt:
                return False
            return True
    except Exception:
        pass
    return True


def synchroniser_offres_brutes(
    df_nouveau_scan: pd.DataFrame,
    sources_scannees: list[str] | None = None,
    marquer_absentes_expirees: bool = True,
    jours_fraicheur_max: int = 45,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Met à jour l'historique brut et la table des offres récentes brutes à partir d'un scan :
    1. Détecte les NOUVELLES offres (id jamais vu -> ajoutées avec `est_active=True`).
    2. Met à jour les offres DÉJÀ CONNUES vues dans le scan (`date_derniere_vue = maintenant`,
       détection des baisses de prix `evolution_prix_euro`).
    3. Détecte les offres PLUS À JOUR : toute offre d'une source scannée qui n'apparaît plus
       dans le scan passe automatiquement à `est_active=False`, `statut='expiree'` et
       `date_desactivation=maintenant`.
    """
    maintenant = pd.Timestamp.now(tz="UTC").floor("s")
    df_scan = df_nouveau_scan.copy()
    df_scan = df_scan.dropna(subset=["id"]).drop_duplicates(subset="id", keep="first").copy()
    df_scan["id"] = df_scan["id"].astype(str)
    sources_existantes = df_scan["source"].tolist() if "source" in df_scan.columns else [""] * len(df_scan)
    urls_existantes = df_scan["url"].tolist() if "url" in df_scan.columns else [""] * len(df_scan)
    df_scan["source"] = [
        detecter_source(i, u, s) for i, u, s in zip(df_scan["id"], urls_existantes, sources_existantes)
    ]

    if sources_scannees is None:
        sources_scannees = sorted(df_scan["source"].unique().tolist())

    df_hist = lire_table_ou_csv(TABLE_HIST_BRUT, CSV_HIST_BRUT)

    # Initialisation si l'historique n'existe pas encore
    if df_hist.empty:
        df_hist = df_scan.copy()
        df_hist["prix_initial"] = pd.to_numeric(df_hist["prix"], errors="coerce")
        df_hist["evolution_prix_euro"] = 0.0
        df_hist["date_premiere_vue"] = maintenant
        df_hist["date_derniere_vue"] = maintenant
        df_hist["date_desactivation"] = pd.NaT
        df_hist["est_active"] = True
        df_hist["statut"] = "active"
        stats = {
            "nouvelles": len(df_hist),
            "mises_a_jour": 0,
            "baisses_prix": 0,
            "expirees": 0,
            "total_historique": len(df_hist),
            "total_actives": len(df_hist),
        }
    else:
        df_hist = normaliser_types_df(df_hist)
        df_hist["id"] = df_hist["id"].astype(str)
        if "source" not in df_hist.columns:
            df_hist["source"] = [
                detecter_source(i, u) for i, u in zip(df_hist["id"], df_hist.get("url", [""] * len(df_hist)))
            ]
        for col, default in [
            ("prix_initial", df_hist["prix"]),
            ("evolution_prix_euro", 0.0),
            ("date_premiere_vue", maintenant),
            ("date_derniere_vue", maintenant),
            ("date_desactivation", pd.NaT),
            ("est_active", True),
            ("statut", "active"),
        ]:
            if col not in df_hist.columns:
                df_hist[col] = default
        df_hist = normaliser_types_df(df_hist)

        ids_hist = set(df_hist["id"].tolist())
        ids_scan = set(df_scan["id"].tolist())

        nouvelles_ids = ids_scan - ids_hist
        communes_ids = ids_scan & ids_hist

        # 1. Offres déjà connues et toujours présentes : mise à jour prix + date_derniere_vue
        map_scan = df_scan.set_index("id").to_dict(orient="index")
        nb_baisses = 0
        for idx, row in df_hist[df_hist["id"].isin(communes_ids)].iterrows():
            ad_id = row["id"]
            nouv = map_scan[ad_id]
            nouv_prix = pd.to_numeric(nouv.get("prix"), errors="coerce")
            prix_init = pd.to_numeric(row.get("prix_initial"), errors="coerce")
            if pd.isna(prix_init) and pd.notna(nouv_prix):
                prix_init = nouv_prix
                df_hist.at[idx, "prix_initial"] = prix_init

            if pd.notna(nouv_prix):
                df_hist.at[idx, "prix"] = nouv_prix
                if pd.notna(prix_init):
                    evol = float(nouv_prix - prix_init)
                    df_hist.at[idx, "evolution_prix_euro"] = evol
                    if evol < 0:
                        nb_baisses += 1

            df_hist.at[idx, "date_derniere_vue"] = maintenant
            df_hist.at[idx, "est_active"] = True
            df_hist.at[idx, "date_desactivation"] = pd.NaT
            df_hist.at[idx, "statut"] = (
                "baisse_prix" if (df_hist.at[idx, "evolution_prix_euro"] or 0) < 0 else "active"
            )

        # 2. Offres d'une source scannée qui ont disparu du scan -> marquées expirées (plus à jour)
        nb_expirees = 0
        if marquer_absentes_expirees:
            mask_disparues = (
                df_hist["source"].isin(sources_scannees)
                & (~df_hist["id"].isin(ids_scan))
                & (df_hist["est_active"] == True)  # noqa: E712
            )
            nb_expirees = int(mask_disparues.sum())
            df_hist.loc[mask_disparues, "est_active"] = False
            df_hist.loc[mask_disparues, "statut"] = "expiree"
            df_hist.loc[mask_disparues, "date_desactivation"] = maintenant

        # 3. Ajout des nouvelles offres
        if nouvelles_ids:
            df_nouv = df_scan[df_scan["id"].isin(nouvelles_ids)].copy()
            df_nouv["prix_initial"] = pd.to_numeric(df_nouv["prix"], errors="coerce")
            df_nouv["evolution_prix_euro"] = 0.0
            df_nouv["date_premiere_vue"] = maintenant
            df_nouv["date_derniere_vue"] = maintenant
            df_nouv["date_desactivation"] = pd.NaT
            df_nouv["est_active"] = True
            df_nouv["statut"] = "nouvelle"
            df_hist = pd.concat([df_hist, df_nouv], ignore_index=True)

        stats = {
            "nouvelles": len(nouvelles_ids),
            "mises_a_jour": len(communes_ids),
            "baisses_prix": nb_baisses,
            "expirees": nb_expirees,
            "total_historique": len(df_hist),
            "total_actives": int(df_hist["est_active"].sum()),
        }

    # Filtrage des offres récentes / encore à jour
    df_recent = df_hist[df_hist["est_active"] == True].copy()  # noqa: E712

    # Sauvegarde CSV miroirs + Tables PostgreSQL (Brutes)
    df_hist.to_csv(CSV_HIST_BRUT, index=False)
    df_recent.to_csv(CSV_RECENT_BRUT, index=False)
    nettoyer_anciennes_tables_doublons()
    ecrire_table_sql(df_hist, TABLE_HIST_BRUT)
    ecrire_table_sql(df_recent, TABLE_RECENT_BRUT)

    return df_hist, df_recent, stats


def sauvegarder_tables_features(df_hist_feat: pd.DataFrame, df_recent_feat: pd.DataFrame):
    """
    Sauvegarde les 2 tables après Feature Engineering dans PostgreSQL et en CSV miroir :
    - `offres_historique_features` (toutes les offres actives + expirées après Feature Engineering)
    - `offres_recentes_features`   (uniquement les offres récentes/actives + prédictions IA)
    """
    df_hist_feat.to_csv(CSV_HIST_FEAT, index=False)
    df_recent_feat.to_csv(CSV_RECENT_FEAT, index=False)
    ecrire_table_sql(df_hist_feat, TABLE_HIST_FEAT)
    ecrire_table_sql(df_recent_feat, TABLE_RECENT_FEAT)


# Compatibilité descendante avec l'ancienne fonction
def creer_table_depuis_df(df: pd.DataFrame, nom_table: str = TABLE_HIST_BRUT):
    ecrire_table_sql(df, nom_table)


def lire(requete: str, params: dict | None = None) -> pd.DataFrame:
    return pd.read_sql(text(requete), engine, params=params)


def afficher_etat_postgres():
    """Affiche la liste des tables présentes dans PostgreSQL avec leur nombre de lignes et colonnes."""
    if not db_disponible():
        return
    with engine.connect() as conn:
        tables = conn.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename")
        ).scalars().all()
        print("\n[État actuel de la base PostgreSQL 'immobilier']")
        for t in tables:
            nb_lignes = conn.execute(text(f'SELECT COUNT(*) FROM "{t}"')).scalar()
            nb_cols = conn.execute(
                text("SELECT COUNT(*) FROM information_schema.columns WHERE table_name=:t"),
                {"t": t},
            ).scalar()
            print(f"  • {t:<30} : {nb_lignes:>4} lignes | {nb_cols:>2} colonnes")


if __name__ == "__main__":
    import Merge

    Merge.executer_pipeline()
    afficher_etat_postgres()
