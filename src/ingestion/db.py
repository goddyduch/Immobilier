"""
Gestion PostgreSQL et cycle de vie des annonces immobilières.

Gère les 4 tables demandées dans la base PostgreSQL `immobilier` :
1. `offres_historique_brut`     : Toutes les offres (actives + expirées) brutes + suivi (dates, statut, évolution prix)
2. `offres_historique_features` : Toutes les offres (actives + expirées) après Feature Engineering (base d'entraînement ML)
3. `offres_recentes_brut`       : Uniquement les offres encore actives/récentes brutes
4. `offres_recentes_features`   : Uniquement les offres encore actives/récentes après Feature Engineering + Prédictions IA
"""

import getpass
import json
import os
import uuid
from contextlib import contextmanager
import pandas as pd
import requests
from sqlalchemy import create_engine, text

USER = getpass.getuser()
DB_NAME = os.getenv("POSTGRES_DB", "immobilier")
DEFAULT_DB_URL = f"postgresql+psycopg2://{USER}@localhost:5432/{DB_NAME}"
ENGINE_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL)

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


def _backfill_geom_sql(conn, nom_table: str):
    """Met à jour la colonne PostGIS `geom` (GEOGRAPHY Point 4326) si lat/lon présents."""
    try:
        cols = {
            r[0]
            for r in conn.execute(
                text("SELECT column_name FROM information_schema.columns WHERE table_name = :t"),
                {"t": nom_table},
            ).fetchall()
        }
        if {"geom", "latitude", "longitude"}.issubset(cols):
            conn.execute(text(f"""
                UPDATE "{nom_table}"
                SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography
                WHERE geom IS NULL
                  AND latitude IS NOT NULL AND longitude IS NOT NULL
                  AND latitude BETWEEN -90 AND 90 AND longitude BETWEEN -180 AND 180
            """))
    except Exception:
        pass


def ecrire_table_sql(df: pd.DataFrame, nom_table: str):
    """Écrit un DataFrame dans PostgreSQL atomiquement tout en préservant les index PostGIS."""
    df_sql = normaliser_types_df(df).drop_duplicates(subset="id", keep="first")
    if not db_disponible():
        return False
    with engine.begin() as conn:
        exists = bool(conn.execute(
            text("SELECT to_regclass(:t) IS NOT NULL"), {"t": f'"{nom_table}"'}
        ).scalar())
        if exists:
            cols_ex = {
                r[0]
                for r in conn.execute(
                    text("SELECT column_name FROM information_schema.columns WHERE table_name = :t"),
                    {"t": nom_table},
                ).fetchall()
            }
            for col in df_sql.columns:
                if col not in cols_ex:
                    dtype = str(df_sql[col].dtype)
                    sql_type = "DOUBLE PRECISION" if "float" in dtype else "BIGINT" if "int" in dtype else "BOOLEAN" if "bool" in dtype else "TEXT"
                    conn.execute(text(f'ALTER TABLE "{nom_table}" ADD COLUMN IF NOT EXISTS "{col}" {sql_type}'))
            conn.execute(text(f'TRUNCATE TABLE "{nom_table}"'))
            df_sql.to_sql(nom_table, conn, if_exists="append", index=False, method="multi", chunksize=1000)
            _backfill_geom_sql(conn, nom_table)
        else:
            tmp_table = f"_tmp_{nom_table}_{uuid.uuid4().hex[:8]}"
            df_sql.to_sql(tmp_table, conn, if_exists="replace", index=False, method="multi", chunksize=1000)
            conn.execute(text(f'ALTER TABLE "{tmp_table}" ADD PRIMARY KEY (id)'))
            conn.execute(text(f'ALTER TABLE "{tmp_table}" RENAME TO "{nom_table}"'))
            if "code_territoire" in df_sql.columns:
                conn.execute(text(f'CREATE INDEX IF NOT EXISTS "idx_{nom_table}_terr" ON "{nom_table}" (code_territoire)'))
    print(f"[PostgreSQL] Table '{nom_table}' écrite atomiquement ({len(df_sql)} lignes, {len(df_sql.columns)} colonnes).")
    return True


def postgis_disponible(conn=None) -> bool:
    """Vérifie si l'extension PostGIS est installée sur le serveur PostgreSQL."""
    try:
        if conn is not None:
            return bool(conn.execute(text(
                "SELECT EXISTS(SELECT 1 FROM pg_available_extensions WHERE name = 'postgis')"
            )).scalar())
        if not db_disponible():
            return False
        with engine.connect() as c:
            return bool(c.execute(text(
                "SELECT EXISTS(SELECT 1 FROM pg_available_extensions WHERE name = 'postgis')"
            )).scalar())
    except Exception:
        return False


def migrer_schema_multivilles(init_sql_path: str = "db/init.sql") -> bool:
    """
    Migration NON DESTRUCTIVE vers le schéma multi-villes (12 villes-centres) :
    - Adapte automatiquement PostGIS (GEOGRAPHY/GIST) vs PostgreSQL standard (lat/lon + BTREE)
    - Peuple `ref_territoires`
    - Ajoute `code_territoire`, `est_ville_centre` (+ `geom` si PostGIS) sans perte de données.
    """
    if not db_disponible():
        return False
    from src.ingestion.territoires import VILLES_CENTRES

    with open(init_sql_path, "r", encoding="utf-8") as f:
        ddl = f.read()

    with engine.begin() as conn:
        has_pgis = postgis_disponible(conn)
        if not has_pgis:
            print("[PostgreSQL] Extension PostGIS absente localement -> bascule propre sur index BTREE lat/lon.")

        for stmt in [s.strip() for s in ddl.split(";") if s.strip()]:
            if not has_pgis:
                upper = stmt.upper()
                if "CREATE EXTENSION IF NOT EXISTS POSTGIS" in upper:
                    continue
                if "USING GIST" in upper:
                    continue
                stmt = (
                    stmt.replace("GEOGRAPHY(Point, 4326) NOT NULL", "TEXT")
                    .replace("GEOGRAPHY(Point, 4326)", "TEXT")
                    .replace("GEOMETRY(Geometry, 4326)", "TEXT")
                )
            conn.execute(text(stmt))

        for code, cfg in VILLES_CENTRES.items():
            if has_pgis:
                conn.execute(
                    text("""
                        INSERT INTO ref_territoires (
                            code_territoire, nom, departement, code_insee, codes_postaux, rayon_urbain_km,
                            centre_geom, gare_geom, campus_geom, chu_geom, actif
                        ) VALUES (
                            :code, :nom, :dept, :insee, :cps, :rayon,
                            ST_SetSRID(ST_MakePoint(:c_lon, :c_lat), 4326)::geography,
                            ST_SetSRID(ST_MakePoint(:g_lon, :g_lat), 4326)::geography,
                            ST_SetSRID(ST_MakePoint(:u_lon, :u_lat), 4326)::geography,
                            ST_SetSRID(ST_MakePoint(:h_lon, :h_lat), 4326)::geography,
                            TRUE
                        )
                        ON CONFLICT (code_territoire) DO UPDATE SET
                            nom = EXCLUDED.nom,
                            departement = EXCLUDED.departement,
                            code_insee = EXCLUDED.code_insee,
                            codes_postaux = EXCLUDED.codes_postaux,
                            rayon_urbain_km = EXCLUDED.rayon_urbain_km,
                            centre_geom = EXCLUDED.centre_geom,
                            gare_geom = EXCLUDED.gare_geom,
                            campus_geom = EXCLUDED.campus_geom,
                            chu_geom = EXCLUDED.chu_geom,
                            updated_at = NOW()
                    """),
                    {
                        "code": code, "nom": cfg.nom, "dept": cfg.departement,
                        "insee": cfg.code_insee, "cps": list(cfg.codes_postaux), "rayon": cfg.rayon_urbain_km,
                        "c_lon": cfg.centre_lon, "c_lat": cfg.centre_lat,
                        "g_lon": cfg.gare_lon, "g_lat": cfg.gare_lat,
                        "u_lon": cfg.campus_lon, "u_lat": cfg.campus_lat,
                        "h_lon": cfg.chu_lon, "h_lat": cfg.chu_lat,
                    },
                )
            else:
                conn.execute(
                    text("""
                        INSERT INTO ref_territoires (
                            code_territoire, nom, departement, code_insee, codes_postaux, rayon_urbain_km, actif
                        ) VALUES (
                            :code, :nom, :dept, :insee, :cps, :rayon, TRUE
                        )
                        ON CONFLICT (code_territoire) DO UPDATE SET
                            nom = EXCLUDED.nom,
                            departement = EXCLUDED.departement,
                            code_insee = EXCLUDED.code_insee,
                            codes_postaux = EXCLUDED.codes_postaux,
                            rayon_urbain_km = EXCLUDED.rayon_urbain_km,
                            updated_at = NOW()
                    """),
                    {
                        "code": code, "nom": cfg.nom, "dept": cfg.departement,
                        "insee": cfg.code_insee, "cps": list(cfg.codes_postaux), "rayon": cfg.rayon_urbain_km,
                    },
                )

        tables_offres = [
            TABLE_HIST_BRUT, TABLE_HIST_FEAT, TABLE_RECENT_BRUT, TABLE_RECENT_FEAT, TABLE_PREDICTIONS
        ]
        for t in tables_offres:
            exists = bool(conn.execute(text("SELECT to_regclass(:t) IS NOT NULL"), {"t": f'"{t}"'}).scalar())
            if not exists:
                continue
            conn.execute(text(f'ALTER TABLE "{t}" ADD COLUMN IF NOT EXISTS code_territoire VARCHAR(32) DEFAULT \'poitiers\''))
            conn.execute(text(f'ALTER TABLE "{t}" ADD COLUMN IF NOT EXISTS est_ville_centre BOOLEAN DEFAULT TRUE'))
            if has_pgis:
                conn.execute(text(f'ALTER TABLE "{t}" ADD COLUMN IF NOT EXISTS geom GEOGRAPHY(Point, 4326)'))
            conn.execute(text(f'UPDATE "{t}" SET code_territoire = \'poitiers\' WHERE code_territoire IS NULL'))
            cols = {
                r[0]
                for r in conn.execute(
                    text("SELECT column_name FROM information_schema.columns WHERE table_name = :t"),
                    {"t": t},
                ).fetchall()
            }
            if "code_postal" in cols:
                conn.execute(text(f"""
                    UPDATE "{t}"
                    SET est_ville_centre = FALSE
                    WHERE code_territoire = 'poitiers'
                      AND code_postal IS NOT NULL
                      AND TRIM(CAST(code_postal AS TEXT)) NOT IN ('86000', '86000.0')
                """))
            if has_pgis:
                _backfill_geom_sql(conn, t)
                conn.execute(text(f'CREATE INDEX IF NOT EXISTS "idx_{t}_geom" ON "{t}" USING GIST (geom)'))
            elif {"latitude", "longitude"}.issubset(cols):
                conn.execute(text(f'CREATE INDEX IF NOT EXISTS "idx_{t}_lat_lon" ON "{t}" (latitude, longitude)'))
            conn.execute(text(f'CREATE INDEX IF NOT EXISTS "idx_{t}_terr_centre" ON "{t}" (code_territoire, est_ville_centre)'))

    print("[PostgreSQL] Migration non destructive multi-villes (12 villes-centres) terminée.")
    return True


def get_ids_connus(source: str | None = None, code_territoire: str | None = None) -> set[str]:
    """Retourne les IDs d'annonces déjà connus en base (pour scraping incrémental)."""
    if not db_disponible():
        return set()
    clauses = []
    params: dict = {}
    if source:
        clauses.append("source = :src")
        params["src"] = source
    if code_territoire:
        clauses.append("code_territoire = :terr")
        params["terr"] = code_territoire
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(f'SELECT id FROM "{TABLE_HIST_BRUT}"{where}'), params).fetchall()
        return {str(r[0]) for r in rows if r[0] is not None}
    except Exception:
        return set()


@contextmanager
def verrou_pipeline(lock_id: int = 86001):
    """Advisory lock PostgreSQL pour empêcher le chevauchement de deux jobs batch."""
    if not db_disponible():
        yield True
        return
    with engine.connect() as conn:
        acquired = bool(conn.execute(text("SELECT pg_try_advisory_lock(:lid)"), {"lid": lock_id}).scalar())
        try:
            yield acquired
        finally:
            if acquired:
                conn.execute(text("SELECT pg_advisory_unlock(:lid)"), {"lid": lock_id})


@contextmanager
def verrou_pipeline_territoire(source: str, code_territoire: str):
    """Advisory lock déterministe scopé sur `(source, code_territoire)`."""
    import zlib
    lid = zlib.crc32(f"{source}:{code_territoire}".encode("utf-8")) & 0x7FFFFFFF
    with verrou_pipeline(lock_id=lid) as acq:
        yield acq


def est_en_cooldown(source: str, code_territoire: str) -> bool:
    """Vérifie si la source ou le couple `(source, code_territoire)` est en cooldown (ex. 3h après blocage)."""
    if not db_disponible():
        return False
    try:
        with engine.connect() as conn:
            src_cd = conn.execute(
                text("SELECT cooldown_until > NOW() FROM source_circuit_breaker WHERE source = :s"),
                {"s": source},
            ).scalar()
            if src_cd:
                return True
            terr_cd = conn.execute(
                text("SELECT cooldown_until > NOW() FROM scraping_state WHERE source = :s AND code_territoire = :t"),
                {"s": source, "t": code_territoire},
            ).scalar()
            return bool(terr_cd)
    except Exception:
        return False


def enregistrer_succes_scrape(source: str, code_territoire: str, nb_annonces: int, scrape_complet: bool = True):
    """Enregistre un passage réussi et réinitialise le compteur d'échecs."""
    if not db_disponible():
        return
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO scraping_state (
                        source, code_territoire, last_scraped_at, last_success_at,
                        last_scrape_status, scrape_complet, cooldown_until, consecutive_failures,
                        nb_annonces_vues, last_error, updated_at
                    ) VALUES (
                        :s, :t, NOW(), NOW(), 'ok', :comp, NULL, 0, :nb, NULL, NOW()
                    )
                    ON CONFLICT (source, code_territoire) DO UPDATE SET
                        last_scraped_at = NOW(),
                        last_success_at = NOW(),
                        last_scrape_status = 'ok',
                        scrape_complet = EXCLUDED.scrape_complet,
                        cooldown_until = NULL,
                        consecutive_failures = 0,
                        nb_annonces_vues = EXCLUDED.nb_annonces_vues,
                        last_error = NULL,
                        updated_at = NOW()
                """),
                {"s": source, "t": code_territoire, "comp": bool(scrape_complet), "nb": int(nb_annonces)},
            )
            conn.execute(
                text("""
                    INSERT INTO source_circuit_breaker (source, cooldown_until, blocked_reason, consecutive_blocks, updated_at)
                    VALUES (:s, NULL, NULL, 0, NOW())
                    ON CONFLICT (source) DO UPDATE SET
                        cooldown_until = NULL, blocked_reason = NULL, consecutive_blocks = 0, updated_at = NOW()
                """),
                {"s": source},
            )
    except Exception as e:
        print(f"[ScrapingState] Erreur enregistrer_succes_scrape : {e}")


def enregistrer_blocage_scrape(source: str, code_territoire: str, motif: str, cooldown_hours: int = 3):
    """Circuit-breaker : pose un cooldown de 3h en DB lorsqu'un site bloque le scraping."""
    if not db_disponible():
        return
    try:
        with engine.begin() as conn:
            conn.execute(
                text(f"""
                    INSERT INTO scraping_state (
                        source, code_territoire, last_scraped_at, last_scrape_status, scrape_complet,
                        cooldown_until, consecutive_failures, last_error, updated_at
                    ) VALUES (
                        :s, :t, NOW(), 'blocked', FALSE,
                        NOW() + INTERVAL '{int(cooldown_hours)} hours', 1, :m, NOW()
                    )
                    ON CONFLICT (source, code_territoire) DO UPDATE SET
                        last_scraped_at = NOW(),
                        last_scrape_status = 'blocked',
                        scrape_complet = FALSE,
                        cooldown_until = NOW() + INTERVAL '{int(cooldown_hours)} hours',
                        consecutive_failures = scraping_state.consecutive_failures + 1,
                        last_error = EXCLUDED.last_error,
                        updated_at = NOW()
                """),
                {"s": source, "t": code_territoire, "m": str(motif)[:500]},
            )
            conn.execute(
                text(f"""
                    INSERT INTO source_circuit_breaker (
                        source, cooldown_until, blocked_reason, blocked_at, consecutive_blocks, updated_at
                    ) VALUES (
                        :s, NOW() + INTERVAL '{int(cooldown_hours)} hours', :m, NOW(), 1, NOW()
                    )
                    ON CONFLICT (source) DO UPDATE SET
                        cooldown_until = NOW() + INTERVAL '{int(cooldown_hours)} hours',
                        blocked_reason = EXCLUDED.blocked_reason,
                        blocked_at = NOW(),
                        consecutive_blocks = source_circuit_breaker.consecutive_blocks + 1,
                        updated_at = NOW()
                """),
                {"s": source, "m": str(motif)[:500]},
            )
        print(f"[Circuit-Breaker] Source '{source}' ({code_territoire}) bloquée -> cooldown {cooldown_hours}h : {motif}")
    except Exception as e:
        print(f"[Circuit-Breaker] Erreur enregistrement cooldown : {e}")


def prochaine_ville_eligible(source: str = "bienici", min_interval_minutes: int = 80) -> str | None:
    """
    Sélectionne dynamiquement la prochaine ville-centre éligible (cadence 1h20 = 80 min),
    en sautant toute ville en cooldown 3h sans décaler les autres.
    """
    if not db_disponible():
        return "poitiers"
    try:
        with engine.connect() as conn:
            last_global = conn.execute(
                text("""
                    SELECT MAX(last_scraped_at)
                    FROM scraping_state
                    WHERE last_scrape_status = 'ok'
                """)
            ).scalar()
            if last_global is not None:
                recent = conn.execute(
                    text("SELECT :lg > NOW() - (:mins || ' minutes')::INTERVAL"),
                    {"lg": last_global, "mins": int(min_interval_minutes)},
                ).scalar()
                if recent:
                    return None

            src_blocked = conn.execute(
                text("SELECT cooldown_until > NOW() FROM source_circuit_breaker WHERE source = :s"),
                {"s": source},
            ).scalar()
            if src_blocked:
                return None

            row = conn.execute(
                text("""
                    SELECT rt.code_territoire
                    FROM ref_territoires rt
                    LEFT JOIN scraping_state ss
                      ON ss.code_territoire = rt.code_territoire AND ss.source = :s
                    WHERE rt.actif = TRUE
                      AND (ss.cooldown_until IS NULL OR ss.cooldown_until <= NOW())
                    ORDER BY ss.last_scraped_at ASC NULLS FIRST, rt.code_territoire ASC
                    LIMIT 1
                """),
                {"s": source},
            ).fetchone()
            return str(row[0]) if row else None
    except Exception:
        return None


def get_etat_scraping() -> dict[str, dict]:
    """Retourne fraîcheur et statut cooldown par ville pour le Radar / Monitoring."""
    if not db_disponible():
        return {}
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT
                    rt.code_territoire,
                    rt.nom,
                    MAX(ss.last_success_at) AS last_success_at,
                    MAX(ss.cooldown_until) AS cooldown_until,
                    BOOL_OR(COALESCE(ss.cooldown_until > NOW(), FALSE)) AS en_cooldown
                FROM ref_territoires rt
                LEFT JOIN scraping_state ss ON ss.code_territoire = rt.code_territoire
                GROUP BY rt.code_territoire, rt.nom
            """)).fetchall()
        out = {}
        for code, nom, last_ok, cd_until, en_cd in rows:
            out[str(code)] = {
                "nom": str(nom),
                "last_success_at": last_ok.isoformat() if last_ok else None,
                "cooldown_until": cd_until.isoformat() if cd_until else None,
                "en_cooldown": bool(en_cd),
            }
        return out
    except Exception:
        return {}


def enregistrer_quarantaine(df_rejets: pd.DataFrame, motif_col: str = "motif_rejet"):
    """Enregistre les annonces rejetées par les Data Contracts dans `offres_quarantaine`."""
    if df_rejets is None or df_rejets.empty or not db_disponible():
        return
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS offres_quarantaine (
                    quarantine_id BIGSERIAL PRIMARY KEY,
                    id VARCHAR(128),
                    source VARCHAR(32),
                    motif_rejet TEXT NOT NULL,
                    raw_payload JSONB NOT NULL,
                    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """))
            for _, row in df_rejets.iterrows():
                payload = {k: (None if pd.isna(v) else str(v) if not isinstance(v, (int, float, bool)) else v)
                           for k, v in row.items() if k != motif_col}
                conn.execute(
                    text("""
                        INSERT INTO offres_quarantaine (id, source, motif_rejet, raw_payload)
                        VALUES (:id, :src, :motif, CAST(:payload AS JSONB))
                    """),
                    {
                        "id": str(row.get("id", "")),
                        "src": str(row.get("source", "")),
                        "motif": str(row.get(motif_col, "anomalie")),
                        "payload": json.dumps(payload, ensure_ascii=False),
                    },
                )
    except Exception as e:
        print(f"[Quarantaine] Enregistrement ignoré : {e}")


def log_pipeline_run(step_name: str, status: str, stats: dict | None = None, error: str | None = None, run_id: str | None = None) -> str:
    """Enregistre ou met à jour un run dans `pipeline_runs`."""
    rid = run_id or str(uuid.uuid4())
    if not db_disponible():
        return rid
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    run_id VARCHAR(64) PRIMARY KEY,
                    step_name VARCHAR(64) NOT NULL,
                    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    finished_at TIMESTAMPTZ,
                    status VARCHAR(24) NOT NULL DEFAULT 'running',
                    stats_json JSONB,
                    error_message TEXT
                )
            """))
            if status == "running":
                conn.execute(
                    text("INSERT INTO pipeline_runs (run_id, step_name, status) VALUES (:rid, :step, :st) ON CONFLICT (run_id) DO NOTHING"),
                    {"rid": rid, "step": step_name, "st": status},
                )
            else:
                conn.execute(
                    text("""
                        INSERT INTO pipeline_runs (run_id, step_name, status, finished_at, stats_json, error_message)
                        VALUES (:rid, :step, :st, NOW(), CAST(:stats AS JSONB), :err)
                        ON CONFLICT (run_id) DO UPDATE SET
                            status = EXCLUDED.status,
                            finished_at = NOW(),
                            stats_json = EXCLUDED.stats_json,
                            error_message = EXCLUDED.error_message
                    """),
                    {"rid": rid, "step": step_name, "st": status, "stats": json.dumps(stats or {}, ensure_ascii=False), "err": error},
                )
    except Exception:
        pass
    return rid


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
    code_territoire: str | None = None,
    scrape_complet: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Met à jour l'historique brut et la table des offres récentes brutes à partir d'un scan :
    1. Détecte les NOUVELLES offres (id jamais vu -> ajoutées avec `est_active=True`).
    2. Met à jour les offres DÉJÀ CONNUES vues dans le scan (`date_derniere_vue = maintenant`,
       détection des baisses de prix `evolution_prix_euro`).
    3. Détecte les offres PLUS À JOUR : uniquement si `marquer_absentes_expirees` ET `scrape_complet`
       sont vrais, et scopé strictement sur `(source, code_territoire)`.
    """
    from src.ingestion.territoires import est_ville_centre, resoudre_territoire

    maintenant = pd.Timestamp.now(tz="UTC").floor("s")
    df_scan = df_nouveau_scan.copy()
    df_scan = df_scan.dropna(subset=["id"]).drop_duplicates(subset="id", keep="first").copy()
    df_scan["id"] = df_scan["id"].astype(str)
    sources_existantes = df_scan["source"].tolist() if "source" in df_scan.columns else [""] * len(df_scan)
    urls_existantes = df_scan["url"].tolist() if "url" in df_scan.columns else [""] * len(df_scan)
    df_scan["source"] = [
        detecter_source(i, u, s) for i, u, s in zip(df_scan["id"], urls_existantes, sources_existantes)
    ]
    if code_territoire:
        if "code_territoire" not in df_scan.columns:
            df_scan["code_territoire"] = code_territoire
        else:
            df_scan["code_territoire"] = df_scan["code_territoire"].fillna(code_territoire)
    else:
        df_scan["code_territoire"] = [
            resoudre_territoire(cp, v) or "poitiers"
            for cp, v in zip(df_scan.get("code_postal", [""] * len(df_scan)), df_scan.get("ville", [""] * len(df_scan)))
        ]
    df_scan["est_ville_centre"] = [
        est_ville_centre(cp, v)
        for cp, v in zip(df_scan.get("code_postal", [""] * len(df_scan)), df_scan.get("ville", [""] * len(df_scan)))
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

        # 2. Offres d'une source scannée qui ont disparu du scan -> marquées expirées
        # Garde-fou : uniquement si scrape_complet=True, scopé par (source, code_territoire), et >= 50% du parc actif
        nb_expirees = 0
        if marquer_absentes_expirees and scrape_complet:
            if "code_territoire" not in df_hist.columns:
                df_hist["code_territoire"] = "poitiers"
            terrs_cibles = [code_territoire] if code_territoire else sorted(df_scan["code_territoire"].dropna().unique().tolist())
            for terr in terrs_cibles:
                for src in sources_scannees:
                    mask_parc = (df_hist["source"] == src) & (df_hist["code_territoire"] == terr)
                    nb_actives_src = int((mask_parc & (df_hist["est_active"] == True)).sum())  # noqa: E712
                    nb_scan_src = int(((df_scan["source"] == src) & (df_scan["code_territoire"] == terr)).sum())
                    if nb_actives_src >= 20 and nb_scan_src < 0.50 * nb_actives_src:
                        print(f"[Garde-fou Expiration] {src}/{terr} : scan trop petit ({nb_scan_src}/{nb_actives_src} actives), expiration ignorée.")
                        continue
                    mask_disparues = (
                        mask_parc
                        & (~df_hist["id"].isin(ids_scan))
                        & (df_hist["est_active"] == True)  # noqa: E712
                    )
                    n_exp = int(mask_disparues.sum())
                    if n_exp > 0:
                        nb_expirees += n_exp
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
    from src.ingestion import merge as Merge

    Merge.executer_pipeline()
    afficher_etat_postgres()
