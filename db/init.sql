-- Initialisation et migration PostgreSQL / PostGIS pour Justeprix (12 villes-centres V1)
CREATE EXTENSION IF NOT EXISTS postgis;

-- Référentiel des 12 villes-centres (hors La Rochelle et hors couronnes périphériques)
CREATE TABLE IF NOT EXISTS ref_territoires (
    code_territoire VARCHAR(32) PRIMARY KEY,
    nom TEXT NOT NULL,
    departement VARCHAR(3) NOT NULL,
    code_insee VARCHAR(5) NOT NULL,
    codes_postaux TEXT[] NOT NULL,
    rayon_urbain_km NUMERIC(5,2) NOT NULL DEFAULT 5.5,
    centre_geom GEOGRAPHY(Point, 4326),
    gare_geom GEOGRAPHY(Point, 4326),
    campus_geom GEOGRAPHY(Point, 4326),
    chu_geom GEOGRAPHY(Point, 4326),
    bienici_zone_ids JSONB,
    jinka_search_url TEXT,
    actif BOOLEAN NOT NULL DEFAULT TRUE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_ref_territoires_centre ON ref_territoires USING GIST (centre_geom);

-- État du scraping & circuit-breaker par (source, code_territoire)
CREATE TABLE IF NOT EXISTS scraping_state (
    source VARCHAR(32) NOT NULL,
    code_territoire VARCHAR(32) NOT NULL,
    last_scraped_at TIMESTAMPTZ,
    last_success_at TIMESTAMPTZ,
    last_scrape_status VARCHAR(32) NOT NULL DEFAULT 'never',
    scrape_complet BOOLEAN NOT NULL DEFAULT FALSE,
    cooldown_until TIMESTAMPTZ,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    nb_annonces_vues INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (source, code_territoire)
);
CREATE INDEX IF NOT EXISTS idx_scraping_state_cooldown ON scraping_state (cooldown_until);

-- Circuit-breaker global par source (cooldown 3h en cas de blocage anti-bot/403/429)
CREATE TABLE IF NOT EXISTS source_circuit_breaker (
    source VARCHAR(32) PRIMARY KEY,
    cooldown_until TIMESTAMPTZ,
    blocked_reason TEXT,
    blocked_at TIMESTAMPTZ,
    consecutive_blocks INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Transactions notariales DVF (indexées PostGIS GEOGRAPHY pour ST_DWithin métrique)
CREATE TABLE IF NOT EXISTS dvf_transactions (
    dvf_uid VARCHAR(96) PRIMARY KEY,
    id_mutation VARCHAR(64),
    date_mutation DATE,
    annee_mut SMALLINT,
    valeur_fonciere DOUBLE PRECISION,
    surface_reelle_bati DOUBLE PRECISION,
    pm2 DOUBLE PRECISION,
    type_local VARCHAR(32),
    nombre_pieces_principales INTEGER,
    code_commune VARCHAR(5),
    code_postal VARCHAR(10),
    code_territoire VARCHAR(32),
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    geom GEOGRAPHY(Point, 4326)
);
CREATE INDEX IF NOT EXISTS idx_dvf_geom ON dvf_transactions USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_dvf_terr_type_date ON dvf_transactions (code_territoire, type_local, date_mutation DESC);

-- Contours & indicateurs IRIS INSEE
CREATE TABLE IF NOT EXISTS ref_iris (
    iris_code VARCHAR(16) PRIMARY KEY,
    iris_name TEXT,
    code_commune VARCHAR(5),
    departement VARCHAR(3),
    code_territoire VARCHAR(32),
    revenu_median DOUBLE PRECISION,
    taux_hlm DOUBLE PRECISION,
    part_proprietaires DOUBLE PRECISION,
    taux_vacance DOUBLE PRECISION,
    geom GEOMETRY(Geometry, 4326)
);
CREATE INDEX IF NOT EXISTS idx_ref_iris_geom ON ref_iris USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_ref_iris_terr ON ref_iris (code_territoire, code_commune);

-- Quartiers Prioritaires de la Politique de la Ville (QPV ANCT)
CREATE TABLE IF NOT EXISTS ref_qpv (
    qpv_code VARCHAR(32) PRIMARY KEY,
    qpv_nom TEXT,
    code_territoire VARCHAR(32),
    geom GEOMETRY(Geometry, 4326)
);
CREATE INDEX IF NOT EXISTS idx_ref_qpv_geom ON ref_qpv USING GIST (geom);

-- Points d'intérêt Transports & Commerces (batch BPE INSEE / Geofabrik OSM)
CREATE TABLE IF NOT EXISTS ref_poi (
    poi_id BIGSERIAL PRIMARY KEY,
    code_territoire VARCHAR(32) NOT NULL,
    categorie VARCHAR(64) NOT NULL,
    nom TEXT,
    source VARCHAR(32) NOT NULL DEFAULT 'bpe_osm',
    geom GEOGRAPHY(Point, 4326) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ref_poi_geom ON ref_poi USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_ref_poi_terr_cat ON ref_poi (code_territoire, categorie);

-- Table de quarantaine pour les annonces rejetées par les Data Contracts
CREATE TABLE IF NOT EXISTS offres_quarantaine (
    quarantine_id BIGSERIAL PRIMARY KEY,
    id VARCHAR(128),
    source VARCHAR(32),
    code_territoire VARCHAR(32),
    motif_rejet TEXT NOT NULL,
    raw_payload JSONB NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_quarantaine_detected ON offres_quarantaine (detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_quarantaine_source_motif ON offres_quarantaine (source, motif_rejet);

-- Historique des exécutions de pipeline
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id VARCHAR(64) PRIMARY KEY,
    step_name VARCHAR(64) NOT NULL,
    code_territoire VARCHAR(32),
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    status VARCHAR(24) NOT NULL DEFAULT 'running',
    stats_json JSONB,
    error_message TEXT
);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_started ON pipeline_runs (started_at DESC);

-- Rapports de qualité SLA des données
CREATE TABLE IF NOT EXISTS data_quality_reports (
    report_id BIGSERIAL PRIMARY KEY,
    run_id VARCHAR(64),
    table_cible VARCHAR(64) NOT NULL,
    code_territoire VARCHAR(32),
    nb_lignes INTEGER NOT NULL,
    taux_remplissage_json JSONB NOT NULL,
    sla_passed BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_dq_reports_created ON data_quality_reports (created_at DESC);

-- Snapshots de monitoring Drift (global + découpé par ville)
CREATE TABLE IF NOT EXISTS model_monitoring_snapshots (
    snapshot_id BIGSERIAL PRIMARY KEY,
    model_version VARCHAR(64),
    code_territoire VARCHAR(32) DEFAULT 'global',
    nb_historique INTEGER,
    nb_actives INTEGER,
    nb_quarantaine INTEGER,
    completude_json JSONB,
    data_drift_psi_json JSONB,
    prediction_drift_json JSONB,
    perf_dvf_json JSONB,
    latences_json JSONB,
    statut_global VARCHAR(24) NOT NULL DEFAULT 'ok',
    alertes_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_monitoring_created ON model_monitoring_snapshots (created_at DESC);
ALTER TABLE model_monitoring_snapshots ADD COLUMN IF NOT EXISTS code_territoire VARCHAR(32) DEFAULT 'global';
CREATE INDEX IF NOT EXISTS idx_monitoring_terr ON model_monitoring_snapshots (code_territoire, created_at DESC);
