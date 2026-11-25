-- Initialisation PostgreSQL / PostGIS pour IA-Immobilier V1
CREATE EXTENSION IF NOT EXISTS postgis;

-- Table de quarantaine pour les annonces rejetées par les Data Contracts
CREATE TABLE IF NOT EXISTS offres_quarantaine (
    quarantine_id BIGSERIAL PRIMARY KEY,
    id VARCHAR(128),
    source VARCHAR(32),
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
    nb_lignes INTEGER NOT NULL,
    taux_remplissage_json JSONB NOT NULL,
    sla_passed BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_dq_reports_created ON data_quality_reports (created_at DESC);

-- Snapshots de monitoring Drift (Data Drift PSI, Prediction Drift, Latences, Perf DVF)
CREATE TABLE IF NOT EXISTS model_monitoring_snapshots (
    snapshot_id BIGSERIAL PRIMARY KEY,
    model_version VARCHAR(64),
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
