-- =============================================================================
-- ClaimGuard — database schema (PostgreSQL-oriented; adapt types for SQLite/MySQL)
-- Maps to API models: ClaimInput, ClaimResult, AgentResult, blockchain/IPFS fields
-- =============================================================================

-- -----------------------------------------------------------------------------
-- claims: one row per submitted claim and consensus outcome
-- -----------------------------------------------------------------------------
CREATE TABLE claims (
    claim_id            UUID PRIMARY KEY,
    patient_id          VARCHAR(64) NOT NULL,
    provider_id         VARCHAR(64) NOT NULL,
    amount              NUMERIC(18, 2) NOT NULL,
    insurance           VARCHAR(16) NOT NULL
        CHECK (insurance IN ('CNSS', 'CNOPS')),
    documents_json      JSONB NOT NULL DEFAULT '[]'::jsonb,
    history_json        JSONB NOT NULL DEFAULT '[]'::jsonb,
    decision            VARCHAR(16) NOT NULL
        CHECK (decision IN ('APPROVED', 'REJECTED')),
    consensus_score     NUMERIC(5, 2) NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Blockchain / IPFS (nullable when not configured or claim rejected)
    tx_hash             VARCHAR(128),
    ipfs_hash           VARCHAR(128),
    ipfs_hashes_json    JSONB NOT NULL DEFAULT '[]'::jsonb,
    claim_hash          VARCHAR(128),
    zk_proof_hash       VARCHAR(128)
);

CREATE INDEX idx_claims_patient_id ON claims (patient_id);
CREATE INDEX idx_claims_provider_id ON claims (provider_id);
CREATE INDEX idx_claims_decision ON claims (decision);
CREATE INDEX idx_claims_created_at ON claims (created_at DESC);
CREATE INDEX idx_claims_tx_hash ON claims (tx_hash) WHERE tx_hash IS NOT NULL;

COMMENT ON TABLE claims IS 'Main claim record with input payload snapshot and final consensus + optional chain/IPFS proof.';
COMMENT ON COLUMN claims.claim_hash IS 'On-chain claim id hash (hex), never raw PII.';
COMMENT ON COLUMN claims.zk_proof_hash IS 'Mock ZK proof hash from backend; not raw patient data.';

-- -----------------------------------------------------------------------------
-- claim_agent_results: one row per agent per claim (5 agents per claim)
-- -----------------------------------------------------------------------------
CREATE TABLE claim_agent_results (
    id                  BIGSERIAL PRIMARY KEY,
    claim_id            UUID NOT NULL REFERENCES claims (claim_id) ON DELETE CASCADE,
    agent_name          VARCHAR(128) NOT NULL,
    agent_decision      BOOLEAN NOT NULL,
    agent_score         NUMERIC(5, 2) NOT NULL,
    reasoning           TEXT NOT NULL,
    details_json        JSONB NOT NULL DEFAULT '{}'::jsonb,
    sort_order          SMALLINT NOT NULL DEFAULT 0
);

CREATE UNIQUE INDEX uq_claim_agent ON claim_agent_results (claim_id, agent_name);
CREATE INDEX idx_agent_results_claim ON claim_agent_results (claim_id);

COMMENT ON TABLE claim_agent_results IS 'Per-agent structured JSON breakdown (Anomaly, Pattern, Identity, Document, Policy).';

-- -----------------------------------------------------------------------------
-- Optional: audit log for immutable append-only events (compliance / debugging)
-- -----------------------------------------------------------------------------
CREATE TABLE claim_audit_events (
    id                  BIGSERIAL PRIMARY KEY,
    claim_id            UUID NOT NULL REFERENCES claims (claim_id) ON DELETE CASCADE,
    event_type          VARCHAR(64) NOT NULL,
    payload_json        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_audit_claim ON claim_audit_events (claim_id, created_at DESC);

COMMENT ON TABLE claim_audit_events IS 'Optional append-only events (e.g. SUBMITTED, CHAIN_CONFIRMED).';

-- =============================================================================
-- SQLite variant (if you use SQLite instead of PostgreSQL)
-- Uncomment and use instead of the above, or run with minor type tweaks.
-- =============================================================================
/*
CREATE TABLE claims (
    claim_id            TEXT PRIMARY KEY,
    patient_id          TEXT NOT NULL,
    amount              REAL NOT NULL,
    insurance           TEXT NOT NULL CHECK (insurance IN ('CNSS', 'CNOPS')),
    documents_json      TEXT NOT NULL DEFAULT '[]',
    history_json        TEXT NOT NULL DEFAULT '[]',
    decision            TEXT NOT NULL CHECK (decision IN ('APPROVED', 'REJECTED')),
    consensus_score     REAL NOT NULL,
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    tx_hash             TEXT,
    ipfs_hash           TEXT,
    ipfs_hashes_json    TEXT NOT NULL DEFAULT '[]',
    claim_hash          TEXT,
    zk_proof_hash       TEXT
);

CREATE TABLE claim_agent_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id            TEXT NOT NULL REFERENCES claims (claim_id) ON DELETE CASCADE,
    agent_name          TEXT NOT NULL,
    agent_decision      INTEGER NOT NULL CHECK (agent_decision IN (0, 1)),
    agent_score         REAL NOT NULL,
    reasoning           TEXT NOT NULL,
    details_json        TEXT NOT NULL DEFAULT '{}',
    sort_order          INTEGER NOT NULL DEFAULT 0,
    UNIQUE (claim_id, agent_name)
);

CREATE TABLE claim_audit_events (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id            TEXT NOT NULL REFERENCES claims (claim_id) ON DELETE CASCADE,
    event_type          TEXT NOT NULL,
    payload_json        TEXT NOT NULL DEFAULT '{}',
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_claims_patient_id ON claims (patient_id);
CREATE INDEX idx_claims_created_at ON claims (created_at DESC);
CREATE INDEX idx_agent_results_claim ON claim_agent_results (claim_id);
CREATE INDEX idx_audit_claim ON claim_audit_events (claim_id, created_at DESC);
*/
