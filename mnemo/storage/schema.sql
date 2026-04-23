-- memory.db schema for mnemo
-- All vector dimensions are 384 (all-MiniLM-L6-v2)

-- Core fact store
CREATE TABLE IF NOT EXISTS facts (
    id                  TEXT PRIMARY KEY,
    text                TEXT NOT NULL,
    source              TEXT DEFAULT 'user',
    project             TEXT,

    -- Timestamps (bi-temporal)
    created_at          REAL NOT NULL,
    accessed_at         REAL NOT NULL,
    valid_from          REAL,
    valid_until         REAL,

    -- Ebbinghaus strength factors (paper equation 4)
    access_count        INTEGER DEFAULT 0,
    importance          REAL DEFAULT 0.5,
    confirmations       INTEGER DEFAULT 0,
    emotional_salience  REAL DEFAULT 0.0,

    -- Trust (from paper C1 of trilogy)
    trust_score         REAL DEFAULT 1.0,

    -- Computed forgetting state
    strength            REAL,
    retention           REAL,
    lifecycle           TEXT DEFAULT 'Active',
    precision_bits      INTEGER DEFAULT 32,

    -- Deduplication
    content_hash        TEXT UNIQUE,
    superseded_by       TEXT REFERENCES facts(id)
);

-- Full-text search index for BM25 channel
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    id UNINDEXED,
    text,
    content=facts,
    content_rowid=rowid
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, id, text) VALUES (new.rowid, new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, id, text) VALUES ('delete', old.rowid, old.id, old.text);
    INSERT INTO facts_fts(rowid, id, text) VALUES (new.rowid, new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, id, text) VALUES ('delete', old.rowid, old.id, old.text);
END;

-- Embeddings (separate table for hot/cold access patterns)
CREATE TABLE IF NOT EXISTS embeddings (
    fact_id         TEXT PRIMARY KEY REFERENCES facts(id) ON DELETE CASCADE,
    vector_f32      F32_BLOB(384),
    vector_i8       BLOB,
    vector_i4       BLOB,
    vector_i2       BLOB
);

-- Entities extracted from facts
CREATE TABLE IF NOT EXISTS entities (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    type            TEXT,
    canonical_name  TEXT,
    UNIQUE(canonical_name, type)
);

-- Entity mentions
CREATE TABLE IF NOT EXISTS entity_mentions (
    fact_id         TEXT REFERENCES facts(id) ON DELETE CASCADE,
    entity_id       TEXT REFERENCES entities(id),
    PRIMARY KEY (fact_id, entity_id)
);

-- Entity co-occurrence graph
CREATE TABLE IF NOT EXISTS entity_edges (
    from_entity     TEXT REFERENCES entities(id),
    to_entity       TEXT REFERENCES entities(id),
    relation        TEXT DEFAULT 'co-occurs',
    weight          REAL DEFAULT 1.0,
    last_updated    REAL,
    PRIMARY KEY (from_entity, to_entity)
);

-- Consolidated patterns
CREATE TABLE IF NOT EXISTS patterns (
    id              TEXT PRIMARY KEY,
    pattern_type    TEXT NOT NULL,
    description     TEXT NOT NULL,
    evidence_count  INTEGER DEFAULT 0,
    positive_rate   REAL DEFAULT 0.5,
    confidence      REAL DEFAULT 0.0,
    last_updated    REAL,
    source_fact_ids TEXT
);

-- Soft prompts generated from patterns
CREATE TABLE IF NOT EXISTS soft_prompts (
    id              TEXT PRIMARY KEY,
    pattern_id      TEXT REFERENCES patterns(id),
    template_key    TEXT NOT NULL,
    rendered_text   TEXT NOT NULL,
    token_estimate  INTEGER,
    active          BOOLEAN DEFAULT TRUE,
    created_at      REAL,
    last_used_at    REAL
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_facts_lifecycle ON facts(lifecycle);
CREATE INDEX IF NOT EXISTS idx_facts_project ON facts(project);
CREATE INDEX IF NOT EXISTS idx_facts_accessed ON facts(accessed_at);
CREATE INDEX IF NOT EXISTS idx_facts_retention ON facts(retention);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_entity ON entity_mentions(entity_id);

-- learning.db tables (stored in same file for simplicity)
CREATE TABLE IF NOT EXISTS feedback_signals (
    id              TEXT PRIMARY KEY,
    fact_id         TEXT,
    signal_type     TEXT NOT NULL,
    context         TEXT,
    timestamp       REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS retrieval_log (
    id              TEXT PRIMARY KEY,
    query_text      TEXT,
    channel_scores  TEXT,
    selected_fact_ids TEXT,
    timestamp       REAL NOT NULL
);

-- sessions.db tables (stored in same file for simplicity)
CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,
    project         TEXT,
    started_at      REAL NOT NULL,
    ended_at        REAL,
    summary         TEXT,
    git_branch      TEXT,
    git_commit      TEXT,
    files_touched   TEXT,
    facts_added     INTEGER DEFAULT 0,
    facts_recalled  INTEGER DEFAULT 0
);
