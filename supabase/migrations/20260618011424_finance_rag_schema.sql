CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    published_at TIMESTAMPTZ,
    raw_text TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    source_type TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    chunk_text TEXT NOT NULL,
    published_at TIMESTAMPTZ,
    embedding vector(768) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks(document_id);
CREATE INDEX IF NOT EXISTS chunks_source_type_idx ON chunks(source_type);
CREATE INDEX IF NOT EXISTS chunks_published_at_idx ON chunks(published_at);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS price_snapshots (
    ticker TEXT NOT NULL,
    market_date DATE NOT NULL,
    snapshot_type TEXT NOT NULL CHECK (snapshot_type IN ('open', 'midday', 'close')),
    captured_at TIMESTAMPTZ NOT NULL,
    price NUMERIC(18, 6) NOT NULL,
    day_open NUMERIC(18, 6) NOT NULL,
    day_high NUMERIC(18, 6) NOT NULL,
    day_low NUMERIC(18, 6) NOT NULL,
    previous_close NUMERIC(18, 6),
    volume BIGINT NOT NULL,
    source TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, market_date, snapshot_type)
);

CREATE INDEX IF NOT EXISTS price_snapshots_market_date_idx
    ON price_snapshots(market_date);

ALTER TABLE documents
    ADD CONSTRAINT documents_source_type_check CHECK (source_type = 'news');

ALTER TABLE chunks
    ADD CONSTRAINT chunks_source_type_check CHECK (source_type = 'news');

ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE price_snapshots ENABLE ROW LEVEL SECURITY;
