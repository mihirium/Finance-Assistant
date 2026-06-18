CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    published_at TIMESTAMPTZ,
    ticker TEXT,
    accession_number TEXT,
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
    ticker TEXT,
    published_at TIMESTAMPTZ,
    embedding vector(768) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks(document_id);
CREATE INDEX IF NOT EXISTS chunks_source_type_idx ON chunks(source_type);
CREATE INDEX IF NOT EXISTS chunks_ticker_idx ON chunks(ticker);
CREATE INDEX IF NOT EXISTS chunks_published_at_idx ON chunks(published_at);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS prices_daily (
    ticker TEXT NOT NULL,
    trade_date DATE NOT NULL,
    open NUMERIC(18, 6) NOT NULL,
    high NUMERIC(18, 6) NOT NULL,
    low NUMERIC(18, 6) NOT NULL,
    close NUMERIC(18, 6) NOT NULL,
    adjusted_close NUMERIC(18, 6) NOT NULL,
    volume BIGINT NOT NULL,
    source TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, trade_date)
);

CREATE INDEX IF NOT EXISTS prices_daily_trade_date_idx
    ON prices_daily(trade_date);
