from __future__ import annotations

import os
from datetime import datetime

from finance_rag.embeddings import DEFAULT_EMBEDDING_DIMENSIONS, OllamaEmbedder
from finance_rag.index import LocalVectorIndex
from finance_rag.models import Chunk, Document, SearchResult
from finance_rag.prices import PriceSnapshot


DEFAULT_DATABASE_URL = "postgresql://finance_ai:finance_ai@localhost:5432/finance_ai"


class PgVectorStore:
    def __init__(
        self,
        *,
        database_url: str | None = None,
        embedder: object | None = None,
        dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    ) -> None:
        self.database_url = database_url or os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
        self._embedder = embedder
        self.dimensions = embedder.dimensions if embedder else dimensions

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = OllamaEmbedder(dimensions=self.dimensions)
            self.dimensions = self._embedder.dimensions
        return self._embedder

    def init_schema(self) -> None:
        psycopg = _load_psycopg()
        dimensions = _validate_dimensions(self.dimensions)
        with psycopg.connect(self.database_url) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL,
                    published_at TIMESTAMPTZ,
                    raw_text TEXT NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    source_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL,
                    chunk_text TEXT NOT NULL,
                    published_at TIMESTAMPTZ,
                    embedding vector({dimensions}) NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks(document_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS chunks_source_type_idx ON chunks(source_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS chunks_published_at_idx ON chunks(published_at)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx "
                "ON chunks USING hnsw (embedding vector_cosine_ops)"
            )
            conn.execute(
                """
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
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS price_snapshots_market_date_idx "
                "ON price_snapshots(market_date)"
            )

    def ping(self) -> None:
        psycopg = _load_psycopg()
        with psycopg.connect(self.database_url) as conn:
            conn.execute("SELECT 1").fetchone()

    def ingest_documents(self, documents: list[Document]) -> int:
        psycopg = _load_psycopg()
        self.init_schema()
        index = LocalVectorIndex.from_documents(documents)
        with psycopg.connect(self.database_url) as conn:
            for document in documents:
                conn.execute(
                    """
                    INSERT INTO documents (
                        id, source_type, title, url, published_at, raw_text, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (id) DO UPDATE SET
                        source_type = EXCLUDED.source_type,
                        title = EXCLUDED.title,
                        url = EXCLUDED.url,
                        published_at = EXCLUDED.published_at,
                        raw_text = EXCLUDED.raw_text,
                        updated_at = now()
                    """,
                    (
                        document.id,
                        document.source_type,
                        document.title,
                        document.url,
                        document.published_at,
                        document.text,
                    ),
                )

            for chunk in index.chunks:
                embedding = self.embedder.embed_document(title=chunk.title, text=chunk.text)
                conn.execute(
                    """
                    INSERT INTO chunks (
                        id, document_id, source_type, title, url, chunk_text,
                        published_at, embedding, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector, now())
                    ON CONFLICT (id) DO UPDATE SET
                        source_type = EXCLUDED.source_type,
                        title = EXCLUDED.title,
                        url = EXCLUDED.url,
                        chunk_text = EXCLUDED.chunk_text,
                        published_at = EXCLUDED.published_at,
                        embedding = EXCLUDED.embedding,
                        updated_at = now()
                    """,
                    (
                        chunk.id,
                        chunk.document_id,
                        chunk.source_type,
                        chunk.title,
                        chunk.url,
                        chunk.text,
                        chunk.published_at,
                        _vector_literal(embedding),
                    ),
                )
        return len(index.chunks)

    def reembed_chunks(
        self,
        *,
        source_type: str | None = None,
        limit: int = 0,
        progress_every: int = 25,
        progress_callback: object | None = None,
    ) -> int:
        psycopg = _load_psycopg()
        filters = []
        params: list[object] = []
        if source_type:
            params.append(source_type)
            filters.append(f"source_type = %s")
        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        limit_sql = "LIMIT %s" if limit > 0 else ""
        if limit > 0:
            params.append(limit)

        with psycopg.connect(self.database_url) as conn:
            rows = conn.execute(
                f"""
                SELECT id, title, chunk_text
                FROM chunks
                {where}
                ORDER BY updated_at ASC
                {limit_sql}
                """,
                tuple(params),
            ).fetchall()

            for idx, (chunk_id, title, text) in enumerate(rows, start=1):
                embedding = self.embedder.embed_document(title=title, text=text)
                conn.execute(
                    """
                    UPDATE chunks
                    SET embedding = %s::vector, updated_at = now()
                    WHERE id = %s
                    """,
                    (_vector_literal(embedding), chunk_id),
                )
                if progress_every > 0 and idx % progress_every == 0:
                    conn.commit()
                    if progress_callback:
                        progress_callback(idx, len(rows))
        return len(rows)

    def ingest_price_snapshots(self, snapshots: list[PriceSnapshot]) -> int:
        if not snapshots:
            return 0

        psycopg = _load_psycopg()
        self.init_schema()
        market_date = snapshots[0].market_date
        if any(snapshot.market_date != market_date for snapshot in snapshots):
            raise ValueError("All price snapshots in a batch must have the same market date.")

        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO price_snapshots (
                        ticker, market_date, snapshot_type, captured_at, price,
                        day_open, day_high, day_low, previous_close, volume,
                        source, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (ticker, market_date, snapshot_type) DO UPDATE SET
                        captured_at = EXCLUDED.captured_at,
                        price = EXCLUDED.price,
                        day_open = EXCLUDED.day_open,
                        day_high = EXCLUDED.day_high,
                        day_low = EXCLUDED.day_low,
                        previous_close = EXCLUDED.previous_close,
                        volume = EXCLUDED.volume,
                        source = EXCLUDED.source,
                        updated_at = now()
                    """,
                    [
                        (
                            snapshot.ticker,
                            snapshot.market_date,
                            snapshot.snapshot_type,
                            snapshot.captured_at,
                            snapshot.price,
                            snapshot.day_open,
                            snapshot.day_high,
                            snapshot.day_low,
                            snapshot.previous_close,
                            snapshot.volume,
                            snapshot.source,
                        )
                        for snapshot in snapshots
                    ],
                )
            conn.execute("DELETE FROM price_snapshots WHERE market_date < %s", (market_date,))
        return len(snapshots)

    def search(self, query: str, *, top_k: int = 8) -> list[SearchResult]:
        psycopg = _load_psycopg()
        query_embedding = self.embedder.embed_query(query)
        vector = _vector_literal(query_embedding)
        params = (vector, vector, top_k)

        with psycopg.connect(self.database_url) as conn:
            rows = conn.execute(
                f"""
                SELECT
                    id,
                    document_id,
                    source_type,
                    title,
                    url,
                    chunk_text,
                    published_at,
                    1 - (embedding <=> %s::vector) AS score
                FROM chunks
                WHERE source_type = 'news'
                  AND published_at >= date_trunc('day', now() AT TIME ZONE 'America/New_York')
                      AT TIME ZONE 'America/New_York'
                  AND published_at < (
                      date_trunc('day', now() AT TIME ZONE 'America/New_York') + interval '1 day'
                  ) AT TIME ZONE 'America/New_York'
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                params,
            ).fetchall()

        return [
            SearchResult(
                Chunk(
                    id=row[0],
                    document_id=row[1],
                    source_type=row[2],
                    title=row[3],
                    url=row[4],
                    text=row[5],
                    published_at=_coerce_datetime(row[6]),
                ),
                float(row[7]),
            )
            for row in rows
        ]


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


def _validate_dimensions(dimensions: int) -> int:
    if dimensions < 1 or dimensions > 16000:
        raise ValueError("Embedding dimensions must be between 1 and 16000.")
    return dimensions


def _coerce_datetime(value: object) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    raise TypeError(f"Expected datetime or None, got {type(value)!r}")


def _load_psycopg():
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("Install Postgres support with `pip install -e .` before using --backend pgvector.") from exc
    return psycopg
