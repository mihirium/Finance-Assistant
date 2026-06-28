from __future__ import annotations

import os
from datetime import date, datetime, time as datetime_time, timedelta
from zoneinfo import ZoneInfo

from finance_rag.email_summary import NewsItem, PriceMove
from finance_rag.embeddings import HuggingFaceEmbedder
from finance_rag.models import Document
from finance_rag.prices import PriceSnapshot
from finance_rag.text import chunk_documents


NEW_YORK_TZ = ZoneInfo("America/New_York")


class PgVectorStore:
    def __init__(
        self,
        *,
        database_url: str | None = None,
        embedder: HuggingFaceEmbedder | None = None,
    ) -> None:
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("Set DATABASE_URL to the Supabase transaction pooler URL.")
        self.embedder = embedder

    def ingest_documents(self, documents: list[Document]) -> int:
        chunks = chunk_documents(documents)
        if chunks and self.embedder is None:
            raise ValueError("An embedder is required to ingest news documents.")

        with _connect(self.database_url) as conn:
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

            if documents:
                conn.execute(
                    "DELETE FROM chunks WHERE document_id = ANY(%s::text[])",
                    ([document.id for document in documents],),
                )

            for chunk in chunks:
                assert self.embedder is not None
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
        return len(chunks)

    def prune_news(self, *, as_of: date, retention_days: int) -> int:
        cutoff = _news_retention_cutoff(as_of=as_of, retention_days=retention_days)
        with _connect(self.database_url) as conn:
            cursor = conn.execute(
                """
                DELETE FROM documents
                WHERE source_type = 'news'
                  AND published_at < %s
                """,
                (cutoff,),
            )
            return cursor.rowcount

    def ingest_price_snapshots(
        self,
        snapshots: list[PriceSnapshot],
        *,
        retention_days: int = 30,
    ) -> tuple[int, int]:
        if not snapshots:
            return 0, 0

        market_date = snapshots[0].market_date
        cutoff_date = _price_retention_cutoff(
            market_date=market_date,
            retention_days=retention_days,
        )
        if any(snapshot.market_date != market_date for snapshot in snapshots):
            raise ValueError("All price snapshots in a batch must have the same market date.")

        with _connect(self.database_url) as conn:
            with conn.cursor() as cursor:
                cursor.executemany(
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
            cursor = conn.execute("DELETE FROM price_snapshots WHERE market_date < %s", (cutoff_date,))
            pruned_count = cursor.rowcount
        return len(snapshots), pruned_count

    def fetch_active_email_subscribers(self) -> list[str]:
        with _connect(self.database_url) as conn:
            rows = conn.execute(
                """
                SELECT email
                FROM email_subscribers
                WHERE subscribed = true
                ORDER BY created_at
                """
            ).fetchall()
        return [str(row[0]) for row in rows]

    def fetch_market_news_for_email(self, *, summary_date: date, limit: int = 30) -> list[NewsItem]:
        start = datetime.combine(summary_date, datetime_time.min, tzinfo=NEW_YORK_TZ)
        end = start + timedelta(days=1)
        with _connect(self.database_url) as conn:
            rows = conn.execute(
                """
                SELECT title, url, raw_text
                FROM documents
                WHERE source_type = 'news'
                  AND published_at >= %s
                  AND published_at < %s
                ORDER BY published_at DESC NULLS LAST, updated_at DESC
                LIMIT %s
                """,
                (start, end, limit),
            ).fetchall()
        return [NewsItem(title=str(row[0]), url=str(row[1]), text=str(row[2])) for row in rows]

    def fetch_price_moves_for_email(
        self,
        *,
        summary_date: date,
        limit: int = 10,
    ) -> tuple[list[PriceMove], list[PriceMove]]:
        with _connect(self.database_url) as conn:
            gainers = conn.execute(
                """
                SELECT ticker, price::float8, ((price - previous_close) / previous_close * 100)::float8 AS pct
                FROM price_snapshots
                WHERE market_date = %s
                  AND snapshot_type = 'close'
                  AND previous_close IS NOT NULL
                  AND previous_close <> 0
                ORDER BY pct DESC
                LIMIT %s
                """,
                (summary_date, limit),
            ).fetchall()
            losers = conn.execute(
                """
                SELECT ticker, price::float8, ((price - previous_close) / previous_close * 100)::float8 AS pct
                FROM price_snapshots
                WHERE market_date = %s
                  AND snapshot_type = 'close'
                  AND previous_close IS NOT NULL
                  AND previous_close <> 0
                ORDER BY pct ASC
                LIMIT %s
                """,
                (summary_date, limit),
            ).fetchall()
        return (_price_moves_from_rows(gainers), _price_moves_from_rows(losers))

    def record_market_email_delivery(
        self,
        *,
        email: str,
        summary_date: date,
        subject: str,
        status: str,
        provider_message_id: str | None = None,
        error: str | None = None,
    ) -> None:
        with _connect(self.database_url) as conn:
            conn.execute(
                """
                INSERT INTO market_email_deliveries (
                    email, summary_date, subject, status, provider_message_id, error
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (email, summary_date) DO UPDATE SET
                    subject = EXCLUDED.subject,
                    status = EXCLUDED.status,
                    provider_message_id = EXCLUDED.provider_message_id,
                    error = EXCLUDED.error,
                    created_at = now()
                """,
                (email, summary_date, subject, status, provider_message_id, error),
            )


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


def _news_retention_cutoff(*, as_of: date, retention_days: int) -> datetime:
    if retention_days < 1:
        raise ValueError("retention_days must be at least 1")
    first_kept_date = as_of - timedelta(days=retention_days - 1)
    return datetime.combine(first_kept_date, datetime_time.min, tzinfo=NEW_YORK_TZ)


def _price_retention_cutoff(*, market_date: date, retention_days: int) -> date:
    if retention_days < 1:
        raise ValueError("retention_days must be at least 1")
    return market_date - timedelta(days=retention_days - 1)


def _price_moves_from_rows(rows) -> list[PriceMove]:
    return [
        PriceMove(ticker=str(row[0]), price=float(row[1]), percent_change=float(row[2]))
        for row in rows
    ]


def _load_psycopg():
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("Install project dependencies with `pip install -e .`.") from exc
    return psycopg


def _connect(database_url: str):
    # Transaction poolers reuse server sessions, so named prepared statements
    # can collide across clients.
    return _load_psycopg().connect(database_url, prepare_threshold=None)
