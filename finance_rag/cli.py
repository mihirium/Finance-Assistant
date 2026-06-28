from __future__ import annotations

import argparse
import os
from datetime import date, datetime

from finance_rag.email_summary import build_market_email, send_resend_email
from finance_rag.embeddings import HuggingFaceEmbedder
from finance_rag.pgvector_store import PgVectorStore
from finance_rag.prices import (
    NEW_YORK_TZ,
    SNAPSHOT_TYPES,
    fetch_price_snapshots,
    fetch_sp500_tickers,
    parse_tickers,
)
from finance_rag.sources import fetch_todays_news


DEFAULT_USER_AGENT = "FinanceAI/0.1 contact@example.com"


def main() -> None:
    parser = argparse.ArgumentParser(description="Finance AI ingestion service")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Fetch, embed, and store today's financial news")
    ingest.add_argument(
        "--date",
        default=datetime.now(NEW_YORK_TZ).date().isoformat(),
        help="New York news date in YYYY-MM-DD, default today",
    )
    ingest.add_argument("--database-url", default=None)
    ingest.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    ingest.add_argument(
        "--summaries-only",
        action="store_true",
        help="Skip full-article extraction and index RSS summaries only",
    )
    ingest.add_argument(
        "--retention-days",
        type=int,
        default=30,
        help="Keep this many New York calendar days in Postgres, default 30",
    )

    prices = subparsers.add_parser("ingest-prices", help="Capture an S&P 500 price snapshot")
    prices.add_argument("--snapshot", choices=SNAPSHOT_TYPES, required=True)
    prices.add_argument("--sp500", action="store_true", help="Capture current S&P 500 constituents")
    prices.add_argument("--tickers", default="", help="Optional comma-separated tickers")
    prices.add_argument(
        "--date",
        default=datetime.now(NEW_YORK_TZ).date().isoformat(),
        help="Expected New York market date in YYYY-MM-DD",
    )
    prices.add_argument("--database-url", default=None)
    prices.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    prices.add_argument(
        "--retention-days",
        type=int,
        default=30,
        help="Keep this many New York calendar days of price snapshots, default 30",
    )

    email = subparsers.add_parser("send-market-email", help="Email a daily market summary to subscribers")
    email.add_argument(
        "--date",
        default=datetime.now(NEW_YORK_TZ).date().isoformat(),
        help="New York summary date in YYYY-MM-DD, default today",
    )
    email.add_argument("--database-url", default=None)
    email.add_argument("--dry-run", action="store_true", help="Print the email instead of sending it")
    email.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max subscribers to email, useful for testing",
    )

    args = parser.parse_args()
    if args.command == "ingest":
        _ingest_news(args)
    elif args.command == "ingest-prices":
        _ingest_prices(args)
    elif args.command == "send-market-email":
        _send_market_email(args)


def _ingest_news(args: argparse.Namespace) -> None:
    as_of = date.fromisoformat(args.date)
    if args.retention_days < 1:
        raise SystemExit("--retention-days must be at least 1")

    documents = fetch_todays_news(
        as_of=as_of,
        user_agent=args.user_agent,
        fetch_full_text=not args.summaries_only,
    )
    if not documents:
        raise SystemExit(f"No news items were found for {as_of}; existing data was not changed.")

    try:
        store = PgVectorStore(
            database_url=args.database_url,
            embedder=HuggingFaceEmbedder(),
        )
        chunk_count = store.ingest_documents(documents)
        pruned_count = store.prune_news(as_of=as_of, retention_days=args.retention_days)
    except Exception as exc:
        raise SystemExit(_ingestion_error_message(exc)) from exc

    print(f"Indexed {chunk_count} pgvector chunks from {len(documents)} news items.")
    print(f"Pruned {pruned_count} news documents outside the {args.retention_days}-day window.")


def _ingest_prices(args: argparse.Namespace) -> None:
    if args.retention_days < 1:
        raise SystemExit("--retention-days must be at least 1")

    tickers = parse_tickers(args.tickers)
    if args.sp500:
        try:
            tickers.extend(fetch_sp500_tickers(user_agent=args.user_agent))
        except Exception as exc:
            raise SystemExit(f"Could not fetch the S&P 500 ticker list: {exc}") from exc
    tickers = sorted(set(tickers))
    if not tickers:
        raise SystemExit("Pass --sp500 or --tickers AAPL,MSFT.")

    market_date = date.fromisoformat(args.date)
    print(f"Fetching the {args.snapshot} snapshot for {len(tickers):,} tickers on {market_date}.")
    snapshots, failures = fetch_price_snapshots(
        tickers,
        snapshot_type=args.snapshot,
        market_date=market_date,
        user_agent=args.user_agent,
    )
    minimum_successes = max(1, int(len(tickers) * 0.8))
    if len(snapshots) < minimum_successes:
        sample = "\n".join(f"  {failure}" for failure in failures[:10])
        raise SystemExit(
            f"Only fetched {len(snapshots):,}/{len(tickers):,} prices; refusing a partial snapshot.\n{sample}"
        )

    try:
        store = PgVectorStore(database_url=args.database_url)
        inserted, pruned = store.ingest_price_snapshots(
            snapshots,
            retention_days=args.retention_days,
        )
    except Exception as exc:
        raise SystemExit(_ingestion_error_message(exc)) from exc

    print(f"Inserted or updated {inserted:,} {args.snapshot} price snapshots.")
    print(f"Pruned {pruned:,} price snapshots outside the {args.retention_days}-day window.")
    if failures:
        print(f"Skipped {len(failures):,} tickers. First failures:")
        for failure in failures[:10]:
            print(f"  {failure}")


def _ingestion_error_message(exc: Exception) -> str:
    return (
        "Finance AI ingestion failed.\n\n"
        "Confirm DATABASE_URL uses the Supabase transaction pooler with sslmode=require.\n"
        "News ingestion also requires HF_EMBEDDING_URL and, for a private Space, HF_TOKEN.\n\n"
        f"Original error: {exc}"
    )


def _send_market_email(args: argparse.Namespace) -> None:
    summary_date = date.fromisoformat(args.date)
    try:
        store = PgVectorStore(database_url=args.database_url)
        subscribers = store.fetch_active_email_subscribers()
        if args.limit is not None:
            subscribers = subscribers[: max(args.limit, 0)]
        news_items = store.fetch_market_news_for_email(summary_date=summary_date)
        gainers, losers = store.fetch_price_moves_for_email(summary_date=summary_date)
    except Exception as exc:
        raise SystemExit(_ingestion_error_message(exc)) from exc

    email = build_market_email(
        summary_date=summary_date,
        news_items=news_items,
        gainers=gainers,
        losers=losers,
        generation_url=os.getenv("HF_GENERATION_URL"),
        app_url=os.getenv("MARKET_EMAIL_APP_URL"),
    )

    if args.dry_run:
        print(email.subject)
        print()
        print(email.text)
        print()
        print(f"Would send to {len(subscribers):,} active subscribers.")
        for subscriber in subscribers:
            store.record_market_email_delivery(
                email=subscriber,
                summary_date=summary_date,
                subject=email.subject,
                status="dry_run",
            )
        return

    if not subscribers:
        print("No active email subscribers found.")
        return

    api_key = os.getenv("RESEND_API_KEY")
    sender = os.getenv("MARKET_EMAIL_FROM")
    if not api_key or not sender:
        raise SystemExit("Set RESEND_API_KEY and MARKET_EMAIL_FROM to send market emails.")

    sent = 0
    failed = 0
    for subscriber in subscribers:
        try:
            provider_id = send_resend_email(
                api_key=api_key,
                sender=sender,
                recipient=subscriber,
                subject=email.subject,
                text=email.text,
                html=email.html,
                reply_to=os.getenv("MARKET_EMAIL_REPLY_TO"),
            )
            store.record_market_email_delivery(
                email=subscriber,
                summary_date=summary_date,
                subject=email.subject,
                status="sent",
                provider_message_id=provider_id,
            )
            sent += 1
        except Exception as exc:
            store.record_market_email_delivery(
                email=subscriber,
                summary_date=summary_date,
                subject=email.subject,
                status="failed",
                error=str(exc)[:1000],
            )
            failed += 1

    print(f"Sent {sent:,} market emails for {summary_date}; {failed:,} failed.")


if __name__ == "__main__":
    main()
