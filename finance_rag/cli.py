from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path
from urllib.error import URLError

from finance_rag.embeddings import HashingEmbedder, HuggingFaceEmbedder, OllamaEmbedder, SentenceTransformersEmbedder
from finance_rag.index import LocalVectorIndex
from finance_rag.llm import answer_question
from finance_rag.pgvector_store import DEFAULT_DATABASE_URL, PgVectorStore
from finance_rag.prices import (
    NEW_YORK_TZ,
    SNAPSHOT_TYPES,
    fetch_price_snapshots,
    fetch_sp500_tickers,
    parse_tickers,
)
from finance_rag.sources import fetch_todays_news, save_documents


DEFAULT_DATA_DIR = Path(".finance_rag")
DEFAULT_USER_AGENT = "FinanceAI/0.1 contact@example.com"
BACKENDS = ("local", "pgvector")
EMBEDDING_PROVIDERS = ("ollama", "huggingface", "sentence-transformers", "hashing")


def main() -> None:
    parser = argparse.ArgumentParser(description="Finance RAG chatbot CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_db = subparsers.add_parser("init-db", help="Create Postgres/pgvector tables")
    init_db.add_argument("--database-url", default=None)

    ingest = subparsers.add_parser("ingest", help="Fetch and index today's financial news")
    ingest.add_argument("--date", default=date.today().isoformat(), help="News date in YYYY-MM-DD, default today")
    ingest.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ingest.add_argument("--backend", choices=BACKENDS, default="local")
    ingest.add_argument("--database-url", default=None)
    ingest.add_argument("--embedding-provider", choices=EMBEDDING_PROVIDERS, default="ollama")
    ingest.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    ingest.add_argument(
        "--summaries-only",
        action="store_true",
        help="Skip full-article extraction and index RSS summaries only",
    )

    prices = subparsers.add_parser("ingest-prices", help="Capture a current-day S&P 500 price snapshot")
    prices.add_argument("--snapshot", choices=SNAPSHOT_TYPES, required=True)
    prices.add_argument("--sp500", action="store_true", help="Capture the current S&P 500 constituents")
    prices.add_argument("--tickers", default="", help="Optional comma-separated tickers")
    prices.add_argument(
        "--date",
        default=datetime.now(NEW_YORK_TZ).date().isoformat(),
        help="Expected New York market date in YYYY-MM-DD",
    )
    prices.add_argument("--database-url", default=None)
    prices.add_argument("--user-agent", default=DEFAULT_USER_AGENT)

    reembed = subparsers.add_parser("reembed-chunks", help="Recompute embeddings for existing pgvector chunks")
    reembed.add_argument("--database-url", default=None)
    reembed.add_argument("--embedding-provider", choices=EMBEDDING_PROVIDERS, default="huggingface")
    reembed.add_argument("--limit", type=int, default=0, help="Limit chunks for a test run")

    ask = subparsers.add_parser("ask", help="Ask one question against the local RAG index")
    ask.add_argument("question")
    ask.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ask.add_argument("--backend", choices=BACKENDS, default="local")
    ask.add_argument("--database-url", default=None)
    ask.add_argument("--embedding-provider", choices=EMBEDDING_PROVIDERS, default="ollama")
    ask.add_argument("--top-k", type=int, default=8)
    ask.add_argument("--no-synthesis", action="store_true", help="Print retrieved passages instead of a synthesized answer")

    chat = subparsers.add_parser("chat", help="Start an interactive finance chat")
    chat.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    chat.add_argument("--backend", choices=BACKENDS, default="local")
    chat.add_argument("--database-url", default=None)
    chat.add_argument("--embedding-provider", choices=EMBEDDING_PROVIDERS, default="ollama")
    chat.add_argument("--top-k", type=int, default=8)
    chat.add_argument("--no-synthesis", action="store_true", help="Print retrieved passages instead of a synthesized answer")

    args = parser.parse_args()
    if args.command == "init-db":
        _init_db(args)
    elif args.command == "ingest":
        _ingest(args)
    elif args.command == "ingest-prices":
        _ingest_prices(args)
    elif args.command == "reembed-chunks":
        _reembed_chunks(args)
    elif args.command == "ask":
        _ask(
            args.question,
            backend=args.backend,
            data_dir=Path(args.data_dir),
            database_url=args.database_url,
            embedding_provider=args.embedding_provider,
            synthesize=not args.no_synthesis,
            top_k=args.top_k,
        )
    elif args.command == "chat":
        _chat(
            backend=args.backend,
            data_dir=Path(args.data_dir),
            database_url=args.database_url,
            embedding_provider=args.embedding_provider,
            synthesize=not args.no_synthesis,
            top_k=args.top_k,
        )


def _init_db(args: argparse.Namespace) -> None:
    store = PgVectorStore(database_url=args.database_url)
    try:
        store.init_schema()
    except Exception as exc:
        raise SystemExit(_backend_error_message(exc)) from exc
    print("Postgres pgvector schema is ready.")


def _ingest(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    as_of = date.fromisoformat(args.date)
    documents = fetch_todays_news(
        as_of=as_of,
        user_agent=args.user_agent,
        fetch_full_text=not args.summaries_only,
    )

    news_count = len(documents)

    if args.backend == "pgvector":
        store = PgVectorStore(database_url=args.database_url, embedder=_make_embedder(args.embedding_provider))
        try:
            chunk_count = store.ingest_documents(documents)
        except Exception as exc:
            raise SystemExit(_backend_error_message(exc)) from exc
        print(f"Indexed {chunk_count} pgvector chunks from {news_count} news items.")
        return

    docs_path = data_dir / "documents.json"
    index_path = data_dir / "index.json"
    save_documents(documents, docs_path)
    index = LocalVectorIndex.from_documents(documents)
    index.save(index_path)

    print(f"Indexed {len(index.chunks)} local chunks from {news_count} news items.")
    print(f"Saved index to {index_path}")


def _reembed_chunks(args: argparse.Namespace) -> None:
    store = PgVectorStore(database_url=args.database_url, embedder=_make_embedder(args.embedding_provider))
    try:
        count = store.reembed_chunks(
            source_type="news",
            limit=args.limit,
            progress_callback=lambda done, total: print(f"Re-embedded {done:,}/{total:,} chunks..."),
        )
    except Exception as exc:
        raise SystemExit(_backend_error_message(exc)) from exc
    print(f"Re-embedded {count:,} chunks with {args.embedding_provider}.")


def _ingest_prices(args: argparse.Namespace) -> None:
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

    store = PgVectorStore(database_url=args.database_url)
    try:
        inserted = store.ingest_price_snapshots(snapshots)
    except Exception as exc:
        raise SystemExit(_backend_error_message(exc)) from exc

    print(f"Inserted or updated {inserted:,} {args.snapshot} price snapshots.")
    if failures:
        print(f"Skipped {len(failures):,} tickers. First failures:")
        for failure in failures[:10]:
            print(f"  {failure}")


def _ask(
    question: str,
    *,
    backend: str,
    data_dir: Path,
    database_url: str,
    embedding_provider: str,
    synthesize: bool,
    top_k: int,
) -> None:
    if backend == "pgvector":
        store = PgVectorStore(database_url=database_url, embedder=_make_embedder(embedding_provider))
        try:
            results = store.search(question, top_k=top_k)
        except Exception as exc:
            raise SystemExit(_backend_error_message(exc)) from exc
        print(answer_question(question, results, synthesize=synthesize))
        return

    index_path = data_dir / "index.json"
    if not index_path.exists():
        raise SystemExit(f"No index found at {index_path}. Run `finance-chat ingest` first.")
    index = LocalVectorIndex.load(index_path)
    results = index.search(question, top_k=top_k)
    print(answer_question(question, results, synthesize=synthesize))


def _chat(
    *,
    backend: str,
    data_dir: Path,
    database_url: str,
    embedding_provider: str,
    synthesize: bool,
    top_k: int,
) -> None:
    index = None
    store = None
    if backend == "pgvector":
        store = PgVectorStore(database_url=database_url, embedder=_make_embedder(embedding_provider))
        try:
            store.ping()
        except Exception as exc:
            raise SystemExit(_backend_error_message(exc)) from exc
    else:
        index_path = data_dir / "index.json"
        if not index_path.exists():
            raise SystemExit(f"No index found at {index_path}. Run `finance-chat ingest` first.")
        index = LocalVectorIndex.load(index_path)

    print("Finance chat ready. Ask a question, or type exit.")
    while True:
        question = input("\n> ").strip()
        if question.lower() in {"exit", "quit", ":q"}:
            break
        if not question:
            continue
        if store:
            try:
                results = store.search(question, top_k=top_k)
            except Exception as exc:
                raise SystemExit(_backend_error_message(exc)) from exc
        else:
            assert index is not None
            results = index.search(question, top_k=top_k)
        print()
        print(answer_question(question, results, synthesize=synthesize))


def _backend_error_message(exc: Exception) -> str:
    if "Hugging Face" in str(exc) or "HF_EMBEDDING_URL" in str(exc):
        return (
            "Could not get embeddings from Hugging Face.\n\n"
            "Make sure your Hugging Face Space is running and set:\n"
            "  export HF_EMBEDDING_URL=https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space/embed\n\n"
            "Then retry with the Hugging Face provider:\n"
            "  finance-chat reembed-chunks --embedding-provider huggingface --limit 25\n\n"
            f"Original error: {exc}"
        )

    if isinstance(exc, URLError) or "Ollama" in str(exc):
        return (
            "Could not get local embeddings from Ollama.\n\n"
            "Make sure Ollama is running and the embedding model is installed:\n"
            "  ollama pull nomic-embed-text\n\n"
            "Then retry with the Ollama provider:\n"
            "  finance-chat ingest --backend pgvector --embedding-provider ollama\n\n"
            f"Original error: {exc}"
        )

    return (
        "Could not connect to Postgres/pgvector.\n\n"
        "For Supabase, use the transaction pooler URL with sslmode=require.\n"
        "For local Postgres, make sure Docker Desktop is running and run:\n"
        "  docker compose up -d\n\n"
        f"Original error: {exc}"
    )


def _make_embedder(provider: str):
    if provider == "ollama":
        return OllamaEmbedder()
    if provider == "huggingface":
        return HuggingFaceEmbedder()
    if provider == "sentence-transformers":
        return SentenceTransformersEmbedder()
    if provider == "hashing":
        return HashingEmbedder()
    raise ValueError(f"Unknown embedding provider: {provider}")


if __name__ == "__main__":
    main()
