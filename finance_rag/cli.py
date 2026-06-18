from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from urllib.error import URLError

from finance_rag.embeddings import HashingEmbedder, HuggingFaceEmbedder, OllamaEmbedder
from finance_rag.index import LocalVectorIndex
from finance_rag.llm import answer_question
from finance_rag.pgvector_store import DEFAULT_DATABASE_URL, PgVectorStore
from finance_rag.prices import (
    estimate_daily_price_rows,
    estimate_daily_price_storage_mb,
    fetch_sp500_tickers,
    fetch_yahoo_daily_bars,
    years_before,
)
from finance_rag.sources import fetch_latest_10k, fetch_todays_news, parse_tickers, save_documents
from finance_rag.sources import fetch_sec_filings, parse_forms


DEFAULT_DATA_DIR = Path(".finance_rag")
DEFAULT_USER_AGENT = "FinanceAI/0.1 contact@example.com"
BACKENDS = ("local", "pgvector")
EMBEDDING_PROVIDERS = ("ollama", "huggingface", "hashing")


def main() -> None:
    parser = argparse.ArgumentParser(description="Finance RAG chatbot CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_db = subparsers.add_parser("init-db", help="Create Postgres/pgvector tables")
    init_db.add_argument("--database-url", default=None)

    ingest = subparsers.add_parser("ingest", help="Fetch today's news and optional latest 10-K filings")
    ingest.add_argument("--tickers", default="", help="Comma-separated tickers whose latest 10-K should be indexed")
    ingest.add_argument("--date", default=date.today().isoformat(), help="News date in YYYY-MM-DD, default today")
    ingest.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ingest.add_argument("--backend", choices=BACKENDS, default="local")
    ingest.add_argument("--database-url", default=None)
    ingest.add_argument("--embedding-provider", choices=EMBEDDING_PROVIDERS, default="ollama")
    ingest.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="SEC requires a descriptive user agent with contact info",
    )

    prices = subparsers.add_parser("ingest-prices", help="Fetch historical daily price bars into Postgres")
    prices.add_argument("--tickers", default="", help="Comma-separated tickers to fetch")
    prices.add_argument("--sp500", action="store_true", help="Fetch current S&P 500 constituents")
    prices.add_argument("--years", type=int, default=10, help="Years of daily bars to fetch")
    prices.add_argument("--database-url", default=None)
    prices.add_argument("--user-agent", default=DEFAULT_USER_AGENT)

    sec = subparsers.add_parser("ingest-sec", help="Fetch historical SEC filings into pgvector")
    sec.add_argument("--tickers", default="", help="Comma-separated tickers to fetch")
    sec.add_argument("--sp500", action="store_true", help="Fetch current S&P 500 constituents")
    sec.add_argument("--years", type=int, default=1, help="Years of filings to fetch")
    sec.add_argument("--forms", default="10-K,10-Q", help="Comma-separated SEC forms, default 10-K,10-Q")
    sec.add_argument("--limit-tickers", type=int, default=0, help="Limit number of tickers for test runs")
    sec.add_argument("--start-after", default="", help="Start after this ticker in the sorted ticker list")
    sec.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    sec.add_argument("--database-url", default=None)
    sec.add_argument("--embedding-provider", choices=EMBEDDING_PROVIDERS, default="ollama")
    sec.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Re-embed filings already present in Postgres instead of skipping them",
    )
    sec.add_argument("--user-agent", default=DEFAULT_USER_AGENT)

    reembed = subparsers.add_parser("reembed-chunks", help="Recompute embeddings for existing pgvector chunks")
    reembed.add_argument("--database-url", default=None)
    reembed.add_argument("--embedding-provider", choices=EMBEDDING_PROVIDERS, default="huggingface")
    reembed.add_argument("--source-type", choices=("news", "filing"), default=None)
    reembed.add_argument("--ticker", default="", help="Optional ticker filter")
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
    elif args.command == "ingest-sec":
        _ingest_sec(args)
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
    documents = fetch_todays_news(as_of=as_of, user_agent=args.user_agent)

    for ticker in parse_tickers(args.tickers):
        filing = fetch_latest_10k(ticker=ticker, user_agent=args.user_agent, cache_dir=data_dir / "cache")
        if filing:
            documents.append(filing)

    news_count = sum(1 for doc in documents if doc.source_type == "news")
    filing_count = sum(1 for doc in documents if doc.source_type == "filing")

    if args.backend == "pgvector":
        store = PgVectorStore(database_url=args.database_url, embedder=_make_embedder(args.embedding_provider))
        try:
            chunk_count = store.ingest_documents(documents)
        except Exception as exc:
            raise SystemExit(_backend_error_message(exc)) from exc
        print(f"Indexed {chunk_count} pgvector chunks from {news_count} news items and {filing_count} filings.")
        return

    docs_path = data_dir / "documents.json"
    index_path = data_dir / "index.json"
    save_documents(documents, docs_path)
    index = LocalVectorIndex.from_documents(documents)
    index.save(index_path)

    print(f"Indexed {len(index.chunks)} local chunks from {news_count} news items and {filing_count} filings.")
    print(f"Saved index to {index_path}")


def _ingest_prices(args: argparse.Namespace) -> None:
    if args.years < 1:
        raise SystemExit("--years must be at least 1")

    tickers = parse_tickers(args.tickers)
    if args.sp500:
        try:
            tickers.extend(fetch_sp500_tickers(user_agent=args.user_agent))
        except Exception as exc:
            raise SystemExit(f"Could not fetch S&P 500 ticker list: {exc}") from exc

    tickers = sorted(set(tickers))
    if not tickers:
        raise SystemExit("Pass --tickers AAPL,MSFT or --sp500.")

    end_date = date.today()
    start_date = years_before(end_date, args.years)
    estimated_rows = estimate_daily_price_rows(ticker_count=len(tickers), years=args.years)
    low_mb, high_mb = estimate_daily_price_storage_mb(ticker_count=len(tickers), years=args.years)
    print(
        f"Fetching about {estimated_rows:,} daily bars for {len(tickers):,} tickers "
        f"from {start_date} to {end_date}."
    )
    print(f"Estimated Postgres storage with indexes: roughly {low_mb:,}-{high_mb:,} MB.")

    store = PgVectorStore(database_url=args.database_url)
    try:
        store.init_schema()
    except Exception as exc:
        raise SystemExit(_backend_error_message(exc)) from exc

    total_rows = 0
    failures: list[str] = []
    for idx, ticker in enumerate(tickers, start=1):
        try:
            bars = fetch_yahoo_daily_bars(
                ticker,
                start_date=start_date,
                end_date=end_date,
                user_agent=args.user_agent,
            )
            total_rows += store.ingest_price_bars(bars)
            print(f"[{idx}/{len(tickers)}] {ticker}: {len(bars)} rows")
        except Exception as exc:
            failures.append(f"{ticker}: {exc}")
            print(f"[{idx}/{len(tickers)}] {ticker}: failed ({exc})")

    print(f"Inserted or updated {total_rows:,} daily price rows.")
    if failures:
        print(f"{len(failures)} tickers failed:")
        for failure in failures[:20]:
            print(f"  {failure}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")


def _ingest_sec(args: argparse.Namespace) -> None:
    if args.years < 1:
        raise SystemExit("--years must be at least 1")

    forms = parse_forms(args.forms)
    if not forms:
        raise SystemExit("Pass at least one SEC form with --forms.")

    tickers = parse_tickers(args.tickers)
    if args.sp500:
        try:
            tickers.extend(fetch_sp500_tickers(user_agent=args.user_agent))
        except Exception as exc:
            raise SystemExit(f"Could not fetch S&P 500 ticker list: {exc}") from exc

    tickers = sorted(set(tickers))
    if args.start_after:
        start_after = args.start_after.upper()
        tickers = [ticker for ticker in tickers if ticker > start_after]
    if args.limit_tickers:
        tickers = tickers[: args.limit_tickers]
    if not tickers:
        raise SystemExit("Pass --tickers AAPL,MSFT or --sp500.")

    end_date = date.today()
    start_date = years_before(end_date, args.years)
    print(
        f"Fetching SEC forms {', '.join(sorted(forms))} for {len(tickers):,} tickers "
        f"from {start_date} to {end_date}."
    )
    print(f"This writes filing chunks and {args.embedding_provider} embeddings to pgvector one filing at a time.")

    store = PgVectorStore(database_url=args.database_url, embedder=_make_embedder(args.embedding_provider))
    try:
        store.init_schema()
    except Exception as exc:
        raise SystemExit(_backend_error_message(exc)) from exc

    total_filings = 0
    total_chunks = 0
    failures: list[str] = []
    cache_dir = Path(args.data_dir) / "cache"
    for idx, ticker in enumerate(tickers, start=1):
        try:
            documents = fetch_sec_filings(
                ticker=ticker,
                forms=forms,
                start_date=start_date,
                end_date=end_date,
                user_agent=args.user_agent,
                cache_dir=cache_dir,
            )
        except Exception as exc:
            failures.append(f"{ticker}: fetch failed: {exc}")
            print(f"[{idx}/{len(tickers)}] {ticker}: fetch failed ({exc})")
            continue

        ticker_chunks = 0
        skipped = 0
        for document in documents:
            if not args.refresh_existing and store.document_exists(document.id):
                skipped += 1
                continue
            try:
                chunks = store.ingest_documents([document])
            except Exception as exc:
                failures.append(f"{ticker} {document.accession_number}: ingest failed: {exc}")
                print(f"[{idx}/{len(tickers)}] {ticker}: ingest failed for {document.title} ({exc})")
                continue
            total_filings += 1
            total_chunks += chunks
            ticker_chunks += chunks
        skip_note = f", {skipped} skipped" if skipped else ""
        print(f"[{idx}/{len(tickers)}] {ticker}: {len(documents)} filings, {ticker_chunks} chunks{skip_note}")

    print(f"Inserted or updated {total_filings:,} SEC filings and {total_chunks:,} chunks.")
    if failures:
        print(f"{len(failures)} failures:")
        for failure in failures[:20]:
            print(f"  {failure}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")


def _reembed_chunks(args: argparse.Namespace) -> None:
    store = PgVectorStore(database_url=args.database_url, embedder=_make_embedder(args.embedding_provider))
    ticker = args.ticker.strip().upper() or None
    try:
        count = store.reembed_chunks(
            source_type=args.source_type,
            ticker=ticker,
            limit=args.limit,
        )
    except Exception as exc:
        raise SystemExit(_backend_error_message(exc)) from exc
    print(f"Re-embedded {count:,} chunks with {args.embedding_provider}.")


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
            '  finance-chat ingest-sec --tickers AAPL --years 1 --forms 10-K,10-Q --embedding-provider ollama --user-agent "Your Name your.email@example.com"\n\n'
            f"Original error: {exc}"
        )

    return (
        "Could not connect to Postgres/pgvector.\n\n"
        "Make sure Docker Desktop is running, then start the database:\n"
        "  docker compose up -d\n\n"
        "Then initialize and ingest:\n"
        "  finance-chat init-db\n"
        "  finance-chat ingest --backend pgvector --tickers AAPL,MSFT,NVDA "
        '--user-agent "Your Name your.email@example.com"\n\n'
        f"Original error: {exc}"
    )


def _make_embedder(provider: str):
    if provider == "ollama":
        return OllamaEmbedder()
    if provider == "huggingface":
        return HuggingFaceEmbedder()
    if provider == "hashing":
        return HashingEmbedder()
    raise ValueError(f"Unknown embedding provider: {provider}")


if __name__ == "__main__":
    main()
