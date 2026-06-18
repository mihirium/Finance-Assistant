# Finance AI

A small CLI RAG chatbot for asking what happened in markets today, grounded in same-day finance news and optional SEC 10-K context.

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
ollama pull nomic-embed-text
finance-chat ingest --tickers AAPL,MSFT --user-agent "Your Name your.email@example.com"
finance-chat chat
```

Ask questions like:

```text
what happened today in mega-cap tech?
why is apple moving today, and what 10-k risks are relevant?
what macro themes are showing up in today's finance news?
```

## How It Works

`finance-chat ingest` fetches finance RSS feeds for the selected date and the latest 10-K for any tickers you pass. It chunks the documents and stores a local BM25-style retrieval index under `.finance_rag/index.json`.

`finance-chat ask` and `finance-chat chat` retrieve the most relevant chunks and use a local Ollama chat model to write a concise cited answer. Use `--no-synthesis` to inspect the raw retrieved passages instead. Set `OLLAMA_CHAT_MODEL` to override the default chat model, `llama3.2`.

## Commands

```bash
finance-chat ingest --tickers NVDA,JPM --date 2026-06-16 --user-agent "Your Name your.email@example.com"
finance-chat ask "what happened today in banks?"
finance-chat chat
```

SEC asks automated clients to use a descriptive user agent with contact info. Pass yours with `--user-agent`.

## Postgres + pgvector

Start Postgres with pgvector:

```bash
docker compose up -d
```

Install the Postgres client dependency after pulling these changes:

```bash
pip install -e .
```

Initialize the schema:

```bash
finance-chat init-db
```

Ingest news and filings into pgvector:

```bash
finance-chat ingest --backend pgvector --embedding-provider ollama --tickers AAPL,MSFT,NVDA --user-agent "Your Name your.email@example.com"
```

Ask against pgvector:

```bash
finance-chat ask --backend pgvector --embedding-provider ollama "what happened today in AI stocks?"
finance-chat chat --backend pgvector --embedding-provider ollama
```

Inspect raw retrieval instead of a synthesized answer:

```bash
finance-chat ask --backend pgvector --embedding-provider ollama --no-synthesis "what happened today in AI stocks?"
```

The default database URL is:

```text
postgresql://finance_ai:finance_ai@localhost:5432/finance_ai
```

Override it with `DATABASE_URL` or `--database-url`.

## Web App

The Next.js chat UI lives in `web/`.

```bash
cd web
npm install
npm run dev
```

By default, the web app uses a mock `/api/chat` response so the interface works before the Python API exists. Once the backend API is available, set:

```bash
BACKEND_API_URL=http://localhost:8000
```

The frontend will proxy `POST /api/chat` requests to `BACKEND_API_URL/chat`.

For the hosted demo path, Vercel can be the backend API directly. In that setup,
do not set `BACKEND_API_URL`. Set these Vercel environment variables instead:

```text
SUPABASE_DATABASE_URL=postgresql://...
HF_EMBEDDING_URL=https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space/embed
HF_GENERATION_URL=https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space/generate
HF_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
```

The Vercel route embeds the user's question through Hugging Face, queries
Supabase pgvector, then sends the retrieved chunks to Hugging Face for the final
answer.

## Hugging Face Space

The Hugging Face model service lives in `hf-space/`.

Create a Hugging Face Space with:

```text
SDK: Docker
Hardware: CPU Basic
```

Upload or push the contents of `hf-space/`. The Space exposes:

```text
POST /embed
POST /generate
```

The default embedding model is `sentence-transformers/all-mpnet-base-v2`, which
returns 768-dimensional vectors. That matches the current pgvector schema.

Important: query embeddings and stored chunk embeddings must come from the same
model. If Supabase already contains chunks embedded with Ollama, re-embed them
after the Hugging Face Space is live:

```bash
export DATABASE_URL="postgresql://..."
export HF_EMBEDDING_URL="https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space/embed"

finance-chat reembed-chunks --embedding-provider huggingface --limit 25
finance-chat reembed-chunks --embedding-provider huggingface
```

Use the `--limit 25` run as a smoke test before re-embedding everything.

## API

The FastAPI backend exposes the RAG chat service used by the web app.

```bash
finance-api
```

By default it runs at:

```text
http://127.0.0.1:8000
```

Useful endpoints:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"what risks did Apple disclose about supply chain and manufacturing?","embedding_provider":"ollama","top_k":5}'
```

To connect the web app to the API:

```bash
cd web
BACKEND_API_URL=http://127.0.0.1:8000 npm run dev
```

## Historical Prices

Daily OHLCV prices are stored in Postgres in `prices_daily`. This is structured data, not vector data.

Ingest 10 years of daily bars for the current S&P 500:

```bash
finance-chat ingest-prices --sp500 --years 10 --user-agent "Your Name your.email@example.com"
```

Test a small batch first:

```bash
finance-chat ingest-prices --tickers AAPL,MSFT,NVDA --years 10 --user-agent "Your Name your.email@example.com"
```

Check the stored rows:

```bash
docker compose exec postgres psql -U finance_ai -d finance_ai -c "SELECT COUNT(*) FROM prices_daily;"
docker compose exec postgres psql -U finance_ai -d finance_ai -c "SELECT ticker, MIN(trade_date), MAX(trade_date), COUNT(*) FROM prices_daily GROUP BY ticker ORDER BY ticker LIMIT 20;"
```

For 500 tickers over 10 years, expect roughly 1.26 million rows. The app estimates storage before ingestion; typical Postgres storage with indexes should be in the hundreds of MB, usually well under 1 GB.

## SEC Filing Backfill

SEC filing chunks and embeddings are stored in pgvector. Ollama provides local embeddings for large backfills.

Install Ollama, then pull the default embedding model:

```bash
ollama pull nomic-embed-text
```

The default pgvector embedding provider is `ollama`, using `nomic-embed-text`. If you change `OLLAMA_EMBEDDING_MODEL`, make sure the model returns 768-dimensional vectors or recreate the pgvector schema with a matching dimension.

Ingest one ticker for the past year:

```bash
finance-chat ingest-sec --tickers AAPL --years 1 --forms 10-K,10-Q --embedding-provider ollama --user-agent "Your Name your.email@example.com"
```

Test the first 5 current S&P 500 tickers:

```bash
finance-chat ingest-sec --sp500 --limit-tickers 5 --years 1 --forms 10-K,10-Q --embedding-provider ollama --user-agent "Your Name your.email@example.com"
```

Run the full current S&P 500 past-year backfill:

```bash
finance-chat ingest-sec --sp500 --years 1 --forms 10-K,10-Q --embedding-provider ollama --user-agent "Your Name your.email@example.com"
```

Check filing/vector chunk counts:

```bash
docker compose exec postgres psql -U finance_ai -d finance_ai -c "SELECT ticker, COUNT(*) FROM documents WHERE source_type = 'filing' GROUP BY ticker ORDER BY ticker LIMIT 20;"
docker compose exec postgres psql -U finance_ai -d finance_ai -c "SELECT COUNT(*) FROM chunks WHERE source_type = 'filing';"
```

For current S&P 500 constituents, one year of `10-K` plus `10-Q` is roughly 2,000 filings before missing/delisted/form-calendar quirks.

## Tests

```bash
python3 -m unittest discover -s tests
```

## Next Useful Upgrades

- Add earnings call transcripts and 8-K filings.
- Swap the local lexical index for embeddings and a vector database.
- Add scheduled daily ingestion.
- Add ticker/entity extraction so questions automatically pull the right filings.
