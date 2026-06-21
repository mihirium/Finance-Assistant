# Finance AI

A news-first RAG assistant for asking what happened in financial markets today. It retrieves same-day reporting, searches it with pgvector, and produces cited answers with local or Hugging Face models.

The project intentionally does not store SEC filings or historical price bars.

## Local Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .

docker compose up -d
finance-chat init-db

ollama pull nomic-embed-text
finance-chat ingest --backend pgvector --embedding-provider ollama
finance-chat chat --backend pgvector --embedding-provider ollama
```

Use `--date YYYY-MM-DD` to ingest news for a specific date. Without it, `ingest` uses today.

```bash
finance-chat ingest --backend pgvector --embedding-provider ollama --date 2026-06-20
finance-chat ask --backend pgvector --embedding-provider ollama "what happened in markets today?"
```

The default local database URL is:

```text
postgresql://finance_ai:finance_ai@localhost:5432/finance_ai
```

Override it with `DATABASE_URL` or `--database-url`.

## Architecture

1. `finance-chat ingest` reads financial RSS feeds and keeps stories published on the requested date.
2. The app chunks each story and computes a 768-dimensional embedding.
3. Postgres stores the news metadata, text, chunks, and vectors.
4. A question is embedded with the same model and matched against news chunks using pgvector cosine similarity.
5. The retrieved context is sent to the answer model, which writes a short response with citations.

## Daily 4 PM Ingestion

GitHub Actions runs the ingestion workflow every day at 4:00 PM New York time. The workflow accounts for both EST and EDT, and manual runs bypass the time check.

Add these repository secrets under **GitHub > Settings > Secrets and variables > Actions**:

```text
SUPABASE_DATABASE_URL=postgresql://...pooler.supabase.com:6543/postgres?sslmode=require
HF_EMBEDDING_URL=https://YOUR_SPACE.hf.space/embed
HF_TOKEN=hf_... # required for a private Space
```

Use the Supabase transaction pooler URL so GitHub's IPv4 runners can connect. The workflow is in `.github/workflows/ingest-daily-news.yml`. Test it immediately from **GitHub > Actions > Ingest daily financial news > Run workflow**.

The chat retrieval paths only consider stories from the current New York calendar day, preventing older but similar articles from leaking into a "what happened today" answer.

## Supabase Migration

The news-only migration empties the old document/vector corpus and drops historical prices. This is destructive by design. After it runs, ingest today's news to seed the new product.

```bash
supabase link --project-ref gktoboieleghbtsiksdt
supabase db push
```

Verify the hosted database afterward:

```sql
SELECT source_type, COUNT(*) FROM documents GROUP BY source_type;
SELECT COUNT(*) FROM chunks;
SELECT to_regclass('public.prices_daily');
```

The final query should return `NULL`. Ingest current news after applying the migration:

```bash
export DATABASE_URL="postgresql://..."
finance-chat ingest --backend pgvector --embedding-provider sentence-transformers
```

For local sentence-transformers embeddings, install the optional dependency first:

```bash
pip install -e ".[local-embeddings]"
```

## Web App

The Next.js application lives in `web/`.

```bash
cd web
npm install
npm run dev
```

For Vercel-native RAG, configure:

```text
SUPABASE_DATABASE_URL=postgresql://...
HF_EMBEDDING_URL=https://YOUR_SPACE.hf.space/embed
HF_GENERATION_URL=https://YOUR_SPACE.hf.space/generate
HF_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
HF_TOKEN=... # required for a private Space
```

The Vercel route embeds the question through Hugging Face, retrieves only `news` chunks from Supabase, and sends those chunks to the Hugging Face generation endpoint.

## Hugging Face Space

The Docker Space service lives in `hf-space/` and exposes:

```text
POST /embed
POST /generate
```

Its default embedding model is `sentence-transformers/all-mpnet-base-v2`. Stored document vectors and query vectors must always use the same embedding model.

## API

Run the Python API locally with:

```bash
finance-api
```

It listens on `http://127.0.0.1:8000` by default.

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"what happened in markets today?","embedding_provider":"ollama","top_k":8}'
```

## Tests

```bash
python3 -m unittest discover -s tests
cd web && npm run build
```

## Next Steps

- Fetch full article text where publisher terms permit it.
- Add a short retention policy for stale news.
- Extract company and ticker entities from each story for filtering.
