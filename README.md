# Finance AI

Finance AI is a news-first market assistant. Scheduled jobs ingest current financial reporting and S&P 500 price snapshots into Supabase; the Next.js app retrieves same-day pgvector context and asks a private Hugging Face Space to write cited answers.

## Architecture

```text
GitHub Actions -> Python ingestion -> Supabase Postgres/pgvector
User -> Next.js on Vercel -> Supabase retrieval -> Hugging Face -> cited answer
GitHub Actions -> Python email job -> Resend -> subscribed users
```

The Python package is ingestion-only. Vercel owns the chat API, Supabase migrations own the schema, and `hf-space/` owns embedding and generation inference.

## Scheduled Ingestion

`.github/workflows/ingest-daily-news.yml` runs every day at 4:00 PM New York time. It:

- Reads 15 finance and business RSS feeds.
- Deduplicates canonical article URLs.
- Extracts full article text where available, with RSS-summary fallback.
- Splits articles into overlapping 260-word chunks.
- Embeds every chunk with `sentence-transformers/all-mpnet-base-v2`.
- Stores documents and vectors in Supabase and retains 30 calendar days.

`.github/workflows/ingest-sp500-prices.yml` runs on weekdays at 9:30 AM, 12:00 PM, and 4:00 PM New York time. It stores `open`, `midday`, and `close` snapshots for current S&P 500 constituents and retains 30 calendar days.

`.github/workflows/send-market-email.yml` runs every day at 4:30 PM New York time. It reads same-day news, close price movers, and active `email_subscribers`, generates a three-paragraph market summary, and sends it by email.

All scheduled workflows account for EST and EDT.

## Required Secrets

Add these under **GitHub > Settings > Secrets and variables > Actions**:

```text
SUPABASE_DATABASE_URL=postgresql://...pooler.supabase.com:6543/postgres?sslmode=require
HF_EMBEDDING_URL=https://YOUR_SPACE.hf.space/embed
HF_GENERATION_URL=https://YOUR_SPACE.hf.space/generate
HF_TOKEN=hf_...
RESEND_API_KEY=re_...
MARKET_EMAIL_FROM=Finance AI <summary@yourdomain.com>
MARKET_EMAIL_REPLY_TO=optional-reply@yourdomain.com
MARKET_EMAIL_APP_URL=https://your-vercel-app.vercel.app
```

Use the Supabase transaction pooler URL so GitHub's IPv4 runners can connect.

## Ingestion CLI

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .

export DATABASE_URL="postgresql://..."
export HF_EMBEDDING_URL="https://YOUR_SPACE.hf.space/embed"
export HF_TOKEN="hf_..."

finance-chat ingest
finance-chat ingest-prices --sp500 --snapshot close
finance-chat send-market-email --dry-run
```

Useful options:

```bash
finance-chat ingest --date 2026-06-21 --retention-days 30
finance-chat ingest --summaries-only
finance-chat ingest-prices --tickers AAPL,MSFT,NVDA --snapshot midday
finance-chat send-market-email --date 2026-06-21 --limit 5
```

## Supabase

Schema changes live in `supabase/migrations/`. Apply pending migrations with:

```bash
supabase link --project-ref gktoboieleghbtsiksdt
supabase db push
```

Primary tables:

- `documents`: source article metadata and full extracted text.
- `chunks`: retrieval text and 768-dimensional embeddings.
- `price_snapshots`: structured S&P 500 open, midday, and close observations.
- `email_subscribers`: users who should receive the daily market email.
- `market_email_deliveries`: delivery log for sent, failed, and dry-run summaries.

Do not delete applied migration files; Supabase uses them as migration history.

Add a subscriber manually while the signup UI is still under construction:

```sql
INSERT INTO email_subscribers (email)
VALUES ('you@example.com')
ON CONFLICT (email) DO UPDATE SET subscribed = true, updated_at = now();
```

## Web App

The Next.js application lives in `web/`.

```bash
cd web
npm install
npm run dev
```

Vercel environment variables:

```text
SUPABASE_DATABASE_URL=postgresql://...
HF_EMBEDDING_URL=https://YOUR_SPACE.hf.space/embed
HF_GENERATION_URL=https://YOUR_SPACE.hf.space/generate
HF_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
HF_TOKEN=hf_...
```

The Vercel route embeds the question, retrieves current-day news chunks from Supabase, and sends those chunks to Hugging Face for answer generation.

## Hugging Face Space

The Docker Space in `hf-space/` exposes:

```text
POST /embed
POST /generate
```

Stored document vectors and live query vectors must always use the same embedding model.

## Verification

```bash
python3 -m unittest discover -s tests
cd web && npm run build
```
