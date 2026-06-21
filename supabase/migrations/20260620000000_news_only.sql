-- Start the news-only product with an empty corpus. This removes legacy SEC
-- filings as well as stale news; deleting documents cascades to vector chunks.
TRUNCATE TABLE public.documents CASCADE;

DROP TABLE IF EXISTS public.prices_daily;

DROP INDEX IF EXISTS public.chunks_ticker_idx;

ALTER TABLE public.documents
    DROP COLUMN IF EXISTS ticker,
    DROP COLUMN IF EXISTS accession_number;

ALTER TABLE public.chunks
    DROP COLUMN IF EXISTS ticker;

ALTER TABLE public.documents
    DROP CONSTRAINT IF EXISTS documents_source_type_check;

ALTER TABLE public.documents
    ADD CONSTRAINT documents_source_type_check CHECK (source_type = 'news');

ALTER TABLE public.chunks
    DROP CONSTRAINT IF EXISTS chunks_source_type_check;

ALTER TABLE public.chunks
    ADD CONSTRAINT chunks_source_type_check CHECK (source_type = 'news');

ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chunks ENABLE ROW LEVEL SECURITY;
