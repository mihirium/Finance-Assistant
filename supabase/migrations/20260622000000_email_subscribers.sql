CREATE TABLE IF NOT EXISTS public.email_subscribers (
    email TEXT PRIMARY KEY,
    subscribed BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.market_email_deliveries (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    email TEXT NOT NULL REFERENCES public.email_subscribers(email) ON DELETE CASCADE,
    summary_date DATE NOT NULL,
    subject TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('sent', 'failed', 'dry_run')),
    provider_message_id TEXT,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (email, summary_date)
);

CREATE INDEX IF NOT EXISTS email_subscribers_subscribed_idx
    ON public.email_subscribers(subscribed);

CREATE INDEX IF NOT EXISTS market_email_deliveries_summary_date_idx
    ON public.market_email_deliveries(summary_date);

ALTER TABLE public.email_subscribers ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.market_email_deliveries ENABLE ROW LEVEL SECURITY;
