CREATE TABLE IF NOT EXISTS public.price_snapshots (
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
);

CREATE INDEX IF NOT EXISTS price_snapshots_market_date_idx
    ON public.price_snapshots(market_date);

ALTER TABLE public.price_snapshots ENABLE ROW LEVEL SECURITY;
