from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser


SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


@dataclass(frozen=True)
class PriceBar:
    ticker: str
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    adjusted_close: float
    volume: int
    source: str = "yahoo"


def fetch_yahoo_daily_bars(
    ticker: str,
    *,
    start_date: date,
    end_date: date,
    user_agent: str,
) -> list[PriceBar]:
    yahoo_ticker = to_yahoo_ticker(ticker)
    params = urllib.parse.urlencode(
        {
            "period1": _unix_seconds(start_date),
            "period2": _unix_seconds(end_date + timedelta(days=1)),
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        }
    )
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(yahoo_ticker)}?{params}"
    data = json.loads(_request(url, user_agent=user_agent))
    return parse_yahoo_chart_response(data, ticker=ticker.upper())


def parse_yahoo_chart_response(data: dict, *, ticker: str) -> list[PriceBar]:
    result = data.get("chart", {}).get("result") or []
    if not result:
        error = data.get("chart", {}).get("error")
        if error:
            raise ValueError(f"Yahoo chart error for {ticker}: {error}")
        return []

    chart = result[0]
    timestamps = chart.get("timestamp") or []
    quote = (chart.get("indicators", {}).get("quote") or [{}])[0]
    adjclose = (chart.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose") or []
    bars: list[PriceBar] = []

    for idx, timestamp in enumerate(timestamps):
        open_ = _value_at(quote.get("open"), idx)
        high = _value_at(quote.get("high"), idx)
        low = _value_at(quote.get("low"), idx)
        close = _value_at(quote.get("close"), idx)
        volume = _value_at(quote.get("volume"), idx)
        adjusted_close = _value_at(adjclose, idx)
        if None in {open_, high, low, close, volume, adjusted_close}:
            continue
        bars.append(
            PriceBar(
                ticker=ticker,
                trade_date=datetime.fromtimestamp(timestamp, timezone.utc).date(),
                open=float(open_),
                high=float(high),
                low=float(low),
                close=float(close),
                adjusted_close=float(adjusted_close),
                volume=int(volume),
            )
        )
    return bars


def fetch_sp500_tickers(*, user_agent: str) -> list[str]:
    html = _request(SP500_URL, user_agent=user_agent)
    return parse_sp500_tickers(html)


def parse_sp500_tickers(html: str) -> list[str]:
    parser = _SP500TableParser()
    parser.feed(html)
    return parser.tickers


def to_yahoo_ticker(ticker: str) -> str:
    return ticker.upper().replace(".", "-")


def estimate_daily_price_rows(*, ticker_count: int, years: int) -> int:
    return int(ticker_count * years * 252)


def estimate_daily_price_storage_mb(*, ticker_count: int, years: int) -> tuple[int, int]:
    rows = estimate_daily_price_rows(ticker_count=ticker_count, years=years)
    low_mb = max(1, int(rows * 250 / 1_000_000))
    high_mb = max(low_mb, int(rows * 700 / 1_000_000))
    return low_mb, high_mb


def years_before(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(year=value.year - years, day=28)


def _request(url: str, *, user_agent: str, timeout: int = 30) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "application/json, text/html, */*",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def _unix_seconds(value: date) -> int:
    return int(datetime(value.year, value.month, value.day, tzinfo=timezone.utc).timestamp())


def _value_at(values: list | None, idx: int):
    if not values or idx >= len(values):
        return None
    return values[idx]


class _SP500TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tickers: list[str] = []
        self._in_table = False
        self._in_row = False
        self._cell_index = -1
        self._capture_cell = False
        self._cell_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "table" and attrs_dict.get("id") == "constituents":
            self._in_table = True
        elif self._in_table and tag == "tr":
            self._in_row = True
            self._cell_index = -1
        elif self._in_row and tag in {"td", "th"}:
            self._cell_index += 1
            self._capture_cell = self._cell_index == 0
            self._cell_text = []

    def handle_data(self, data: str) -> None:
        if self._capture_cell:
            self._cell_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "table" and self._in_table:
            self._in_table = False
        elif tag == "tr" and self._in_row:
            self._in_row = False
        elif tag in {"td", "th"} and self._capture_cell:
            value = re.sub(r"\s+", "", "".join(self._cell_text)).upper()
            if value and value != "SYMBOL":
                self.tickers.append(value)
            self._capture_cell = False
