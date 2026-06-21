from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from zoneinfo import ZoneInfo


SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
NEW_YORK_TZ = ZoneInfo("America/New_York")
SNAPSHOT_TYPES = ("open", "midday", "close")


@dataclass(frozen=True)
class PriceSnapshot:
    ticker: str
    market_date: date
    snapshot_type: str
    captured_at: datetime
    price: float
    day_open: float
    day_high: float
    day_low: float
    previous_close: float | None
    volume: int
    source: str = "yahoo"


def fetch_sp500_tickers(*, user_agent: str) -> list[str]:
    html = _request_text(SP500_URL, user_agent=user_agent)
    return parse_sp500_tickers(html)


def fetch_price_snapshots(
    tickers: list[str],
    *,
    snapshot_type: str,
    market_date: date,
    user_agent: str,
    max_workers: int = 12,
) -> tuple[list[PriceSnapshot], list[str]]:
    if snapshot_type not in SNAPSHOT_TYPES:
        raise ValueError(f"snapshot_type must be one of: {', '.join(SNAPSHOT_TYPES)}")

    snapshots: list[PriceSnapshot] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                fetch_yahoo_snapshot,
                ticker,
                snapshot_type=snapshot_type,
                market_date=market_date,
                user_agent=user_agent,
            ): ticker
            for ticker in tickers
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                snapshot = future.result()
            except Exception as exc:
                failures.append(f"{ticker}: {exc}")
                continue
            if snapshot is None:
                failures.append(f"{ticker}: no quote for {market_date.isoformat()}")
            else:
                snapshots.append(snapshot)

    snapshots.sort(key=lambda snapshot: snapshot.ticker)
    failures.sort()
    return snapshots, failures


def fetch_yahoo_snapshot(
    ticker: str,
    *,
    snapshot_type: str,
    market_date: date,
    user_agent: str,
) -> PriceSnapshot | None:
    yahoo_ticker = to_yahoo_ticker(ticker)
    period_start = datetime.combine(market_date, datetime_time.min, tzinfo=NEW_YORK_TZ)
    period_end = period_start + timedelta(days=1)
    params = urllib.parse.urlencode(
        {
            "period1": int(period_start.timestamp()),
            "period2": int(period_end.timestamp()),
            "interval": "1m",
            "includePrePost": "false",
            "events": "div,splits",
        }
    )
    url = f"{YAHOO_CHART_URL}/{urllib.parse.quote(yahoo_ticker)}?{params}"
    data = json.loads(_request_text(url, user_agent=user_agent, retries=3))
    return parse_yahoo_snapshot(
        data,
        ticker=ticker.upper(),
        snapshot_type=snapshot_type,
        expected_date=market_date,
    )


def parse_yahoo_snapshot(
    data: dict,
    *,
    ticker: str,
    snapshot_type: str,
    expected_date: date,
) -> PriceSnapshot | None:
    results = data.get("chart", {}).get("result") or []
    if not results:
        error = data.get("chart", {}).get("error")
        if error:
            raise ValueError(str(error))
        return None

    chart = results[0]
    timestamps = chart.get("timestamp") or []
    quote = (chart.get("indicators", {}).get("quote") or [{}])[0]
    closes = quote.get("close") or []
    valid_indexes = [
        idx
        for idx, timestamp in enumerate(timestamps)
        if _value_at(closes, idx) is not None
        and datetime.fromtimestamp(timestamp, timezone.utc).astimezone(NEW_YORK_TZ).date() == expected_date
    ]
    if not valid_indexes:
        return None

    latest_idx = valid_indexes[-1]
    captured_at = datetime.fromtimestamp(timestamps[latest_idx], timezone.utc)
    opens = [_value_at(quote.get("open"), idx) for idx in valid_indexes]
    highs = [_value_at(quote.get("high"), idx) for idx in valid_indexes]
    lows = [_value_at(quote.get("low"), idx) for idx in valid_indexes]
    volumes = [_value_at(quote.get("volume"), idx) for idx in valid_indexes]

    valid_opens = [float(value) for value in opens if value is not None]
    valid_highs = [float(value) for value in highs if value is not None]
    valid_lows = [float(value) for value in lows if value is not None]
    if not valid_opens or not valid_highs or not valid_lows:
        return None

    previous_close = chart.get("meta", {}).get("chartPreviousClose")
    return PriceSnapshot(
        ticker=ticker,
        market_date=expected_date,
        snapshot_type=snapshot_type,
        captured_at=captured_at,
        price=float(closes[latest_idx]),
        day_open=valid_opens[0],
        day_high=max(valid_highs),
        day_low=min(valid_lows),
        previous_close=float(previous_close) if previous_close is not None else None,
        volume=sum(int(value) for value in volumes if value is not None),
    )


def parse_sp500_tickers(html: str) -> list[str]:
    parser = _SP500TableParser()
    parser.feed(html)
    return parser.tickers


def parse_tickers(value: str) -> list[str]:
    return [part.strip().upper() for part in value.split(",") if part.strip()]


def to_yahoo_ticker(ticker: str) -> str:
    return ticker.upper().replace(".", "-")


def _request_text(url: str, *, user_agent: str, retries: int = 1, timeout: int = 30) -> str:
    for attempt in range(retries):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": user_agent, "Accept": "application/json, text/html, */*"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(charset, errors="replace")
        except HTTPError as exc:
            if exc.code != 429 or attempt == retries - 1:
                raise
        except URLError:
            if attempt == retries - 1:
                raise
        time.sleep(2**attempt)
    raise RuntimeError(f"Could not fetch {url}")


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
