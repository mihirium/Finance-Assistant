from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from finance_rag.models import Document
from finance_rag.text import clean_text


NEWS_FEEDS = {
    "yahoo-finance": "https://finance.yahoo.com/news/rssindex",
    "marketwatch-top": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "cnbc-finance": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
}


def _request(url: str, *, user_agent: str, timeout: int = 20) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "application/json, text/html, application/xml, text/xml;q=0.9, */*;q=0.8",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def fetch_todays_news(
    *,
    as_of: date,
    user_agent: str,
    feed_urls: dict[str, str] | None = None,
) -> list[Document]:
    feeds = feed_urls or NEWS_FEEDS
    documents: list[Document] = []

    for feed_name, url in feeds.items():
        try:
            xml_text = _request(url, user_agent=user_agent)
            root = ET.fromstring(xml_text)
        except Exception as exc:
            print(f"warning: could not fetch {feed_name}: {exc}")
            continue

        for item in root.findall(".//item"):
            title = clean_text(item.findtext("title") or "")
            link = clean_text(item.findtext("link") or "")
            description = clean_text(item.findtext("description") or "")
            pub_date = _parse_rss_date(item.findtext("pubDate"))
            if pub_date and pub_date.date() != as_of:
                continue
            if not title or not link:
                continue
            text = f"{title}\n\n{description}".strip()
            documents.append(
                Document(
                    id=f"news:{feed_name}:{link}",
                    source_type="news",
                    title=title,
                    url=link,
                    published_at=pub_date,
                    text=text,
                )
            )

    return _dedupe_documents(documents)


def fetch_latest_10k(
    *,
    ticker: str,
    user_agent: str,
    cache_dir: Path,
) -> Document | None:
    cik = _lookup_cik(ticker=ticker, user_agent=user_agent, cache_dir=cache_dir)
    if cik is None:
        print(f"warning: no SEC CIK found for {ticker}")
        return None

    submissions_url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
    submissions = json.loads(_request(submissions_url, user_agent=user_agent))
    recent = submissions["filings"]["recent"]
    for idx, form in enumerate(recent.get("form", [])):
        if form != "10-K":
            continue
        accession = recent["accessionNumber"][idx]
        primary_doc = recent["primaryDocument"][idx]
        filing_date = recent["filingDate"][idx]
        accession_path = accession.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_path}/{primary_doc}"
        html_text = _request(url, user_agent=user_agent)
        text = clean_text(html_text)
        return Document(
            id=f"filing:{ticker.upper()}:{accession}",
            source_type="filing",
            title=f"{ticker.upper()} 10-K filed {filing_date}",
            url=url,
            published_at=datetime.fromisoformat(filing_date).replace(tzinfo=timezone.utc),
            text=text,
            ticker=ticker.upper(),
            accession_number=accession,
        )
    return None


def fetch_sec_filings(
    *,
    ticker: str,
    forms: set[str],
    start_date: date,
    end_date: date,
    user_agent: str,
    cache_dir: Path,
) -> list[Document]:
    cik = _lookup_cik(ticker=ticker, user_agent=user_agent, cache_dir=cache_dir)
    if cik is None:
        print(f"warning: no SEC CIK found for {ticker}")
        return []

    submissions_url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
    submissions = json.loads(_request(submissions_url, user_agent=user_agent))
    recent = submissions["filings"]["recent"]
    documents: list[Document] = []

    for filing in _recent_filing_candidates(
        recent,
        forms=forms,
        start_date=start_date,
        end_date=end_date,
    ):
        accession_path = filing["accession_number"].replace("-", "")
        primary_doc = filing["primary_document"]
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_path}/{primary_doc}"
        html_text = _request(url, user_agent=user_agent)
        form = filing["form"]
        filing_date = filing["filing_date"]
        text = clean_text(html_text)
        documents.append(
            Document(
                id=f"filing:{ticker.upper()}:{accession_path}",
                source_type="filing",
                title=f"{ticker.upper()} {form} filed {filing_date.isoformat()}",
                url=url,
                published_at=datetime.combine(filing_date, datetime.min.time(), tzinfo=timezone.utc),
                text=text,
                ticker=ticker.upper(),
                accession_number=filing["accession_number"],
            )
        )

    return documents


def save_documents(documents: list[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for document in documents:
        row = document.__dict__.copy()
        row["published_at"] = document.published_at.isoformat() if document.published_at else None
        rows.append(row)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def load_documents(path: Path) -> list[Document]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    documents = []
    for row in rows:
        published_at = row.get("published_at")
        row["published_at"] = datetime.fromisoformat(published_at) if published_at else None
        documents.append(Document(**row))
    return documents


def _parse_rss_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _lookup_cik(*, ticker: str, user_agent: str, cache_dir: Path) -> int | None:
    cache_path = cache_dir / "sec_company_tickers.json"
    if cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        data = json.loads(_request("https://www.sec.gov/files/company_tickers.json", user_agent=user_agent))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data), encoding="utf-8")

    ticker = ticker.upper()
    for row in data.values():
        if row.get("ticker", "").upper() == ticker:
            return int(row["cik_str"])
    return None


def _recent_filing_candidates(
    recent: dict,
    *,
    forms: set[str],
    start_date: date,
    end_date: date,
) -> list[dict[str, object]]:
    candidates = []
    wanted_forms = {form.upper() for form in forms}
    for idx, form in enumerate(recent.get("form", [])):
        normalized_form = form.upper()
        if normalized_form not in wanted_forms:
            continue
        filing_date = date.fromisoformat(recent["filingDate"][idx])
        if filing_date < start_date or filing_date > end_date:
            continue
        primary_document = recent["primaryDocument"][idx]
        accession_number = recent["accessionNumber"][idx]
        if not primary_document or not accession_number:
            continue
        candidates.append(
            {
                "form": normalized_form,
                "filing_date": filing_date,
                "primary_document": primary_document,
                "accession_number": accession_number,
            }
        )
    return candidates


def _dedupe_documents(documents: list[Document]) -> list[Document]:
    seen: set[str] = set()
    unique = []
    for document in documents:
        key = _canonical_url(document.url) or re.sub(r"\W+", " ", document.title.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        unique.append(document)
    return unique


def _canonical_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def parse_tickers(value: str) -> list[str]:
    return [part.strip().upper() for part in value.split(",") if part.strip()]


def parse_forms(value: str) -> set[str]:
    return {part.strip().upper() for part in value.split(",") if part.strip()}
