from __future__ import annotations

import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo

from finance_rag.models import Document
from finance_rag.text import clean_text


NEWS_FEEDS = {
    "yahoo-finance": "https://finance.yahoo.com/news/rssindex",
    "marketwatch-top": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "marketwatch-realtime": "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
    "cnbc-top": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "cnbc-business": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10001147",
    "cnbc-finance": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    "cnbc-earnings": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839135",
    "cnbc-economy": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
    "cnbc-investing": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069",
    "cnbc-market-insider": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20409666",
    "nyt-business": "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
    "bbc-business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "npr-business": "https://feeds.npr.org/1006/rss.xml",
    "guardian-business": "https://www.theguardian.com/us/business/rss",
}
NEW_YORK_TZ = ZoneInfo("America/New_York")
MIN_ARTICLE_WORDS = 120
MAX_ARTICLE_CHARS = 60_000
ARTICLE_FETCH_WORKERS = 6


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
    fetch_full_text: bool = True,
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
            if pub_date is None or pub_date.astimezone(NEW_YORK_TZ).date() != as_of:
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

    documents = _dedupe_documents(documents)
    if not fetch_full_text or not documents:
        return documents
    return _enrich_news_documents(documents, user_agent=user_agent)


def _enrich_news_documents(documents: list[Document], *, user_agent: str) -> list[Document]:
    enriched_by_id: dict[str, Document] = {}
    with ThreadPoolExecutor(max_workers=ARTICLE_FETCH_WORKERS) as executor:
        futures = {
            executor.submit(_enrich_news_document, document, user_agent=user_agent): document
            for document in documents
        }
        for future in as_completed(futures):
            original = futures[future]
            try:
                enriched_by_id[original.id] = future.result()
            except Exception:
                enriched_by_id[original.id] = original

    enriched = [enriched_by_id[document.id] for document in documents]
    full_text_count = sum(1 for original, result in zip(documents, enriched) if result.text != original.text)
    print(f"Fetched full article text for {full_text_count}/{len(documents)} news items.")
    return enriched


def _enrich_news_document(document: Document, *, user_agent: str) -> Document:
    html_text = _request(document.url, user_agent=user_agent, timeout=25)
    article_text = _extract_article_text(html_text, url=document.url)
    if article_text is None:
        return document
    return Document(
        id=document.id,
        source_type=document.source_type,
        title=document.title,
        url=document.url,
        published_at=document.published_at,
        text=article_text[:MAX_ARTICLE_CHARS],
    )


def _extract_article_text(html_text: str, *, url: str) -> str | None:
    try:
        from trafilatura import extract
    except ImportError as exc:
        raise RuntimeError("Install article extraction support with `pip install -e .`.") from exc

    extracted = extract(
        html_text,
        url=url,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )
    if not extracted:
        return None
    text = clean_text(extracted)
    if len(text.split()) < MIN_ARTICLE_WORDS:
        return None
    return text


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
