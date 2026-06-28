from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from datetime import date
from html import escape
from typing import Any


RESEND_EMAIL_URL = "https://api.resend.com/emails"


@dataclass(frozen=True)
class NewsItem:
    title: str
    url: str
    text: str


@dataclass(frozen=True)
class PriceMove:
    ticker: str
    price: float
    percent_change: float


@dataclass(frozen=True)
class MarketEmail:
    subject: str
    text: str
    html: str


def build_market_email(
    *,
    summary_date: date,
    news_items: list[NewsItem],
    gainers: list[PriceMove],
    losers: list[PriceMove],
    generation_url: str | None = None,
    app_url: str | None = None,
) -> MarketEmail:
    subject = f"Market summary for {summary_date.isoformat()}"
    text = _generate_summary(
        summary_date=summary_date,
        news_items=news_items,
        gainers=gainers,
        losers=losers,
        generation_url=generation_url,
    )
    html = _to_html(text=text, app_url=app_url)
    return MarketEmail(subject=subject, text=text, html=html)


def send_resend_email(
    *,
    api_key: str,
    sender: str,
    recipient: str,
    subject: str,
    text: str,
    html: str,
    reply_to: str | None = None,
) -> str | None:
    payload: dict[str, Any] = {
        "from": sender,
        "to": [recipient],
        "subject": subject,
        "text": text,
        "html": html,
    }
    if reply_to:
        payload["reply_to"] = reply_to

    request = urllib.request.Request(
        RESEND_EMAIL_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    provider_id = data.get("id")
    return str(provider_id) if provider_id else None


def _generate_summary(
    *,
    summary_date: date,
    news_items: list[NewsItem],
    gainers: list[PriceMove],
    losers: list[PriceMove],
    generation_url: str | None,
) -> str:
    if generation_url:
        answer = _ask_generation_service(
            generation_url=generation_url,
            summary_date=summary_date,
            news_items=news_items,
            gainers=gainers,
            losers=losers,
        )
        if answer:
            return _normalize_three_paragraphs(answer)
    return _fallback_summary(summary_date=summary_date, news_items=news_items, gainers=gainers, losers=losers)


def _ask_generation_service(
    *,
    generation_url: str,
    summary_date: date,
    news_items: list[NewsItem],
    gainers: list[PriceMove],
    losers: list[PriceMove],
) -> str:
    contexts = _summary_contexts(news_items=news_items, gainers=gainers, losers=losers)
    request = urllib.request.Request(
        generation_url,
        data=json.dumps(
            {
                "question": (
                    f"Write exactly three concise paragraphs summarizing the U.S. market on "
                    f"{summary_date.isoformat()}. Use the news and price-move context. "
                    "Do not use bullet points."
                ),
                "contexts": contexts,
            }
        ).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.getenv("HF_TOKEN") else {}),
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data.get("answer", "")).strip()


def _summary_contexts(
    *,
    news_items: list[NewsItem],
    gainers: list[PriceMove],
    losers: list[PriceMove],
) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    market_moves = _format_price_moves(gainers[:5], losers[:5])
    if market_moves:
        contexts.append(
            {
                "id": 1,
                "title": "S&P 500 price snapshots",
                "sourceType": "prices",
                "url": "",
                "text": market_moves,
            }
        )
    for index, item in enumerate(news_items[:12], start=2):
        contexts.append(
            {
                "id": index,
                "title": item.title,
                "sourceType": "news",
                "url": item.url,
                "text": item.text[:1200],
            }
        )
    return contexts


def _fallback_summary(
    *,
    summary_date: date,
    news_items: list[NewsItem],
    gainers: list[PriceMove],
    losers: list[PriceMove],
) -> str:
    leading_titles = "; ".join(item.title for item in news_items[:4]) or "the stored market news"
    paragraph_one = (
        f"Markets closed {summary_date.isoformat()} with investors focused on {leading_titles}."
    )

    move_text = _format_price_moves(gainers[:3], losers[:3])
    paragraph_two = (
        f"Within the S&P 500 snapshots, the biggest moves were: {move_text}."
        if move_text
        else "The stored price snapshots did not include enough close data to rank daily movers yet."
    )

    paragraph_three = (
        "For follow-up questions, ask Finance AI about the day by company, sector, theme, or source. "
        "The assistant will answer from the news and price context stored for the session."
    )
    return "\n\n".join([paragraph_one, paragraph_two, paragraph_three])


def _format_price_moves(gainers: list[PriceMove], losers: list[PriceMove]) -> str:
    parts: list[str] = []
    if gainers:
        parts.append(
            "top gainers "
            + ", ".join(f"{move.ticker} {move.percent_change:+.2f}%" for move in gainers)
        )
    if losers:
        parts.append(
            "top decliners "
            + ", ".join(f"{move.ticker} {move.percent_change:+.2f}%" for move in losers)
        )
    return "; ".join(parts)


def _normalize_three_paragraphs(text: str) -> str:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    if len(paragraphs) >= 3:
        return "\n\n".join(paragraphs[:3])
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(paragraphs).strip())
    if len(sentences) >= 3:
        return "\n\n".join(sentences[:3])
    return "\n\n".join(paragraphs)


def _to_html(*, text: str, app_url: str | None) -> str:
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    body = "\n".join(f"<p>{escape(paragraph)}</p>" for paragraph in paragraphs)
    if app_url:
        body += f'\n<p><a href="{escape(app_url, quote=True)}">Ask Finance AI a follow-up question</a></p>'
    return f"<!doctype html><html><body>{body}</body></html>"
