import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from finance_rag.models import Document
from finance_rag.sources import (
    NEW_YORK_TZ,
    _canonical_url,
    _enrich_news_document,
    _extract_article_text,
    _parse_rss_date,
)


class SourceTests(unittest.TestCase):
    def test_parse_rss_date_normalizes_to_utc(self) -> None:
        parsed = _parse_rss_date("Fri, 19 Jun 2026 09:30:00 -0400")

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.isoformat(), "2026-06-19T13:30:00+00:00")

    def test_rss_date_maps_to_new_york_calendar_day(self) -> None:
        parsed = _parse_rss_date("Sun, 21 Jun 2026 01:30:00 +0000")

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.astimezone(NEW_YORK_TZ).isoformat(), "2026-06-20T21:30:00-04:00")

    def test_canonical_url_removes_query_and_fragment(self) -> None:
        self.assertEqual(
            _canonical_url("https://example.com/story?utm_source=rss#section"),
            "https://example.com/story",
        )

    def test_extract_article_text_keeps_substantial_main_content(self) -> None:
        article = " ".join(f"market analysis detail{i}" for i in range(150))
        html = f"<html><body><nav>Navigation</nav><article><p>{article}</p></article></body></html>"

        extracted = _extract_article_text(html, url="https://example.com/story")

        self.assertIsNotNone(extracted)
        self.assertIn("detail149", extracted)
        self.assertNotIn("Navigation", extracted)

    def test_extract_article_text_rejects_thin_pages(self) -> None:
        html = "<html><body><article><p>Subscribe to continue reading.</p></article></body></html>"

        self.assertIsNone(_extract_article_text(html, url="https://example.com/paywall"))

    def test_enrich_document_replaces_summary_with_article_text(self) -> None:
        document = Document(
            id="news:test:1",
            source_type="news",
            title="Market update",
            url="https://example.com/story",
            published_at=datetime(2026, 6, 22, tzinfo=timezone.utc),
            text="Short RSS summary.",
        )
        full_text = "full article text " * 150

        with (
            patch("finance_rag.sources._request", return_value="<html></html>"),
            patch("finance_rag.sources._extract_article_text", return_value=full_text),
        ):
            enriched = _enrich_news_document(document, user_agent="test")

        self.assertEqual(enriched.text, full_text)
        self.assertEqual(enriched.id, document.id)


if __name__ == "__main__":
    unittest.main()
