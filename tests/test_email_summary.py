from datetime import date
import json
import unittest
from unittest.mock import Mock, patch

from finance_rag.email_summary import (
    NewsItem,
    PriceMove,
    _normalize_three_paragraphs,
    build_market_email,
    send_resend_email,
)


class EmailSummaryTests(unittest.TestCase):
    def test_builds_fallback_market_email_with_three_paragraphs(self) -> None:
        email = build_market_email(
            summary_date=date(2026, 6, 22),
            news_items=[NewsItem(title="Stocks rise into the close", url="https://example.com", text="Markets rose.")],
            gainers=[PriceMove(ticker="AAPL", price=200.0, percent_change=2.5)],
            losers=[PriceMove(ticker="MSFT", price=400.0, percent_change=-1.25)],
            app_url="https://finance.example.com",
        )

        self.assertEqual(email.subject, "Market summary for 2026-06-22")
        self.assertEqual(len(email.text.split("\n\n")), 3)
        self.assertIn("AAPL +2.50%", email.text)
        self.assertIn("Ask Finance AI", email.html)

    def test_normalize_three_paragraphs_trims_extra_model_output(self) -> None:
        text = _normalize_three_paragraphs("One.\n\nTwo.\n\nThree.\n\nFour.")

        self.assertEqual(text, "One.\n\nTwo.\n\nThree.")

    def test_send_resend_email_returns_provider_id(self) -> None:
        response = Mock()
        response.read.return_value = json.dumps({"id": "email_123"}).encode("utf-8")
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=None)

        with patch("urllib.request.urlopen", return_value=response) as urlopen:
            provider_id = send_resend_email(
                api_key="secret",
                sender="Finance AI <summary@example.com>",
                recipient="user@example.com",
                subject="Daily market summary",
                text="Hello",
                html="<p>Hello</p>",
                reply_to="reply@example.com",
            )

        self.assertEqual(provider_id, "email_123")
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["to"], ["user@example.com"])
        self.assertEqual(payload["reply_to"], "reply@example.com")
        self.assertEqual(request.headers["Authorization"], "Bearer secret")


if __name__ == "__main__":
    unittest.main()
