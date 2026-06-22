import unittest
from datetime import date
from unittest.mock import Mock, patch

from finance_rag.pgvector_store import (
    _connect,
    _news_retention_cutoff,
    _price_retention_cutoff,
    _validate_dimensions,
    _vector_literal,
)


class PgVectorStoreTests(unittest.TestCase):
    def test_vector_literal_formats_pgvector_input(self) -> None:
        self.assertEqual(_vector_literal([1.0, -2.25, 0.333333333]), "[1.00000000,-2.25000000,0.33333333]")

    def test_validate_dimensions_rejects_bad_values(self) -> None:
        with self.assertRaises(ValueError):
            _validate_dimensions(0)

    def test_connect_disables_automatic_prepared_statements(self) -> None:
        psycopg = Mock()
        with patch("finance_rag.pgvector_store._load_psycopg", return_value=psycopg):
            _connect("postgresql://example")

        psycopg.connect.assert_called_once_with(
            "postgresql://example",
            prepare_threshold=None,
        )

    def test_news_retention_cutoff_keeps_thirty_calendar_dates(self) -> None:
        cutoff = _news_retention_cutoff(
            as_of=date(2026, 6, 21),
            retention_days=30,
        )

        self.assertEqual(cutoff.isoformat(), "2026-05-23T00:00:00-04:00")

    def test_news_retention_rejects_nonpositive_window(self) -> None:
        with self.assertRaises(ValueError):
            _news_retention_cutoff(as_of=date(2026, 6, 21), retention_days=0)

    def test_price_retention_cutoff_keeps_thirty_calendar_dates(self) -> None:
        cutoff = _price_retention_cutoff(
            market_date=date(2026, 6, 22),
            retention_days=30,
        )

        self.assertEqual(cutoff, date(2026, 5, 24))

    def test_price_retention_rejects_nonpositive_window(self) -> None:
        with self.assertRaises(ValueError):
            _price_retention_cutoff(market_date=date(2026, 6, 22), retention_days=0)


if __name__ == "__main__":
    unittest.main()
