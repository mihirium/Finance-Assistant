import unittest
from unittest.mock import Mock, patch

from finance_rag.pgvector_store import _connect, _validate_dimensions, _vector_literal


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


if __name__ == "__main__":
    unittest.main()
