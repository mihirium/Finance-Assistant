import unittest

from finance_rag.pgvector_store import _validate_dimensions, _vector_literal


class PgVectorStoreTests(unittest.TestCase):
    def test_vector_literal_formats_pgvector_input(self) -> None:
        self.assertEqual(_vector_literal([1.0, -2.25, 0.333333333]), "[1.00000000,-2.25000000,0.33333333]")

    def test_validate_dimensions_rejects_bad_values(self) -> None:
        with self.assertRaises(ValueError):
            _validate_dimensions(0)


if __name__ == "__main__":
    unittest.main()
