import unittest

from finance_rag.embeddings import _extract_embedding


class EmbeddingTests(unittest.TestCase):
    def test_extracts_first_embedding(self) -> None:
        self.assertEqual(_extract_embedding({"embeddings": [[1, "2.5"]]}), [1.0, 2.5])

    def test_rejects_missing_embedding(self) -> None:
        with self.assertRaises(ValueError):
            _extract_embedding({})


if __name__ == "__main__":
    unittest.main()
