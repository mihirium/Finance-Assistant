import unittest

from finance_rag.api import _source_from_result
from finance_rag.models import Chunk, SearchResult


class ApiTests(unittest.TestCase):
    def test_source_from_result_uses_web_shape(self) -> None:
        result = SearchResult(
            Chunk(
                id="chunk-1",
                document_id="doc-1",
                source_type="filing",
                title="AAPL 10-K",
                url="https://example.com/aapl",
                text="supply chain risk " * 40,
                ticker="AAPL",
                published_at=None,
            ),
            score=0.75,
        )

        source = _source_from_result(result)

        self.assertEqual(source.sourceType, "filing")
        self.assertEqual(source.ticker, "AAPL")
        self.assertLessEqual(len(source.excerpt), 360)


if __name__ == "__main__":
    unittest.main()
