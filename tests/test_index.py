from datetime import datetime, timezone
import unittest

from finance_rag.index import LocalVectorIndex
from finance_rag.models import Document


class LocalVectorIndexTests(unittest.TestCase):
    def test_index_retrieves_relevant_news_before_filing_noise(self) -> None:
        documents = [
            Document(
                id="news:1",
                source_type="news",
                title="Banks rally after Fed decision",
                url="https://example.com/news",
                published_at=datetime.now(timezone.utc),
                text="Regional banks rallied today after the Federal Reserve signaled fewer credit losses.",
            ),
            Document(
                id="filing:1",
                source_type="filing",
                title="ACME 10-K",
                url="https://example.com/10k",
                published_at=datetime.now(timezone.utc),
                text="The company manufactures industrial parts and faces commodity supply risks.",
                ticker="ACME",
            ),
        ]

        index = LocalVectorIndex.from_documents(documents)
        results = index.search("why did banks rally today")

        self.assertTrue(results)
        self.assertEqual(results[0].chunk.document_id, "news:1")

    def test_source_filter_limits_results(self) -> None:
        documents = [
            Document("news:1", "news", "Market update", "https://example.com/news", None, "rates and stocks"),
            Document("filing:1", "filing", "MSFT 10-K", "https://example.com/10k", None, "rates and risk", "MSFT"),
        ]

        index = LocalVectorIndex.from_documents(documents)
        results = index.search("rates", source_type="filing")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.source_type, "filing")


if __name__ == "__main__":
    unittest.main()
