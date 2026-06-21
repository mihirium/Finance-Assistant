from datetime import datetime, timezone
import unittest

from finance_rag.index import LocalVectorIndex
from finance_rag.models import Document


class LocalVectorIndexTests(unittest.TestCase):
    def test_index_retrieves_relevant_news(self) -> None:
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
                id="news:2",
                source_type="news",
                title="Oil prices rise",
                url="https://example.com/oil",
                published_at=datetime.now(timezone.utc),
                text="Oil prices rose as traders assessed global supply.",
            ),
        ]

        index = LocalVectorIndex.from_documents(documents)
        results = index.search("why did banks rally today")

        self.assertTrue(results)
        self.assertEqual(results[0].chunk.document_id, "news:1")

if __name__ == "__main__":
    unittest.main()
