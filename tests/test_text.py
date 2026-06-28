from datetime import datetime, timezone
import unittest

from finance_rag.models import Document
from finance_rag.text import chunk_documents


class ChunkingTests(unittest.TestCase):
    def test_full_article_creates_overlapping_chunks(self) -> None:
        document = Document(
            id="news:1",
            source_type="news",
            title="Market update",
            url="https://example.com/news",
            published_at=datetime.now(timezone.utc),
            text=" ".join(f"word{index}" for index in range(500)),
        )

        chunks = chunk_documents([document])

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].document_id, document.id)
        self.assertEqual(chunks[0].text.split()[-45:], chunks[1].text.split()[:45])


if __name__ == "__main__":
    unittest.main()
