import unittest

from finance_rag.embeddings import HashingEmbedder, _extract_ollama_embedding
from finance_rag.llm import _direct_context_warning, _ensure_citations
from finance_rag.models import Chunk, SearchResult


class LocalModelTests(unittest.TestCase):
    def test_extracts_ollama_embedding_values(self) -> None:
        self.assertEqual(_extract_ollama_embedding({"embedding": [1, "2.5"]}), [1.0, 2.5])

    def test_raises_when_ollama_embedding_has_no_values(self) -> None:
        with self.assertRaises(ValueError):
            _extract_ollama_embedding({})

    def test_hashing_embedder_returns_stable_dimensions(self) -> None:
        embedder = HashingEmbedder(dimensions=8)

        first = embedder.embed_query("revenue risk")
        second = embedder.embed_query("revenue risk")

        self.assertEqual(len(first), 8)
        self.assertEqual(first, second)

    def test_warns_when_question_entity_is_missing_from_results(self) -> None:
        results = [
            SearchResult(
                Chunk(
                    id="1",
                    document_id="news:1",
                    source_type="news",
                    title="Megacap stocks face valuation pressure",
                    url="https://example.com",
                    text="Apple and Microsoft are discussed in relation to market valuations.",
                    ticker=None,
                    published_at=None,
                ),
                3.2,
            )
        ]

        warning = _direct_context_warning("why is SpaceX so highly valued?", results)

        self.assertIsNotNone(warning)
        self.assertIn("SpaceX", warning or "")

    def test_no_warning_when_question_entity_is_present(self) -> None:
        results = [
            SearchResult(
                Chunk(
                    id="1",
                    document_id="news:1",
                    source_type="news",
                    title="SpaceX tender offer values company",
                    url="https://example.com",
                    text="SpaceX is valued by investors based on launch cadence and Starlink growth.",
                    ticker=None,
                    published_at=None,
                ),
                4.0,
            )
        ]

        self.assertIsNone(_direct_context_warning("why is SpaceX so highly valued?", results))

    def test_ensure_citations_appends_sources_when_missing(self) -> None:
        results = [
            SearchResult(
                Chunk("1", "doc1", "filing", "AAPL 10-K", "https://example.com/1", "supply risk", "AAPL", None),
                0.9,
            ),
            SearchResult(
                Chunk("2", "doc2", "filing", "AAPL 10-Q", "https://example.com/2", "manufacturing risk", "AAPL", None),
                0.8,
            ),
        ]

        answer = _ensure_citations("Apple disclosed supplier concentration risk.", results)

        self.assertIn("Sources: [1], [2]", answer)


if __name__ == "__main__":
    unittest.main()
