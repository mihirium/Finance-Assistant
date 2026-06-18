from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from finance_rag.models import Chunk, Document, SearchResult
from finance_rag.text import chunk_text, tokenize


class LocalVectorIndex:
    """Small BM25-style local retrieval index.

    This gives us a persistent RAG retrieval layer without requiring a vector
    database. It is intentionally simple and swappable for embeddings later.
    """

    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self.chunks = chunks or []
        self._doc_freq: Counter[str] = Counter()
        self._term_freqs: list[Counter[str]] = []
        self._avg_len = 0.0
        self._build()

    @classmethod
    def from_documents(cls, documents: list[Document]) -> "LocalVectorIndex":
        chunks: list[Chunk] = []
        for document in documents:
            for idx, text in enumerate(chunk_text(document.text)):
                chunks.append(
                    Chunk(
                        id=f"{document.id}:{idx}",
                        document_id=document.id,
                        source_type=document.source_type,
                        title=document.title,
                        url=document.url,
                        text=text,
                        ticker=document.ticker,
                        published_at=document.published_at,
                    )
                )
        return cls(chunks)

    def _build(self) -> None:
        self._doc_freq.clear()
        self._term_freqs = []
        total_len = 0
        for chunk in self.chunks:
            counts = Counter(tokenize(f"{chunk.title} {chunk.text}"))
            self._term_freqs.append(counts)
            self._doc_freq.update(counts.keys())
            total_len += sum(counts.values())
        self._avg_len = total_len / len(self.chunks) if self.chunks else 0.0

    def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        source_type: str | None = None,
    ) -> list[SearchResult]:
        query_terms = tokenize(query)
        if not query_terms or not self.chunks:
            return []

        n = len(self.chunks)
        k1 = 1.5
        b = 0.75
        scores: defaultdict[int, float] = defaultdict(float)

        for term in query_terms:
            doc_freq = self._doc_freq.get(term, 0)
            if doc_freq == 0:
                continue
            idf = math.log(1 + (n - doc_freq + 0.5) / (doc_freq + 0.5))
            for idx, counts in enumerate(self._term_freqs):
                chunk = self.chunks[idx]
                if source_type and chunk.source_type != source_type:
                    continue
                freq = counts.get(term, 0)
                if freq == 0:
                    continue
                doc_len = sum(counts.values()) or 1
                denom = freq + k1 * (1 - b + b * doc_len / (self._avg_len or 1))
                scores[idx] += idf * (freq * (k1 + 1)) / denom

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [SearchResult(self.chunks[idx], score) for idx, score in ranked[:top_k]]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = []
        for chunk in self.chunks:
            row = asdict(chunk)
            row["published_at"] = chunk.published_at.isoformat() if chunk.published_at else None
            payload.append(row)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LocalVectorIndex":
        rows = json.loads(path.read_text(encoding="utf-8"))
        chunks = []
        for row in rows:
            published_at = row.get("published_at")
            row["published_at"] = datetime.fromisoformat(published_at) if published_at else None
            chunks.append(Chunk(**row))
        return cls(chunks)
