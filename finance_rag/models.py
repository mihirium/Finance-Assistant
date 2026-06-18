from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


SourceType = Literal["news", "filing"]


@dataclass(frozen=True)
class Document:
    id: str
    source_type: SourceType
    title: str
    url: str
    published_at: datetime | None
    text: str
    ticker: str | None = None
    accession_number: str | None = None


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    source_type: SourceType
    title: str
    url: str
    text: str
    ticker: str | None
    published_at: datetime | None


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
