from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


SourceType = Literal["news"]


@dataclass(frozen=True)
class Document:
    id: str
    source_type: SourceType
    title: str
    url: str
    published_at: datetime | None
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    source_type: SourceType
    title: str
    url: str
    text: str
    published_at: datetime | None
