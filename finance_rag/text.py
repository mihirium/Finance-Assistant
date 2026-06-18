from __future__ import annotations

import html
import re
from collections.abc import Iterable


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9$.-]{1,}")
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def clean_text(value: str) -> str:
    value = html.unescape(value)
    value = TAG_RE.sub(" ", value)
    value = SPACE_RE.sub(" ", value)
    return value.strip()


def tokenize(value: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(value)]


def chunk_text(
    text: str,
    *,
    max_words: int = 260,
    overlap_words: int = 45,
) -> Iterable[str]:
    words = text.split()
    if not words:
        return

    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        yield " ".join(words[start:end])
        if end == len(words):
            break
        start = max(end - overlap_words, start + 1)
