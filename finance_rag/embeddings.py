from __future__ import annotations

import json
import os
import urllib.request
from urllib.error import HTTPError


DEFAULT_EMBEDDING_DIMENSIONS = 768
DEFAULT_HF_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_HF_MAX_CHARS = 4000


class HuggingFaceEmbedder:
    """Client for the matching `/embed` endpoint in `hf-space`."""

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        token: str | None = None,
        model: str | None = None,
        dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        max_chars: int | None = None,
    ) -> None:
        self.endpoint = (endpoint or os.getenv("HF_EMBEDDING_URL") or "").rstrip("/")
        self.token = token or os.getenv("HF_TOKEN")
        self.model = model or os.getenv("HF_EMBEDDING_MODEL", DEFAULT_HF_EMBEDDING_MODEL)
        self.dimensions = dimensions
        self.max_chars = int(os.getenv("HF_EMBEDDING_MAX_CHARS", max_chars or DEFAULT_HF_MAX_CHARS))
        if not self.endpoint:
            raise ValueError("Set HF_EMBEDDING_URL to the Hugging Face Space /embed endpoint.")

    def embed_document(self, *, title: str, text: str) -> list[float]:
        payload = {
            "model": self.model,
            "texts": [f"title: {title or 'none'} | text: {text}"[: self.max_chars]],
        }
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Hugging Face embedding request failed: HTTP {exc.code}: {detail}") from exc

        values = _extract_embedding(data)
        if len(values) != self.dimensions:
            raise ValueError(
                f"Hugging Face returned {len(values)} dimensions; expected {self.dimensions}."
            )
        return values


def _extract_embedding(data: dict) -> list[float]:
    values = data.get("embedding")
    if values is None and data.get("embeddings"):
        values = data["embeddings"][0]
    if not values:
        raise ValueError("Hugging Face embedding response did not contain values")
    return [float(value) for value in values]
