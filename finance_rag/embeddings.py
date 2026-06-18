from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.request
from collections import Counter
from urllib.error import HTTPError


DEFAULT_EMBEDDING_DIMENSIONS = 768
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_MAX_CHARS = 4000
DEFAULT_HF_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_HF_MAX_CHARS = 4000
DEFAULT_SENTENCE_TRANSFORMERS_MODEL = DEFAULT_HF_EMBEDDING_MODEL
DEFAULT_SENTENCE_TRANSFORMERS_MAX_CHARS = 4000
TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9$.-]{1,}")


class HashingEmbedder:
    """Local quota-free embedding fallback for smoke tests.

    This is not as semantically rich as an embedding model, but it gives
    pgvector stable dense vectors when Ollama is unavailable.
    """

    def __init__(self, *, dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS) -> None:
        self.dimensions = dimensions

    def embed_document(self, *, title: str, text: str) -> list[float]:
        return self._embed(f"{title} {text}")

    def embed_query(self, query: str) -> list[float]:
        return self._embed(query)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        counts = Counter(token.lower() for token in TOKEN_RE.findall(text))
        for token, count in counts.items():
            bucket = _stable_hash(token) % self.dimensions
            sign = -1.0 if _stable_hash(f"{token}:sign") % 2 else 1.0
            vector[bucket] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class OllamaEmbedder:
    def __init__(
        self,
        *,
        host: str | None = None,
        model: str | None = None,
        dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        max_chars: int | None = None,
    ) -> None:
        self.host = (host or os.getenv("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
        self.model = model or os.getenv("OLLAMA_EMBEDDING_MODEL", DEFAULT_OLLAMA_EMBEDDING_MODEL)
        self.dimensions = dimensions
        self.max_chars = int(os.getenv("OLLAMA_EMBEDDING_MAX_CHARS", max_chars or DEFAULT_OLLAMA_MAX_CHARS))

    def embed_document(self, *, title: str, text: str) -> list[float]:
        return self._embed(f"title: {title or 'none'} | text: {text}")

    def embed_query(self, query: str) -> list[float]:
        return self._embed(query)

    def _embed(self, text: str) -> list[float]:
        text = text[: self.max_chars]
        payload = {
            "model": self.model,
            "prompt": text,
        }
        request = urllib.request.Request(
            f"{self.host}/api/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama embedding request failed: HTTP {exc.code}: {detail}") from exc
        values = _extract_ollama_embedding(data)
        if len(values) != self.dimensions:
            raise ValueError(
                f"Ollama model {self.model} returned {len(values)} dimensions, "
                f"but the pgvector schema expects {self.dimensions}."
            )
        return values


class HuggingFaceEmbedder:
    """Remote Hugging Face embedding service client.

    The service should expose POST /embed and return 768-dimensional vectors.
    The matching demo service lives in hf-space/.
    """

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
            raise ValueError("Set HF_EMBEDDING_URL to your Hugging Face Space /embed endpoint.")

    def embed_document(self, *, title: str, text: str) -> list[float]:
        return self._embed(f"title: {title or 'none'} | text: {text}")

    def embed_query(self, query: str) -> list[float]:
        return self._embed(query)

    def _embed(self, text: str) -> list[float]:
        payload = {
            "model": self.model,
            "texts": [text[: self.max_chars]],
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
        values = _extract_hf_embedding(data)
        if len(values) != self.dimensions:
            raise ValueError(
                f"Hugging Face model {self.model} returned {len(values)} dimensions, "
                f"but the pgvector schema expects {self.dimensions}."
            )
        return values


class SentenceTransformersEmbedder:
    """Local sentence-transformers embedder for bulk re-embedding.

    Use the same model as the Hugging Face Space so stored chunk vectors and
    deployed query vectors live in the same vector space.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        max_chars: int | None = None,
    ) -> None:
        self.model = model or os.getenv("SENTENCE_TRANSFORMERS_MODEL", DEFAULT_SENTENCE_TRANSFORMERS_MODEL)
        self.dimensions = dimensions
        self.max_chars = int(
            os.getenv(
                "SENTENCE_TRANSFORMERS_MAX_CHARS",
                max_chars or DEFAULT_SENTENCE_TRANSFORMERS_MAX_CHARS,
            )
        )
        self._model = None

    def embed_document(self, *, title: str, text: str) -> list[float]:
        return self._embed(f"title: {title or 'none'} | text: {text}")

    def embed_query(self, query: str) -> list[float]:
        return self._embed(query)

    def _embed(self, text: str) -> list[float]:
        values = self._load_model().encode(
            text[: self.max_chars],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        result = [float(value) for value in values.tolist()]
        if len(result) != self.dimensions:
            raise ValueError(
                f"sentence-transformers model {self.model} returned {len(result)} dimensions, "
                f"but the pgvector schema expects {self.dimensions}."
            )
        return result

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Install sentence-transformers first: "
                    "pip install 'sentence-transformers>=3.2'"
                ) from exc
            self._model = SentenceTransformer(self.model)
        return self._model


def _extract_ollama_embedding(data: dict) -> list[float]:
    values = data.get("embedding")
    if values is None and data.get("embeddings"):
        values = data["embeddings"][0]
    if not values:
        raise ValueError("Ollama embedding response did not contain values")
    return [float(value) for value in values]


def _extract_hf_embedding(data: dict) -> list[float]:
    values = data.get("embedding")
    if values is None and data.get("embeddings"):
        values = data["embeddings"][0]
    if not values:
        raise ValueError("Hugging Face embedding response did not contain values")
    return [float(value) for value in values]


def _stable_hash(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")
