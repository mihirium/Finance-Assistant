from __future__ import annotations

import os
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from finance_rag.embeddings import HashingEmbedder, HuggingFaceEmbedder, OllamaEmbedder
from finance_rag.llm import answer_question
from finance_rag.models import SearchResult
from finance_rag.pgvector_store import DEFAULT_DATABASE_URL, PgVectorStore


EmbeddingProvider = Literal["ollama", "huggingface", "hashing"]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=20)
    synthesize: bool = True
    embedding_provider: EmbeddingProvider = "ollama"
    source_type: Literal["news", "filing"] | None = None


class SourceResponse(BaseModel):
    title: str
    url: str
    ticker: str | None = None
    sourceType: str
    score: float
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


class HealthResponse(BaseModel):
    ok: bool
    database: bool
    message: str


app = FastAPI(title="Finance AI API", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    store = PgVectorStore(database_url=_database_url(), embedder=_make_embedder("hashing"))
    try:
        store.ping()
    except Exception as exc:
        return HealthResponse(ok=False, database=False, message=str(exc))
    return HealthResponse(ok=True, database=True, message="ready")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    store = PgVectorStore(
        database_url=_database_url(),
        embedder=_make_embedder(request.embedding_provider),
    )
    try:
        results = store.search(
            request.message,
            top_k=request.top_k,
            source_type=request.source_type,
        )
        answer = answer_question(request.message, results, synthesize=request.synthesize)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(
        answer=answer,
        sources=[_source_from_result(result) for result in results],
    )


def _source_from_result(result: SearchResult) -> SourceResponse:
    chunk = result.chunk
    excerpt = chunk.text.strip()
    if len(excerpt) > 360:
        excerpt = excerpt[:357].rstrip() + "..."
    return SourceResponse(
        title=chunk.title,
        url=chunk.url,
        ticker=chunk.ticker,
        sourceType=chunk.source_type,
        score=result.score,
        excerpt=excerpt,
    )


def _make_embedder(provider: EmbeddingProvider):
    if provider == "hashing":
        return HashingEmbedder()
    if provider == "huggingface":
        return HuggingFaceEmbedder()
    return OllamaEmbedder()


def _database_url() -> str:
    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


def _cors_origins() -> list[str]:
    value = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    return [origin.strip() for origin in value.split(",") if origin.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def main() -> None:
    uvicorn.run(
        "finance_rag.api:app",
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("API_RELOAD", "0") == "1",
    )


if __name__ == "__main__":
    main()
