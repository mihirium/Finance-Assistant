from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


DEFAULT_EMBEDDING_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_GENERATION_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1)
    model: str | None = None


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


class Context(BaseModel):
    id: int
    title: str = ""
    ticker: str | None = None
    sourceType: str | None = None
    url: str = ""
    text: str


class GenerateRequest(BaseModel):
    question: str = Field(min_length=1)
    contexts: list[Context] = Field(default_factory=list)


class GenerateResponse(BaseModel):
    answer: str


app = FastAPI(title="Finance AI Hugging Face Space")


@app.get("/")
def root() -> dict[str, str]:
    return {"ok": "true", "service": "finance-ai-hf-space"}


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "ok": "true",
        "embedding_model": _embedding_model_id(),
        "generation_model": _generation_model_id(),
    }


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest) -> EmbedResponse:
    if request.model and request.model != _embedding_model_id():
        raise ValueError(f"This Space is loaded with {_embedding_model_id()}, not {request.model}.")
    embeddings = _embedding_model().encode(
        request.texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return EmbedResponse(embeddings=embeddings.tolist())


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    if not request.contexts:
        return GenerateResponse(answer="I could not find enough context to answer that question.")

    prompt = _build_prompt(request.question, request.contexts)
    output = _generator()(
        prompt,
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "360")),
        do_sample=False,
        return_full_text=False,
    )
    answer = _read_generated_text(output).strip()
    return GenerateResponse(answer=answer or _fallback_answer(request.contexts))


@lru_cache(maxsize=1)
def _embedding_model() -> SentenceTransformer:
    return SentenceTransformer(_embedding_model_id())


@lru_cache(maxsize=1)
def _generator():
    model_id = _generation_model_id()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def _embedding_model_id() -> str:
    return os.getenv("EMBEDDING_MODEL_ID", DEFAULT_EMBEDDING_MODEL_ID)


def _generation_model_id() -> str:
    return os.getenv("GENERATION_MODEL_ID", DEFAULT_GENERATION_MODEL_ID)


def _build_prompt(question: str, contexts: list[Context]) -> str:
    context_text = "\n\n".join(
        f"[{context.id}] {context.title}\n{context.text[:1200]}" for context in contexts[:8]
    )
    return (
        "You are a finance assistant. Answer using only the provided context. "
        "Write 2 to 4 concise paragraphs. Cite sources inline like [1] or [2]. "
        "If the context is not enough, say what is missing.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context_text}\n\n"
        "Answer:\n"
    )


def _read_generated_text(output: Any) -> str:
    if isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, dict):
            return str(first.get("generated_text", ""))
    return str(output)


def _fallback_answer(contexts: list[Context]) -> str:
    snippets = " ".join(f"[{context.id}] {context.text[:240]}" for context in contexts[:3])
    return f"The retrieved context points to these relevant details: {snippets}"
