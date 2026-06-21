from __future__ import annotations

import json
import os
import re
import textwrap
import urllib.request
from urllib.error import HTTPError, URLError

from finance_rag.embeddings import DEFAULT_OLLAMA_HOST
from finance_rag.models import SearchResult


DEFAULT_OLLAMA_CHAT_MODEL = "llama3.2"
ENTITY_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9&.-]{2,}\b")
STOPWORDS = {
    "about",
    "after",
    "before",
    "could",
    "disclose",
    "disclosed",
    "discloses",
    "from",
    "highly",
    "into",
    "it",
    "not",
    "that",
    "the",
    "this",
    "today",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
}


def answer_question(question: str, results: list[SearchResult], *, synthesize: bool = True) -> str:
    if not results:
        return "I could not find relevant context in the local index. Try ingesting today's news again."

    warning = _direct_context_warning(question, results)
    if synthesize:
        try:
            return _ollama_answer(question, results)
        except Exception as exc:
            print(f"warning: Ollama answer synthesis failed, falling back to extractive answer: {exc}")

    return _extractive_answer(question, results, warning=warning)


def _ollama_answer(question: str, results: list[SearchResult]) -> str:
    prompt = _build_prompt(question, results)
    host = (os.getenv("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
    model = os.getenv("OLLAMA_CHAT_MODEL", DEFAULT_OLLAMA_CHAT_MODEL)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    request = urllib.request.Request(
        f"{host}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama generate request failed: HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError("Ollama is not reachable. Start Ollama or use extractive answers.") from exc
    text = data.get("response", "").strip()
    if not text:
        raise ValueError("Ollama response did not contain generated text")
    return _ensure_citations(text, results)


def _build_prompt(question: str, results: list[SearchResult]) -> str:
    context = "\n\n".join(
        f"[{idx}] {result.chunk.title}\nSource: {result.chunk.url}\n{result.chunk.text}"
        for idx, result in enumerate(results, start=1)
    )
    return textwrap.dedent(
        f"""
        You are a finance research assistant. Answer the user's question using only
        the supplied retrieved context.

        Write a clear answer in 2-4 short paragraphs. Do not use bullet points,
        numbered lists, tables, or raw excerpts. Lead with the direct answer, then
        explain the supporting details in prose.

        Cite sources inline as [1], [2] after the claims they support. Every
        paragraph must include at least one citation. If the context is
        insufficient, say what is missing and cite the closest available context.

        Question: {question}

        Retrieved context:
        {context}
        """
    ).strip()


def _ensure_citations(answer: str, results: list[SearchResult]) -> str:
    if re.search(r"\[\d+\]", answer):
        return answer
    citations = ", ".join(f"[{idx}]" for idx in range(1, min(len(results), 3) + 1))
    return f"{answer}\n\nSources: {citations}"


def _extractive_answer(
    question: str,
    results: list[SearchResult],
    *,
    warning: str | None = None,
) -> str:
    lines = [f"Best retrieved context for: {question}", ""]
    if warning:
        lines.extend([warning, ""])
    for idx, result in enumerate(results[:5], start=1):
        snippet = result.chunk.text[:700].strip()
        if len(result.chunk.text) > 700:
            snippet += "..."
        lines.extend(
            [
                f"[{idx}] {result.chunk.title} (news, score={result.score:.2f})",
                snippet,
                f"Source: {result.chunk.url}",
                "",
            ]
        )
    lines.append("Use --no-synthesis to inspect raw retrieval context.")
    return "\n".join(lines).strip()


def _direct_context_warning(question: str, results: list[SearchResult]) -> str | None:
    terms = _important_question_terms(question)
    if not terms:
        return None

    context = " ".join(f"{result.chunk.title} {result.chunk.text}" for result in results[:5]).lower()
    missing = [term for term in terms if term.lower() not in context]
    if not missing:
        return None

    return (
        "Note: the local index does not appear to contain direct context for "
        f"{', '.join(missing)}. These results may be only loosely related."
    )


def _important_question_terms(question: str) -> list[str]:
    terms = []
    for match in ENTITY_RE.finditer(question):
        term = match.group(0)
        normalized = term.lower()
        if normalized in STOPWORDS or len(term) < 4:
            continue
        terms.append(term)
    return terms[:4]
