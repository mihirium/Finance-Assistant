---
title: Finance Assistant
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Finance AI Hugging Face Space

This Space is the model service for the no-card demo deployment.

It exposes:

- `POST /embed` - converts text into 768-dimensional embeddings
- `POST /generate` - writes a concise cited answer from retrieved contexts

## Suggested Space Settings

```text
SDK: Docker
Hardware: CPU Basic
```

The default embedding model is:

```text
sentence-transformers/all-mpnet-base-v2
```

It returns 768-dimensional vectors, matching the current pgvector schema.

The default answer model is:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

It is small enough for a slow free demo. You can override either model with:

```text
EMBEDDING_MODEL_ID=sentence-transformers/all-mpnet-base-v2
GENERATION_MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
```

## Vercel Environment Variables

After the Space is deployed, set:

```text
HF_EMBEDDING_URL=https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space/embed
HF_GENERATION_URL=https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space/generate
SUPABASE_DATABASE_URL=postgresql://...
```

## Local Test

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

```bash
curl -X POST http://127.0.0.1:7860/embed \
  -H "Content-Type: application/json" \
  -d '{"texts":["Apple supply chain risks"]}'
```

```bash
curl -X POST http://127.0.0.1:7860/generate \
  -H "Content-Type: application/json" \
  -d '{"question":"What moved markets today?","contexts":[{"id":1,"title":"Stocks rally after inflation report","text":"Major indexes rose after inflation came in below expectations."}]}'
```
