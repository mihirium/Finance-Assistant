import { NextResponse } from "next/server";
import { Pool } from "pg";

export const runtime = "nodejs";
export const maxDuration = 300;

type ChatRequest = {
  message?: string;
  top_k?: number;
  source_type?: "news" | "filing";
};

type Source = {
  title: string;
  url: string;
  ticker: string | null;
  sourceType: "filing" | "news" | "price";
  score: number;
  excerpt: string;
};

type SourceRow = {
  title: string;
  url: string;
  ticker: string | null;
  sourceType: "filing" | "news" | "price";
  score: string | number;
  excerpt: string;
};

declare global {
  // eslint-disable-next-line no-var
  var financeAiPgPool: Pool | undefined;
}

export async function POST(request: Request) {
  const body = (await request.json()) as ChatRequest;
  const message = body.message?.trim();

  if (!message) {
    return NextResponse.json({ error: "Message is required" }, { status: 400 });
  }

  const backendUrl = process.env.BACKEND_API_URL;
  if (backendUrl) {
    return proxyToPythonBackend(backendUrl, body, message);
  }

  const databaseUrl = process.env.SUPABASE_DATABASE_URL ?? process.env.DATABASE_URL;
  const embeddingUrl = process.env.HF_EMBEDDING_URL;

  if (!databaseUrl || !embeddingUrl) {
    return NextResponse.json(
      {
        error: "Missing deployment configuration",
        detail: "Set SUPABASE_DATABASE_URL and HF_EMBEDDING_URL in Vercel."
      },
      { status: 500 }
    );
  }

  try {
    const embedding = await embedQuestion(embeddingUrl, message);
    const sources = await retrieveSources({
      databaseUrl,
      embedding,
      topK: body.top_k ?? 8,
      sourceType: body.source_type
    });
    const answer = await synthesizeAnswer(message, sources);
    return NextResponse.json({ answer, sources });
  } catch (error) {
    const detail = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: "Chat request failed", detail }, { status: 500 });
  }
}

async function proxyToPythonBackend(backendUrl: string, body: ChatRequest, message: string) {
  const response = await fetch(`${backendUrl.replace(/\/$/, "")}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      top_k: body.top_k ?? 8,
      synthesize: true,
      embedding_provider: "huggingface",
      source_type: body.source_type
    })
  });

  const data = await response.json();
  return NextResponse.json(data, { status: response.status });
}

async function embedQuestion(embeddingUrl: string, question: string) {
  const response = await fetch(embeddingUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(process.env.HF_TOKEN ? { Authorization: `Bearer ${process.env.HF_TOKEN}` } : {})
    },
    body: JSON.stringify({
      model: process.env.HF_EMBEDDING_MODEL ?? "sentence-transformers/all-mpnet-base-v2",
      texts: [question]
    })
  });

  if (!response.ok) {
    throw new Error(`Hugging Face embedding request failed: ${response.status}`);
  }

  const data = (await response.json()) as { embedding?: number[]; embeddings?: number[][] };
  const embedding = data.embedding ?? data.embeddings?.[0];
  if (!embedding?.length) {
    throw new Error("Hugging Face embedding response did not include an embedding.");
  }
  return embedding;
}

async function retrieveSources({
  databaseUrl,
  embedding,
  topK,
  sourceType
}: {
  databaseUrl: string;
  embedding: number[];
  topK: number;
  sourceType?: "news" | "filing";
}) {
  const pool = getPool(databaseUrl);
  const vector = `[${embedding.join(",")}]`;
  const limit = Math.max(1, Math.min(topK, 20));
  const where = sourceType ? "WHERE source_type = $2" : "";
  const params = sourceType ? [vector, sourceType, vector, limit] : [vector, vector, limit];

  const result = await pool.query<SourceRow>(
    `
    SELECT
      title,
      url,
      ticker,
      source_type AS "sourceType",
      1 - (embedding <=> $1::vector) AS score,
      left(chunk_text, 900) AS excerpt
    FROM chunks
    ${where}
    ORDER BY embedding <=> ${sourceType ? "$3" : "$2"}::vector
    LIMIT ${sourceType ? "$4" : "$3"}
    `,
    params
  );

  return result.rows.map((row) => ({
    title: row.title,
    url: row.url,
    ticker: row.ticker,
    sourceType: row.sourceType,
    score: Number(row.score),
    excerpt: row.excerpt
  }));
}

async function synthesizeAnswer(question: string, sources: Source[]) {
  if (sources.length === 0) {
    return "I could not find relevant filing or news context in Supabase for that question yet.";
  }

  const generationUrl = process.env.HF_GENERATION_URL;
  if (!generationUrl) {
    return extractiveAnswer(question, sources);
  }

  const response = await fetch(generationUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(process.env.HF_TOKEN ? { Authorization: `Bearer ${process.env.HF_TOKEN}` } : {})
    },
    body: JSON.stringify({
      question,
      contexts: sources.map((source, index) => ({
        id: index + 1,
        title: source.title,
        ticker: source.ticker,
        sourceType: source.sourceType,
        url: source.url,
        text: source.excerpt
      }))
    })
  });

  if (!response.ok) {
    return extractiveAnswer(question, sources);
  }

  const data = (await response.json()) as { answer?: string };
  return data.answer?.trim() || extractiveAnswer(question, sources);
}

function extractiveAnswer(question: string, sources: Source[]) {
  const context = sources
    .slice(0, 4)
    .map((source, index) => `[${index + 1}] ${source.excerpt}`)
    .join("\n\n");
  return `Here is the best retrieved context for "${question}".\n\n${context}`;
}

function getPool(databaseUrl: string) {
  if (!globalThis.financeAiPgPool) {
    globalThis.financeAiPgPool = new Pool({
      connectionString: databaseUrl,
      ssl: { rejectUnauthorized: false },
      max: 3
    });
  }
  return globalThis.financeAiPgPool;
}
