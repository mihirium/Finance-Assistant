import { NextResponse } from "next/server";
import { Pool } from "pg";

export const runtime = "nodejs";

type SubscribeRequest = {
  email?: string;
};

declare global {
  // eslint-disable-next-line no-var
  var financeAiSubscribePgPool: Pool | undefined;
}

export async function POST(request: Request) {
  const body = (await request.json()) as SubscribeRequest;
  const email = body.email?.trim().toLowerCase();

  if (!email || !isValidEmail(email)) {
    return NextResponse.json({ error: "Enter a valid email address." }, { status: 400 });
  }

  const databaseUrl = process.env.SUPABASE_DATABASE_URL ?? process.env.DATABASE_URL;
  if (!databaseUrl) {
    return NextResponse.json(
      { error: "Missing database configuration." },
      { status: 500 }
    );
  }

  try {
    const pool = getPool(databaseUrl);
    await pool.query(
      `
      INSERT INTO email_subscribers (email, subscribed, updated_at)
      VALUES ($1, true, now())
      ON CONFLICT (email) DO UPDATE SET
        subscribed = true,
        updated_at = now()
      `,
      [email]
    );
    return NextResponse.json({ ok: true });
  } catch (error) {
    const detail = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: "Could not save subscription.", detail }, { status: 500 });
  }
}

function isValidEmail(email: string) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

function getPool(databaseUrl: string) {
  if (!globalThis.financeAiSubscribePgPool) {
    globalThis.financeAiSubscribePgPool = new Pool({
      connectionString: databaseUrl,
      ssl: { rejectUnauthorized: false },
      max: 3
    });
  }
  return globalThis.financeAiSubscribePgPool;
}
