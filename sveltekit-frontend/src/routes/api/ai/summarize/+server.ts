import { json } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";

// Frontend summarize endpoint (TypeScript)
// Proxies to the Go summarizer service to keep UI lightweight.

const SUMMARIZER_BASE =
  process.env.SUMMARIZER_BASE_URL || "http://localhost:8091";

export const GET: RequestHandler = async () => {
  try {
    const res = await fetch(`${SUMMARIZER_BASE}/health`);
    const body = await res.json().catch(() => ({}));
    return json(
      {
        ok: res.ok,
        status: res.status,
        health: body,
        target: `${SUMMARIZER_BASE}/health`,
      },
      { status: res.ok ? 200 : res.status }
    );
  } catch (e: unknown) {
    return json(
      {
        ok: false,
        error:
          e instanceof Error ? e.message : "Summarizer service unreachable",
        target: `${SUMMARIZER_BASE}/health`,
      },
      { status: 503 }
    );
  }
};

export const POST: RequestHandler = async ({ request }) => {
  try {
    const payload = await request.json().catch(() => ({}));
    const res = await fetch(`${SUMMARIZER_BASE}/summarize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const text = await res.text();
    let data: unknown;
    try {
      data = JSON.parse(text);
    } catch {
      data = { message: text };
    }

    if (!res.ok) {
      const err =
        (data && typeof data === "object"
          ? (data as Record<string, unknown>)
          : {}) ?? {};
      return json(
        {
          ok: false,
          status: res.status,
          error:
            (typeof err.error === "string" && err.error) ||
            (typeof err.message === "string" && err.message) ||
            "Summarizer error",
        },
        { status: res.status }
      );
    }

    const okData =
      data && typeof data === "object"
        ? (data as Record<string, unknown>)
        : { data };
    return json({ ok: true, ...okData });
  } catch (e: unknown) {
    return json(
      {
        ok: false,
        error: e instanceof Error ? e.message : "Failed to call summarizer",
      },
      { status: 500 }
    );
  }
};
