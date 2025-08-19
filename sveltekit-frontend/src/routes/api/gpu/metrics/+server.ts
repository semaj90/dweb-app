import type { RequestHandler }, {
json } from "@sveltejs/kit";

const GO_BASE =
  import.meta.env.GO_SERVICE_URL ||
  import.meta.env.GO_SERVER_URL ||
  import.meta.env.GO_MICROSERVICE_URL ||
  "http://localhost:8084";

async function fetchWithTimeout(path: string, timeoutMs = 2500) {
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${GO_BASE}${path}`, { signal: controller.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } finally {
    clearTimeout(t);
  }
}

export const GET: RequestHandler = async () => {
  try {
    const data = await fetchWithTimeout("/api/metrics");
    return json({ ok: true, source: "go", data });
  } catch (err) {
    return json({
      ok: false,
      source: "shim",
      data: { message: "Metrics unavailable" },
      error: (err as Error).message,
    });
  }
};
