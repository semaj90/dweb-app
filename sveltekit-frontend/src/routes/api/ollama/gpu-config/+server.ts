import { ollamaService } from "$lib/server/services/OllamaService";
import type { RequestHandler }, {
json } from "@sveltejs/kit";

const GO_BASE = import.meta.env.GO_SERVICE_URL || "http://localhost:8084";

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
  const healthy = await ollamaService.isHealthy();
  if (!healthy) {
    return json({
      ok: false,
      source: "shim",
      config: { model: "gemma3-legal:latest", gpu: { enabled: false } },
    });
  }
  try {
    const data = await fetchWithTimeout("/api/gpu-status");
    return json({ ok: true, source: "go", config: data });
  } catch (err) {
    return json({
      ok: true,
      source: "shim",
      config: { model: "gemma3-legal:latest", gpu: { enabled: true } },
    });
  }
};
