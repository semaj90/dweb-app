import type { RequestHandler } from '@sveltejs/kit';
import { json } from "@sveltejs/kit";

const GO_BASE = process.env.GO_SERVICE_URL || process.env.GO_SERVER_URL || process.env.GO_MICROSERVICE_URL || "http://localhost:8084";

async function fetchWithTimeout(path: string, timeoutMs = 2500) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  
  try {
    const res = await fetch(`${GO_BASE}${path}`, { 
      signal: controller.signal 
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } finally {
    clearTimeout(timeout);
  }
}

export const GET: RequestHandler = async () => {
  try {
    const data = await fetchWithTimeout("/api/gpu-status");
    return json({ 
      ok: true, 
      source: "go", 
      data 
    });
  } catch (err) {
    // Fallback: try Go health endpoint for minimal GPU availability signal
    try {
      const health = await fetchWithTimeout("/api/health");
      const available = health?.services?.gpu === "enabled" || Boolean(health?.services?.gpu);
      
      return json({ 
        ok: true, 
        source: "go-health", 
        data: { available } 
      });
    } catch (e2) {
      return json(
        {
          ok: false,
          source: "shim",
          data: {
            available: false,
            message: "GPU status service unavailable",
            timestamp: new Date().toISOString()
          },
          error: (err as Error).message
        },
        { status: 200 }
      );
    }
  }
};