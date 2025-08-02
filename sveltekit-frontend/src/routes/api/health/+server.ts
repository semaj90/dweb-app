import { json } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
import { redisVectorService } from "$lib/services/redis-vector-service";

export const GET: RequestHandler = async () => {
  const startTime = Date.now();
  const health = {
    status: "healthy",
    timestamp: new Date().toISOString(),
    services: {
      ollama: "unknown",
      qdrant: "unknown",
      redis: "unknown",
      gpu: "unknown",
    },
    performance: {
      responseTime: 0,
    },
  };

  // Check Ollama
  try {
    const ollamaResponse = await fetch("http://localhost:11434/api/tags", {
      signal: AbortSignal.timeout(5000),
    });
    health.services.ollama = ollamaResponse.ok ? "healthy" : "degraded";
  } catch {
    health.services.ollama = "unhealthy";
  }

  // Check Qdrant
  try {
    const qdrantResponse = await fetch("http://localhost:6333/collections", {
      signal: AbortSignal.timeout(5000),
    });
    health.services.qdrant = qdrantResponse.ok ? "healthy" : "degraded";
  } catch {
    health.services.qdrant = "unhealthy";
  }

  // Check Redis
  try {
    // const redisHealthy = await redisVectorService.healthCheck();
    health.services.redis = "unknown"; // TODO: implement healthCheck
  } catch {
    health.services.redis = "unhealthy";
  }

  // Check GPU status
  try {
    const gpuResponse = await fetch("http://localhost:11434/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "gemma3-legal",
        prompt: "test",
        stream: false,
      }),
      signal: AbortSignal.timeout(10000),
    });
    health.services.gpu = gpuResponse.ok ? "accelerated" : "cpu_fallback";
  } catch {
    health.services.gpu = "unavailable";
  }

  health.performance.responseTime = Date.now() - startTime;

  const unhealthyServices = Object.values(health.services).filter(
    (s) => s === "unhealthy"
  ).length;
  if (unhealthyServices > 1) health.status = "degraded";
  if (unhealthyServices > 2) health.status = "critical";

  const responseStatus = health.status === "healthy" ? 200 : 503;
  return json(health, {
    status: responseStatus,
  });
};
