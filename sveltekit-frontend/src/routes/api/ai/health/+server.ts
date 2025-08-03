import type { RequestHandler } from "@sveltejs/kit";
import { json } from "@sveltejs/kit";
import { ollamaService } from "$lib/server/services/OllamaService";
import { logger } from "$lib/server/logger";

export const GET: RequestHandler = async () => {
  try {
    // Check Ollama health with model list
    const ollamaStartTime = Date.now();
    const ollamaHealthy = await ollamaService.isHealthy();
    const ollamaResponseTime = Date.now() - ollamaStartTime;

    let availableModels: string[] = [];
    if (ollamaHealthy) {
      try {
        const models = await ollamaService.listModels();
        availableModels = models.map((m) => m.name);
      } catch (error) {
        logger.warn("Failed to list Ollama models", error);
      }
    }

    // System information
    const memoryUsage = process.memoryUsage();
    const memoryMB = Math.round(memoryUsage.rss / 1024 / 1024);

    const checks = {
      ollama: {
        healthy: ollamaHealthy,
        models: availableModels,
        responseTime: ollamaResponseTime,
        url: "http://localhost:11434",
      },
      system: {
        memory: `${memoryMB}MB`,
        uptime: `${Math.round((process.uptime() / 3600) * 100) / 100}h`,
        nodeVersion: process.version,
      },
      docker: {
        containers: 0, // TODO: Add actual Docker container count
      },
      timestamp: new Date().toISOString(),
    };

    const overallStatus = ollamaHealthy ? "healthy" : "degraded";

    return json({
      status: overallStatus,
      services: checks,
      message:
        overallStatus === "healthy"
          ? `All systems operational (${availableModels.length} models available)`
          : "Ollama service not available",
    });
  } catch (error) {
    logger.error("Health check failed", error);
    return json(
      {
        status: "critical",
        error: (error as Error).message,
        timestamp: new Date().toISOString(),
        services: {
          ollama: { healthy: false, models: [], responseTime: 0 },
          system: { memory: "0MB", uptime: "0h", nodeVersion: process.version },
        },
      },
      { status: 503 }
    );
  }
};
