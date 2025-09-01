import { json } from '@sveltejs/kit';
import { logger } from "$lib/server/logger";
import type { RequestHandler } from './$types';
import { apiSuccess, apiError, getRequestId, withErrorHandling } from '$lib/server/api/standard-response';

// Mock Ollama service for now - replace with actual service when available
const ollamaService = {
  async isHealthy(): Promise<boolean> {
    try {
      const response = await fetch('http://localhost:11434/api/version', { 
        signal: AbortSignal.timeout(5000) 
      });
      return response.ok;
    } catch {
      return false;
    }
  },
  async listModels(): Promise<Array<{ name: string }>> {
    try {
      const response = await fetch('http://localhost:11434/api/tags');
      if (response.ok) {
        const data = await response.json();
        return data.models || [];
      }
      return [];
    } catch {
      return [];
    }
  }
};

export const GET: RequestHandler = withErrorHandling(async (event) => {
  const requestId = getRequestId(event);
  
  // Check Ollama health with model list
  const ollamaStartTime = Date.now();
  const ollamaHealthy = await ollamaService.isHealthy();
  const ollamaResponseTime = Date.now() - ollamaStartTime;

    let availableModels: string[] = [];
    if (ollamaHealthy) {
      try {
        const models = await ollamaService.listModels();
        availableModels = models.map((m) => m.name);
      } catch (error: any) {
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

    const healthData = {
      status: overallStatus,
      services: checks,
      message:
        overallStatus === "healthy"
          ? `All systems operational (${availableModels.length} models available)`
          : "Ollama service not available",
    };

    return apiSuccess(healthData, undefined, requestId);
});
