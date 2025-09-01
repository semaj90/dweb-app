/// <reference types="vite/client" />
import type { RequestHandler } from './$types';

/**
 * GPU Setup Validation API - Check AI readiness for Legal PDF processing
 * Tests GPU, Ollama, and model availability
 */

const GO_BASE = import.meta.env.GO_SERVICE_URL || import.meta.env.GO_SERVER_URL || import.meta.env.GO_MICROSERVICE_URL || "http://localhost:8084";

// Tiny retry wrapper to smooth transient failures 
async function fetchWithTimeout<T = unknown>(path: string, timeoutMs = 2500): Promise<T> {
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${GO_BASE}${path}`, { signal: controller.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return (await res.json()) as T;
  } finally {
    clearTimeout(t);
  }
}

// Ollama service helper
const ollamaService = {
  isHealthy: async () => {
    try {
      const response = await fetch('http://localhost:11434/api/tags');
      return response.ok;
    } catch {
      return false;
    }
  },
  listModels: async () => {
    try {
      const response = await fetch('http://localhost:11434/api/tags');
      if (response.ok) {
        const data = await response.json();
        return data.models || [];
      }
    } catch {
      return [];
    }
    return [];
  }
};

type OllamaModel = {
  name?: string;
};

type GPUStatus = {
  available?: boolean;
  cuda?: {
    available?: boolean;
    version?: string;
    device_id?: number;
  };
  device?: {
    name?: string;
  };
  memory?: {
    free?: number;
    total?: number;
    used?: number;
  };
  temperature_c?: number;
};

type HealthPayload = {
  services?: {
    gpu?: string | boolean;
  };
};

type AISummarizeChecks = {
  gpu: boolean;
  ollama: boolean;
  model: boolean;
};

type Details = {
  ollama: {
    ok: boolean;
    models_count: number;
    required_model: string;
    model_present: boolean;
  };
  go_service: {
    ok: boolean;
    endpoint: string;
    source?: string;
    version?: string;
    raw?: GPUStatus;
    health?: HealthPayload;
  };
  ai_summarize_checks: AISummarizeChecks;
};

type CachePayload = {
  ok: boolean;
  details: Details;
  message?: string;
};

// Simple in-process cache
let cache: { ts?: number; data?: CachePayload } = {};

export const GET: RequestHandler = async () => {
  try {
    // Simple in-process cache (5s) to avoid repeated probing and UI flapping
    const now = Date.now();
    
    if (cache.ts && cache.data && now - cache.ts < 5000) {
      return json(cache.data);
    }

    // Try gpu-status once; if it fails, wait briefly and retry once
    const gpuAttempt = await Promise.allSettled([
      fetchWithTimeout<GPUStatus>("/api/gpu-status"),
    ]);
    
    let gpuStatus = gpuAttempt[0];
    if (gpuStatus.status === "rejected") {
      await new Promise((r) => setTimeout(r, 150));
      gpuStatus = (await Promise.allSettled([fetchWithTimeout<GPUStatus>("/api/gpu-status")]))[0];
    }

    const [ollamaHealthy, health] = await Promise.allSettled([
      ollamaService.isHealthy(),
      fetchWithTimeout<HealthPayload>("/api/health"),
    ]);

    const ollama_ok = ollamaHealthy.status === "fulfilled" && Boolean(ollamaHealthy.value);

    // If Ollama is healthy, fetch models and check for required model
    const requiredModel = import.meta.env.PRIMARY_MODEL || import.meta.env.MODEL_NAME || "gemma3-legal:latest";
    let models: OllamaModel[] = [];
    
    if (ollama_ok) {
      try {
        models = await ollamaService.listModels();
      } catch {
        models = [];
      }
    }

    const model_present = models.some(
      (m) => m?.name === requiredModel || m?.name?.startsWith(requiredModel)
    );

    const gpu_ok = (gpuStatus.status === "fulfilled" && (() => {
        const v = gpuStatus.value;
        return Boolean(v?.available ?? v?.cuda?.available);
      })()) ||
      (health.status === "fulfilled" && Boolean(health.value?.services?.gpu));

    const details: Details = {
      ollama: {
        ok: ollama_ok,
        models_count: models.length,
        required_model: requiredModel,
        model_present
      },
      go_service: {
        ok: gpu_ok,
        endpoint: `${GO_BASE}/api/gpu-status`,
        source: gpuStatus.status === "fulfilled" ? "go" : "shim",
        version: "v1",
        raw: gpuStatus.status === "fulfilled" ? gpuStatus.value : undefined,
        health: health.status === "fulfilled" ? health.value : undefined
      },
      ai_summarize_checks: {
        gpu: gpu_ok,
        ollama: ollama_ok,
        model: model_present
      }
    };

    const ai_ready = details.ai_summarize_checks.gpu && 
                    details.ai_summarize_checks.ollama && 
                    details.ai_summarize_checks.model;

    const message = !model_present ? "Please download local LLM model" : undefined;

    const payload: CachePayload = {
      ok: ai_ready,
      details,
      message
    };

    // Cache the result
    cache = { ts: now, data: payload };

    return json(payload);
    
  } catch (error: any) {
    console.error('GPU validation error:', error);
    return json({
      ok: false,
      error: error instanceof Error ? error.message : 'Unknown error',
      timestamp: new Date().toISOString()
    });
  }
};