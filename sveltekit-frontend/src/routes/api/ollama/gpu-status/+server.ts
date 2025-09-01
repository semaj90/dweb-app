/// <reference types="vite/client" />
import { ollamaService } from '$lib/server/services/OllamaService';
import type { RequestHandler } from './$types';
import { json } from '@sveltejs/kit';

// Configurable via env
const GO_BASE = (import.meta.env.GO_SERVICE_URL as string) || 'http://localhost:8084';
const TIMEOUT_MS = Number(import.meta.env.GPU_FETCH_TIMEOUT_MS) || 2500;
const RETRIES = Number(import.meta.env.GPU_FETCH_RETRIES) || 2;
const RETRY_DELAY_MS = Number(import.meta.env.GPU_FETCH_RETRY_DELAY_MS) || 300;
const CACHE_TTL_MS = Number(import.meta.env.GPU_STATUS_CACHE_TTL_MS) || 5_000;

type GPUStatus = {
  enabled: boolean;
  device?: string | number;
  [k: string]: any;
};

type FetchResult =
  | { ok: true; source: 'go'; gpu: GPUStatus }
  | { ok: false; source: 'cache' | 'shim'; gpu: GPUStatus; reason?: string };

const DEFAULT_SHIM: GPUStatus = { enabled: false };

let cached: { ts: number; payload: FetchResult } | null = null;

function isValidGpuStatus(payload: any): payload is GPUStatus {
  return !!payload && typeof payload === 'object' && typeof payload.enabled === 'boolean';
}

async function delay(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchWithTimeoutAndRetries(path: string, timeoutMs = TIMEOUT_MS, retries = RETRIES, delayMs = RETRY_DELAY_MS): Promise<GPUStatus> {
  const url = new URL(path, GO_BASE).href;

  let lastErr: any = null;
  for (let attempt = 0; attempt <= retries; attempt++) {
    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(url, { signal: controller.signal });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const jsonResp = await res.json();
      clearTimeout(t);

      if (!isValidGpuStatus(jsonResp)) {
        throw new Error('Invalid GPU status shape from upstream');
      }
      return jsonResp;
    } catch (err) {
      lastErr = err;
      clearTimeout(t);
      if (attempt < retries) {
        // exponential backoff with small jitter
        const backoff = delayMs * 2 ** attempt;
        const jitter = Math.floor(Math.random() * Math.max(10, Math.floor(backoff * 0.2)));
        await delay(backoff + jitter);
        continue;
      }
      throw lastErr;
    }
  }
  // should not reach here
  throw lastErr;
}

export const GET: RequestHandler = async () => {
  // Quick health check for Ollama service. Any failure -> graceful shim/cache fallback.
  try {
    const healthy = await ollamaService.isHealthy();
    if (!healthy) {
      // If we have a recent cached result, return it instead of a hard shim
      if (cached && Date.now() - cached.ts < CACHE_TTL_MS) {
        // return cached but mark as cache source
        return json({ ok: false, source: 'cache', gpu: cached.payload.gpu, reason: 'ollama_unhealthy' }, { status: 200 });
      }

      return json(
        { ok: false, source: 'shim', gpu: { ...DEFAULT_SHIM }, reason: 'ollama_unhealthy' },
        { status: 200 }
      );
    }
  } catch (err: any) {
    console.warn('gpu-status: ollama health check failed:', err?.message ?? err);
    if (cached && Date.now() - cached.ts < CACHE_TTL_MS) {
      return json({ ok: false, source: 'cache', gpu: cached.payload.gpu, reason: 'health_check_error' }, { status: 200 });
    }
    return json({ ok: false, source: 'shim', gpu: { ...DEFAULT_SHIM }, reason: 'health_check_error' }, { status: 200 });
  }

  // Serve from cache if fresh
  if (cached && Date.now() - cached.ts < CACHE_TTL_MS) {
    return json({ ok: true, source: 'cache', gpu: cached.payload.gpu }, { status: 200 });
  }

  // Attempt to fetch from GO service with retries and timeout
  try {
    const gpu = await fetchWithTimeoutAndRetries('/api/gpu-status', TIMEOUT_MS, RETRIES, RETRY_DELAY_MS);
    const payload: FetchResult = { ok: true, source: 'go', gpu };
    cached = { ts: Date.now(), payload };
    return json(payload, { status: 200 });
  } catch (err: any) {
    console.warn('gpu-status: upstream fetch failed:', err?.message ?? err);

    // if cache exists (even stale), return it as best-effort
    if (cached) {
      // keep cache timestamp but return a negative ok to indicate degraded state
      const payload: FetchResult = { ok: false, source: 'cache', gpu: cached.payload.gpu, reason: 'upstream_unreachable' };
      // refresh timestamp to avoid tight loops
      cached = { ts: Date.now(), payload };
      return json(payload, { status: 200 });
    }

    // final fallback shim
    const payload: FetchResult = { ok: false, source: 'shim', gpu: { ...DEFAULT_SHIM, enabled: false }, reason: 'upstream_unreachable' };
    return json(payload, { status: 200 });
  }
};
