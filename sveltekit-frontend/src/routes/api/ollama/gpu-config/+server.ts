/// <reference types="vite/client" />
import { ollamaService } from '$lib/server/services/OllamaService';
import type { RequestHandler } from './$types';
import { json } from '@sveltejs/kit';

/**
 * Production-ready GPU config endpoint
 * - Configurable timeouts/retries via env
 * - Response validation
 * - In-memory cache with TTL
 * - Graceful fallbacks and minimal error exposure
 */

const GO_BASE = (import.meta.env.GO_SERVICE_URL as string) || 'http://localhost:8084';
const TIMEOUT_MS = Number(import.meta.env.GPU_FETCH_TIMEOUT_MS) || 2500;
const RETRIES = Number(import.meta.env.GPU_FETCH_RETRIES) || 2;
const RETRY_DELAY_MS = Number(import.meta.env.GPU_FETCH_RETRY_DELAY_MS) || 300;
const CACHE_TTL_MS = Number(import.meta.env.GPU_CONFIG_CACHE_TTL_MS) || 5_000;

// Minimal shape for expected config. Extend if backend returns more.
type GPUConfig = {
  model?: string;
  gpu: { enabled: boolean; device?: string | number } | { enabled: boolean };
  [k: string]: any;
};

type FetchResult = {
  ok: true;
  source: 'go';
  config: GPUConfig;
} | {
  ok: false;
  source: 'shim' | 'cache';
  config: GPUConfig;
  reason?: string;
};

// Simple in-memory cache to avoid frequent upstream calls
let cached: { ts: number; payload: FetchResult } | null = null;

function isValidGpuConfig(payload: any): payload is GPUConfig {
  if (!payload || typeof payload !== 'object') return false;
  if (!('gpu' in payload)) return false;
  const g = payload.gpu;
  if (typeof g !== 'object' || g === null) return false;
  // require enabled boolean
  if (typeof g.enabled !== 'boolean') return false;
  return true;
}

async function delay(ms: number) {
  return new Promise((res) => setTimeout(res, ms));
}

async function fetchOnce(path: string, timeoutMs: number): Promise<any> {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${GO_BASE}${path}`, { method: 'GET', signal: controller.signal });
    if (!res.ok) throw new Error(`upstream ${res.status}`);
    const data = await res.json();
    return data;
  } finally {
    clearTimeout(id);
  }
}

async function fetchWithRetries(path: string, retries = RETRIES, timeoutMs = TIMEOUT_MS, delayMs = RETRY_DELAY_MS): Promise<GPUConfig> {
  let attempt = 0;
  let lastError: any = null;
  while (attempt <= retries) {
    try {
      const payload = await fetchOnce(path, timeoutMs);
      if (!isValidGpuConfig(payload)) throw new Error('invalid-payload');
      return payload as GPUConfig;
    } catch (err) {
      lastError = err;
      attempt++;
      if (attempt > retries) break;
      // exponential backoff jitter
      const backoff = Math.round(delayMs * Math.pow(2, attempt - 1) + Math.random() * 50);
      await delay(backoff);
    }
  }
  throw lastError;
}

const DEFAULT_SHIM: GPUConfig = { model: 'gemma3-legal:latest', gpu: { enabled: false } };

export const GET: RequestHandler = async () => {
  try {
    const healthy = await Promise.resolve(ollamaService.isHealthy());
    if (!healthy) {
      // If Ollama isn't healthy return 503 and best-effort (cached or shim) config
      const fallback = cached?.payload ?? { ok: false, source: 'shim', config: DEFAULT_SHIM, reason: 'ollama_unhealthy' };
      return json({ ...fallback }, { status: 503 });
    }

    // Return cached if fresh
    if (cached && Date.now() - cached.ts < CACHE_TTL_MS) {
      return json(cached.payload, { status: 200 });
    }

    // Try upstream GO service
    try {
      const cfg = await fetchWithRetries('/api/gpu-status', RETRIES, TIMEOUT_MS, RETRY_DELAY_MS);
      const payload: FetchResult = { ok: true, source: 'go', config: cfg };
      cached = { ts: Date.now(), payload };
      return json(payload, { status: 200 });
    } catch (err: any) {
      console.warn('gpu-config: upstream fetch failed:', err?.message ?? err);
      // On failure use cache if available; otherwise return shim but indicate fallback
      if (cached) {
        const payload: FetchResult = { ok: false, source: 'cache', config: cached.payload.config, reason: 'upstream_unreachable' };
        // refresh timestamp to avoid tight loops
        cached = { ts: Date.now(), payload };
        return json(payload, { status: 200 });
      }

      const payload: FetchResult = { ok: false, source: 'shim', config: { ...DEFAULT_SHIM, gpu: { enabled: false } }, reason: 'upstream_unreachable' };
      // do not cache shim as a positive result; it's a fallback only
      return json(payload, { status: 200 });
    }
  } catch (err: any) {
    console.error('gpu-config: unexpected error', err?.message ?? err);
    // Minimal exposure to clients; return shim and 500
    const payload: FetchResult = { ok: false, source: 'shim', config: DEFAULT_SHIM, reason: 'internal_error' };
    return json(payload, { status: 500 });
  }
};
