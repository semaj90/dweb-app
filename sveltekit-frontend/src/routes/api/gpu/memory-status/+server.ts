/// <reference types="vite/client" />

import type { RequestHandler } from './$types';

// Base URL for Go GPU status service (fallback to localhost)
const GO_BASE =
    import.meta.env.GO_SERVICE_URL ||
    import.meta.env.GO_SERVER_URL ||
    import.meta.env.GO_MICROSERVICE_URL ||
    'http://localhost:8084';

async function fetchWithTimeout(path: string, timeoutMs = 2500): Promise<any> {
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
        const data = await fetchWithTimeout('/api/gpu-status');
        const memory = {
            free: data?.memory?.free ?? null,
            total: data?.memory?.total ?? null,
            used:
                data?.memory?.total != null && data?.memory?.free != null
                    ? data.memory.total - data.memory.free
                    : null
        };
        return json({ ok: true, source: 'go', memory });
    } catch (err: any) {
        return json({
            ok: false,
            source: 'shim',
            memory: { free: null, total: null, used: null }
        });
    }
};