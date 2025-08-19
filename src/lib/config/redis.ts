import type {

/**
 * Redis configuration (local native Redis integration)
 * Requires: npm install redis
 *
 * Uses REDIS_URL if set, otherwise defaults to tcp://127.0.0.1:6379
 * Provides:
 *  - getRedis(): singleton client
 *  - healthCheck()
 *  - pub/sub helpers (publish, subscribe)
 *  - basic key helpers (getJSON/setJSON/withTTL)
 */

    RedisClientType,
    RedisFunctions,
    RedisModules,
    RedisScripts
} from 'redis';

const DEFAULT_URL = 'redis://127.0.0.1:6379';

interface RedisConfig {
    url: string;
    username?: string;
    password?: string;
    socket?: {
        reconnectStrategy?: (retries: number) => number | Error;
    };
}

type Client = RedisClientType<RedisModules, RedisFunctions, RedisScripts>;

declare global {
    // eslint-disable-next-line no-var
    var __REDIS_CLIENT__: Client | undefined;
}

let connectPromise: Promise<Client> | null = null;

function buildConfig(): RedisConfig {
    const url = process.env.REDIS_URL?.trim() || DEFAULT_URL;

    return {
        url,
        username: process.env.REDIS_USERNAME || undefined,
        password: process.env.REDIS_PASSWORD || undefined,
        socket: {
            reconnectStrategy(retries) {
                // Max 10s backoff
                const delay = Math.min(100 + retries * 250, 10_000);
                return delay;
            }
        }
    };
}

async function createClientSingleton(): Promise<Client> {
    if (globalThis.__REDIS_CLIENT__) return globalThis.__REDIS_CLIENT__;
    const { createClient } = await import('redis');

    const client: Client = createClient(buildConfig());

    client.on('error', (err) => {
        console.error('[redis] error', err.message);
    });
    client.on('reconnecting', () => {
        console.warn('[redis] reconnecting...');
    });
    client.on('ready', () => {
        console.log('[redis] ready', client.options?.url || '');
    });

    await client.connect();
    globalThis.__REDIS_CLIENT__ = client;
    return client;
}

export async function getRedis(): Promise<Client> {
    if (globalThis.__REDIS_CLIENT__) return globalThis.__REDIS_CLIENT__;
    if (!connectPromise) connectPromise = createClientSingleton();
    return connectPromise;
}

/* Health check */
export async function healthCheck(timeoutMs = 500): Promise<{
    ok: boolean;
    latencyMs?: number;
    error?: string;
}> {
    try {
        const client = await getRedis();
        const start = performance.now();
        const pong = await Promise.race([
            client.ping(),
            new Promise<string>((_, reject) =>
                setTimeout(() => reject(new Error('timeout')), timeoutMs)
            )
        ]);
        const latency = performance.now() - start;
        return { ok: pong === 'PONG', latencyMs: latency };
    } catch (e: any) {
        return { ok: false, error: e?.message || 'unknown' };
    }
}

/* Pub/Sub helpers */
export async function publish(channel: string, payload: unknown) {
    const client = await getRedis();
    return client.publish(channel, JSON.stringify(payload));
}

export type SubscriptionHandler = (message: any) => void;

export async function subscribe(
    channel: string,
    handler: SubscriptionHandler
): Promise<() => Promise<void>> {
    const { createClient } = await import('redis');
    const sub = createClient(buildConfig());
    await sub.connect();
    await sub.subscribe(channel, (raw) => {
        try {
            handler(JSON.parse(raw));
        } catch {
            handler(raw);
        }
    });
    return async () => {
        try {
            await sub.unsubscribe(channel);
        } finally {
            await sub.quit();
        }
    };
}

/* Key helpers */
export async function setJSON<T>(
    key: string,
    value: T,
    opts?: { ttlSeconds?: number }
) {
    const client = await getRedis();
    const str = JSON.stringify(value);
    if (opts?.ttlSeconds) {
        await client.set(key, str, { EX: opts.ttlSeconds });
    } else {
        await client.set(key, str);
    }
}

export async function getJSON<T>(key: string): Promise<T | null> {
    const client = await getRedis();
    const v = await client.get(key);
    if (!v) return null;
    try {
        return JSON.parse(v) as T;
    } catch {
        return null;
    }
}

export async function withTTL<T>(
    key: string,
    ttlSeconds: number,
    compute: () => Promise<T>
): Promise<T> {
    const cached = await getJSON<T>(key);
    if (cached !== null) return cached;
    const value = await compute();
    await setJSON(key, value, { ttlSeconds });
    return value;
}

/* Graceful shutdown (call from server hooks / process signals) */
export async function closeRedis() {
    if (globalThis.__REDIS_CLIENT__) {
        try {
            await globalThis.__REDIS_CLIENT__.quit();
        } catch {}
        globalThis.__REDIS_CLIENT__ = undefined;
    }
}