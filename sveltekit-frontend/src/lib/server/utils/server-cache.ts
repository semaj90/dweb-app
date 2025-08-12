import { CacheManager } from "../../../../../src/lib/server/cache/loki-cache";

const cache = new CacheManager({ dbPath: "cache/ai-cache.db" });

export type CachePayload = unknown;

export async function setCache(key: string, data: CachePayload) {
  await cache.set(key, data, { contentType: "summary", ttl: 24 * 3600 * 1000 });
}

export async function getCache<T = unknown>(key: string): Promise<T | null> {
  return (await cache.get(key)) as T | null;
}
