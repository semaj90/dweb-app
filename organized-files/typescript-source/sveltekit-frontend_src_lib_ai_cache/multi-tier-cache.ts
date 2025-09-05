const fs = require('fs');
const path = require('path');

/**
 * Multi-Tier Cache for AI / Auto-Encoder Artifacts
 *
 * Tiers (fastest → slowest):
 *  1. In-memory LRU (process local)
 *  2. LokiJS collection (structured in-memory, optional persistence)
 *  3. Redis (server runtime only)
 *  4. IndexedDB (browser runtime only)
 *  5. pgvector (latent vectors) – stub via API endpoint
 *  6. MinIO (binary blobs) – stub via API endpoint / signed upload
 *
 * Notes:
 *  - Each tier is optional; failures degrade gracefully to lower tiers.
 *  - Designed to avoid SSR failures (dynamic imports + env guards).
 *  - Vector persistence & MinIO uploads are stubs you can implement server-side.
 *
 * Environment file support:
 *  - When running in Node, this file will attempt to read a local ".env"
 *    (or a custom path) and populate process.env for keys that aren't set.
 *    This is a simple, file-system based fallback so redis/other URLs can be
 *    detected without an external dependency.
 */

(function loadEnvFromFile(filePath = '.env') {
  // Detect Node runtime
  const runningOnNode = typeof process !== 'undefined' && typeof require === 'function' && !!(process && process.versions && process.versions.node);
  if (!runningOnNode) return;

  try {
    // Use require to keep this file safe for bundlers / browsers (only executed in Node)
    const resolved = path.resolve(process.cwd(), filePath);
    if (!fs.existsSync(resolved)) return;

    const content = fs.readFileSync(resolved, 'utf8');
    for (const rawLine of content.split(/\r?\n/)) {
      const line = rawLine.trim();
      if (!line || line.startsWith('#')) continue;
      const match = line.match(/^([\w.-]+)\s*=\s*(.*)$/);
      if (!match) continue;
      const key = match[1];
      let val = match[2] || '';
      // Remove surrounding quotes if present
      if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) {
        val = val.slice(1, -1);
      }
      // Only set if not already provided in environment
      if (process.env[key] === undefined) process.env[key] = val;
    }
  } catch {
    // best-effort only; silently ignore errors
  }
})();

class LRU<K, V> {
  private max: number;
  private map = new Map<K, V>();
  constructor(max = 256) { this.max = max; }
  get(key: K): V | undefined {
    if (!this.map.has(key)) return undefined;
    const val = this.map.get(key)!;
    this.map.delete(key);
    this.map.set(key, val);
    return val;
  }
  set(key: K, val: V) {
    if (this.map.has(key)) this.map.delete(key);
    this.map.set(key, val);
    if (this.map.size > this.max) {
      const first = this.map.keys().next().value;
      this.map.delete(first);
    }
  }
  has(key: K) { return this.map.has(key); }
}

export interface MultiTierCacheOptions {
  enableRedis?: boolean;
  enableIndexedDB?: boolean;
  enableLoki?: boolean;
  enablePgVector?: boolean;
  enableMinIO?: boolean;
  lruSize?: number;
  redisNamespace?: string;
  lokiCollection?: string;
  indexedDBName?: string;
  indexedDBStore?: string;
  vectorEndpoint?: string;
  blobEndpoint?: string;
}

export interface CacheGetResult<T> {
  value?: T;
  hit: boolean;
  tier?: string;
}

export type CacheValue = Float32Array | Uint8Array | string | object;

function isBrowser() { return typeof window !== 'undefined'; }
function isServer() { return typeof process !== 'undefined' && !isBrowser(); }

let lokiDb: any; // eslint-disable-line @typescript-eslint/no-explicit-any
let lokiCollection: any;
let indexedDbPromise: Promise<IDBDatabase> | null = null;
let redisClient: any; // eslint-disable-line @typescript-eslint/no-explicit-any

async function initLoki(collectionName: string) {
  if (lokiDb || !collectionName) return;
  try {
    const { default: Loki } = await import('lokijs');
    lokiDb = new Loki('ai-cache.db', { autoload: false });
    lokiCollection = lokiDb.addCollection(collectionName, { indices: ['key'] });
  } catch (e) {
    // optional
  }
}

async function initIndexedDB(dbName: string, store: string) {
  if (!isBrowser()) return null;
  if (indexedDbPromise) return indexedDbPromise;
  indexedDbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(dbName, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(store)) {
        db.createObjectStore(store, { keyPath: 'key' });
      }
    };
    req.onerror = () => reject(req.error);
    req.onsuccess = () => resolve(req.result);
  });
  return indexedDbPromise;
}

async function initRedis(namespace: string | undefined) {
  if (!isServer() || redisClient) return;
  try {
    const { createClient } = await import('redis');
    redisClient = createClient({ url: process.env.REDIS_URL });
    redisClient.on('error', () => {});
    await redisClient.connect();
    if (namespace) await redisClient.select(0);
  } catch (e) {
    redisClient = null;
  }
}

export class MultiTierCache {
  private opts: MultiTierCacheOptions;
  private lru: LRU<string, CacheValue>;
  constructor(opts: MultiTierCacheOptions = {}) {
    this.opts = {
      enableRedis: true,
      enableIndexedDB: true,
      enableLoki: true,
      enablePgVector: false,
      enableMinIO: false,
      lruSize: 256,
      lokiCollection: 'aiArtifacts',
      indexedDBName: 'aiCache',
      indexedDBStore: 'entries',
      ...opts
    };
    this.lru = new LRU(this.opts.lruSize);
    if (this.opts.enableLoki && isServer()) initLoki(this.opts.lokiCollection!);
    if (this.opts.enableIndexedDB && isBrowser()) initIndexedDB(this.opts.indexedDBName!, this.opts.indexedDBStore!);
    if (this.opts.enableRedis) initRedis(this.opts.redisNamespace);
  }

  private serialize(value: CacheValue): string | ArrayBufferView {
    if (value instanceof Float32Array || value instanceof Uint8Array) return value;
    return JSON.stringify(value);
  }
  private deserialize<T>(raw: any): T { // eslint-disable-line @typescript-eslint/no-explicit-any
    if (raw == null) return raw;
    if (raw instanceof Float32Array) return raw as T;
    if (raw instanceof Uint8Array) return raw as T;
    if (typeof raw === 'string') { try { return JSON.parse(raw); } catch { return raw as T; } }
    return raw as T;
  }

  async get<T = CacheValue>(key: string): Promise<CacheGetResult<T>> {
    const lruVal = this.lru.get(key);
    if (lruVal !== undefined) return { value: this.deserialize<T>(lruVal), hit: true, tier: 'lru' };
    if (this.opts.enableLoki && lokiCollection) {
      try {
        const doc = lokiCollection.findOne({ key });
        if (doc) { this.lru.set(key, doc.value); return { value: this.deserialize<T>(doc.value), hit: true, tier: 'loki' }; }
      } catch {}
    }
    if (this.opts.enableRedis && redisClient) {
      try {
        const raw = await redisClient.get(key);
        if (raw) { this.lru.set(key, raw); return { value: this.deserialize<T>(raw), hit: true, tier: 'redis' }; }
      } catch {}
    }
    if (this.opts.enableIndexedDB && isBrowser()) {
      try {
        const db = await initIndexedDB(this.opts.indexedDBName!, this.opts.indexedDBStore!);
        if (db) {
          const tx = db.transaction(this.opts.indexedDBStore!, 'readonly');
          const store = tx.objectStore(this.opts.indexedDBStore!);
          const req = store.get(key);
          const val = await new Promise<any>((resolve) => { // eslint-disable-line @typescript-eslint/no-explicit-any
            req.onsuccess = () => resolve(req.result?.value);
            req.onerror = () => resolve(undefined);
          });
          if (val !== undefined) { this.lru.set(key, val); return { value: this.deserialize<T>(val), hit: true, tier: 'indexeddb' }; }
        }
      } catch {}
    }
    return { hit: false };
  }

  async set(key: string, value: CacheValue, opts: { persistVector?: boolean; persistBlob?: boolean } = {}): Promise<void> {
    const serialized = this.serialize(value);
    this.lru.set(key, serialized as any);
    if (this.opts.enableLoki && lokiCollection) {
      try {
        const existing = lokiCollection.findOne({ key });
        if (existing) { existing.value = serialized; lokiCollection.update(existing); } else lokiCollection.insert({ key, value: serialized, ts: Date.now() });
      } catch {}
    }
    if (this.opts.enableRedis && redisClient) {
      try { await redisClient.set(key, typeof serialized === 'string' ? serialized : JSON.stringify(Array.from((serialized as any) as ArrayLike<number>))); } catch {}
    }
    if (this.opts.enableIndexedDB && isBrowser()) {
      try {
        const db = await initIndexedDB(this.opts.indexedDBName!, this.opts.indexedDBStore!);
        if (db) {
          const tx = db.transaction(this.opts.indexedDBStore!, 'readwrite');
          tx.objectStore(this.opts.indexedDBStore!).put({ key, value: serialized, ts: Date.now() });
        }
      } catch {}
    }
    if (opts.persistVector && this.opts.enablePgVector && value instanceof Float32Array && this.opts.vectorEndpoint) {
      try { fetch(this.opts.vectorEndpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ key, vector: Array.from(value), dim: value.length }) }).catch(()=>{}); } catch {}
    }
    if (opts.persistBlob && this.opts.enableMinIO && this.opts.blobEndpoint) {
      try { fetch(this.opts.blobEndpoint, { method: 'POST', body: serialized instanceof Uint8Array ? serialized : (serialized as any) }).catch(()=>{}); } catch {}
    }
  }

  buildKey(parts: (string | number)[]) { return parts.join(':'); }
}

export const multiTierCache = new MultiTierCache({
  enableRedis: typeof process !== 'undefined' && !!process.env.REDIS_URL,
  enableIndexedDB: true,
  enableLoki: true,
  enablePgVector: false,
  enableMinIO: false,
  lruSize: 512,
  lokiCollection: 'aiArtifacts'
});

export type { MultiTierCacheOptions };
