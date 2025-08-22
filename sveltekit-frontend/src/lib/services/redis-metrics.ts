// Lightweight Redis health metrics collector. Tries ioredis first then node-redis if available.
import type { Redis as IORedis } from 'ioredis';
let redis: any = null;
let impl: 'ioredis' | 'redis' | null = null;

export interface RedisHealthMetrics {
  up: number; // 1 if reachable
  last_ping_ms: number; // last successful ping latency
  last_error_ts: number | null;
  last_ok_ts: number | null;
}

const state: RedisHealthMetrics = { up: 0, last_ping_ms: 0, last_error_ts: null, last_ok_ts: null };

async function ensureClient() {
  if (redis) return;
  try {
    const mod = await import('ioredis');
    
    redis = new mod.default(import.meta.env.REDIS_URL || 'redis://localhost:6379');
    impl = 'ioredis';
  } catch {
    try {
      
      const mod2 = await import('redis');
      
      redis = mod2.createClient({ url: import.meta.env.REDIS_URL || 'redis://localhost:6379' });
      await redis.connect();
      impl = 'redis';
    } catch {
      impl = null;
    }
  }
}

function timeout(ms: number){ return new Promise((_r,_j)=> setTimeout(()=>_j(new Error('timeout')), ms)); }

export async function pollRedisHealth(timeoutMs = 500): Promise<RedisHealthMetrics> {
  await ensureClient();
  if (!redis) return state;
  const start = Date.now();
  try {
    let pong: any;
    if (impl === 'ioredis') pong = await Promise.race([(redis as IORedis).ping(), timeout(timeoutMs)]);
    else if (impl === 'redis') pong = await Promise.race([redis.ping(), timeout(timeoutMs)]);
    if (typeof pong === 'string' || pong === true) {
      state.up = 1;
      state.last_ping_ms = Date.now() - start;
      state.last_ok_ts = Date.now();
    }
  } catch {
    state.up = 0;
    state.last_error_ts = Date.now();
  }
  return state;
}

export function getRedisMetrics(): RedisHealthMetrics { return { ...state }; }
