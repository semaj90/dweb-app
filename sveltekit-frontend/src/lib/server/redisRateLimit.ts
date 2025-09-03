import Redis from 'ioredis';
import { getRedisConfig, createServiceConfig, KEY_PATTERNS, LUA_SCRIPTS } from '$lib/config/redis-config';

export interface RedisRateLimitOptions {
  limit: number;           // max requests per window
  windowSec: number;       // window size seconds
  key: string;             // unique key (user id scoped)
  redis?: Redis;           // optional external client
}

const singleton = { client: null as Redis | null };

function getClient(): Redis {
  if (singleton.client) return singleton.client;
  
  // Use centralized Redis configuration for rate limiting
  const config = createServiceConfig('RATE_LIMIT');
  singleton.client = new Redis(config);
  
  singleton.client.on('error', (e: any) => {
    console.error('[redisRateLimit] Redis error:', e.message);
    if (e.message.includes('ECONNREFUSED')) {
      console.error('[redisRateLimit] 💡 Tip: Start Redis with npm run redis:start');
    }
  });
  
  singleton.client.on('connect', () => {
    console.log('[redisRateLimit] ✅ Connected to Redis for rate limiting');
  });
  
  return singleton.client;
}

/**
 * Enhanced Lua script for rate limiting with better performance
 * Uses optimized script from centralized config
 */
let sha: string | null = null;

// Preload the Lua script for better performance
async function ensureScriptLoaded(client: Redis): Promise<string> {
  if (!sha) {
    try {
      sha = await client.script('LOAD', LUA_SCRIPTS.RATE_LIMIT) as string;
      console.log('[redisRateLimit] ✅ Lua script loaded with SHA:', sha.substring(0, 8) + '...');
    } catch (error) {
      console.error('[redisRateLimit] ❌ Failed to load Lua script:', error);
      throw error;
    }
  }
  return sha;
}

export async function redisRateLimit(opts: RedisRateLimitOptions): Promise<{
  allowed: boolean;
  count: number;
  retryAfter: number;
}> {
  const client = opts.redis || getClient();
  
  try {
    // Ensure Lua script is loaded
    const scriptSha = await ensureScriptLoaded(client);
    
    const now = Date.now();
    const key = KEY_PATTERNS.RATE_LIMIT(opts.key);
    
    const res: any = await client.evalsha(
      scriptSha,
      1,
      key,
      now,
      opts.windowSec * 1000,
      opts.limit
    );
    
    const allowed = res[0] === 1;
    const count = res[1];
    const retryAfter = res[2];
    
    // Log rate limit activity (development only)
    if (process.env.NODE_ENV === 'development') {
      console.log(`[redisRateLimit] ${opts.key}: ${count}/${opts.limit} requests in ${opts.windowSec}s window, allowed: ${allowed}`);
    }
    
    return { allowed, count, retryAfter };
  } catch (e: any) {
    console.error('[redisRateLimit] ❌ Rate limit check failed:', e.message);
    
    // Graceful degradation - allow request but log error
    if (e.message.includes('NOSCRIPT')) {
      console.log('[redisRateLimit] 🔄 Script not found, reloading...');
      sha = null; // Reset SHA to force reload
      return redisRateLimit(opts); // Retry once
    }
    
    console.warn('[redisRateLimit] ⚠️ Falling back to allowing request due to Redis error');
    return { allowed: true, count: 1, retryAfter: 0 };
  }
}

export async function closeRedisRateLimit(): Promise<any> {
  if (singleton.client) {
    await (singleton.client as any).quit?.();
    singleton.client = null;
  }
}
