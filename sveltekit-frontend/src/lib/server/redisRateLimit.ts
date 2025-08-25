import Redis from 'ioredis';

export interface RedisRateLimitOptions {
  limit: number;           // max requests per window
  windowSec: number;       // window size seconds
  key: string;             // unique key (user id scoped)
  redis?: Redis;           // optional external client
}

const singleton = { client: null as Redis | null };

function getClient(): Redis {
  if (singleton.client) return singleton.client;
  const url = import.meta.env.REDIS_URL || 'redis://localhost:6379';
  singleton.client = new Redis(url, { lazyConnect: true });
  singleton.client.on('error', (e) => console.error('[redisRateLimit] error', e));
  return singleton.client;
}

/**
 * Sliding window approximation using Lua script (atomic):
 *  - ZADD current timestamp ms
 *  - ZREMRANGEBYSCORE older than window
 *  - ZCARD to count
 *  - Return allowed + ttl
 */
const LUA_SCRIPT = `
local key       = KEYS[1]
local now       = tonumber(ARGV[1])
local window    = tonumber(ARGV[2])
local limit     = tonumber(ARGV[3])
-- remove old
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
-- add current
redis.call('ZADD', key, now, now)
-- count
local count = redis.call('ZCARD', key)
-- set ttl (window) so key expires if idle
redis.call('PEXPIRE', key, window)
local allowed = count <= limit
local retryAfter = 0
if not allowed then
  local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')[2]
  if oldest then
    local diff = (oldest + window) - now
    if diff > 0 then retryAfter = math.floor(diff / 1000) end
  end

return { allowed and 1 or 0, count, retryAfter }
`;

let sha: string | null = null;

export async function redisRateLimit(opts: RedisRateLimitOptions) {
  const client = opts.redis || getClient();
  if (!sha) {
    try {
      // Type cast to satisfy ioredis script overloads
      sha = await (client as any).script('LOAD', LUA_SCRIPT) as string;
    } catch {
      /* ignore */
    }
  }
  const now = Date.now();
  try {
    const res: any = await (client as any).evalsha(
      sha!,
      1,
      `rate:${opts.key}`,
      now,
      opts.windowSec * 1000,
      opts.limit
    );
    const allowed = res[0] === 1;
    const count = res[1];
    const retryAfter = res[2];
    return { allowed, count, retryAfter };
  } catch (e) {
    console.warn('[redisRateLimit] fallback to allowed due to error', e);
    return { allowed: true, count: 1, retryAfter: 0 };
  }
}

export async function closeRedisRateLimit() {
  if (singleton.client) {
    await singleton.client.quit();
    singleton.client = null;
  }
}
