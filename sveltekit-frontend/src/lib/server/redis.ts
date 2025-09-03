
/**
 * Centralized Redis Client for Legal AI Platform
 * Follows the integration guide pattern with enhanced error handling
 */

import Redis from 'ioredis';
import { getRedisConfig } from '$lib/config/redis-config';

// Get optimized Redis configuration
const url = process.env.REDIS_URL ?? 'redis://127.0.0.1:6379';
const config = getRedisConfig();

// Create Redis client with optimized settings
export const redis = new Redis(url, {
  ...config,
  maxRetriesPerRequest: null, // Important for pub/sub
  lazyConnect: false, // Connect immediately
});

// Legacy IORedis connection for backward compatibility
export const REDIS_CONNECTION = redis;

// Enhanced event handlers
redis.on('connect', () => {
  console.log('[redis] ✅ Connected successfully');
});

redis.on('ready', () => {
  console.log('[redis] 🚀 Client ready for operations');
});

redis.on('error', (error) => {
  console.error('[redis] ❌ Connection error:', error.message);
  
  // Provide helpful error messages
  if (error.message.includes('ECONNREFUSED')) {
    console.error('[redis] 💡 Tip: Start Redis server with: npm run redis:start');
  } else if (error.message.includes('NOAUTH')) {
    console.error('[redis] 💡 Tip: Check REDIS_PASSWORD environment variable');
  }
});

redis.on('reconnecting', (delay) => {
  console.log(`[redis] 🔄 Reconnecting in ${delay}ms...`);
});

redis.on('close', () => {
  console.log('[redis] 🔌 Connection closed');
});

// Redis client factory for backward compatibility
export const createRedisInstance = () => {
  return new Redis(config);
};

// Legacy client factory (now uses IORedis for consistency)
let legacyClient: Redis | null = null;

export async function createRedisClient(): Promise<Redis> {
  if (legacyClient) {
    return legacyClient;
  }
  
  legacyClient = redis; // Use the main client
  return legacyClient;
}
export async function getFromCache(key: string): Promise<string | null> {
  try {
    const client = await createRedisClient();
    return await client.get(key);
  } catch (error: any) {
    console.warn("Redis get error:", error);
    return null;
  }
}
export async function setCache(
  key: string,
  value: string,
  expireInSeconds?: number,
): Promise<boolean> {
  try {
    const client = await createRedisClient();
    if (expireInSeconds) {
      await client.setEx(key, expireInSeconds, value);
    } else {
      await client.set(key, value);
    }
    return true;
  } catch (error: any) {
    console.warn("Redis set error:", error);
    return false;
  }
}
export async function deleteFromCache(key: string): Promise<boolean> {
  try {
    const client = await createRedisClient();
    await client.del(key);
    return true;
  } catch (error: any) {
    console.warn("Redis delete error:", error);
    return false;
  }
}
export async function closeRedisConnection(): Promise<any> {
  if (redisClient) {
    await redisClient.quit();
    redisClient = null;
  }
}
