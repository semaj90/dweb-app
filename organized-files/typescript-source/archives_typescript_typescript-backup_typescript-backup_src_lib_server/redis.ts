import { createClient } from 'redis';
import { env } from '$env/dynamic/private';

// Default Redis configuration
const redisConfig = {
  host: env.REDIS_HOST || 'localhost',
  port: parseInt(env.REDIS_PORT || '6379'),
  password: env.REDIS_PASSWORD,
  db: parseInt(env.REDIS_DB || '0'),
};

// Create Redis client
export async function createRedisClient(): Promise<any> {
  const client = createClient({
    url: `redis://${redisConfig.host}:${redisConfig.port}`,
    password: redisConfig.password,
    database: redisConfig.db,
  });

  client.on('error', (err) => {
    console.error('Redis Client Error:', err);
  });

  if (!client.isOpen) {
    await client.connect();
  }

  return client;
}

// Singleton instance
let globalRedisClient: Awaited<ReturnType<typeof createRedisClient>> | null = null;

export async function getRedisClient(): Promise<any> {
  if (!globalRedisClient) {
    globalRedisClient = await createRedisClient();
  }
  return globalRedisClient;
}

export default { createRedisClient, getRedisClient };