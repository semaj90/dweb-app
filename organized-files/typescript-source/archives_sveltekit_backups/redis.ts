import { createClient } from 'redis';

let redisClient: unknown = null;

export async function createRedisClient(): Promise<any> {
  if (redisClient) {
    return redisClient;
  }

  const redisUrl = import.meta.env.REDIS_URL || 'redis://localhost:6379';
  
  redisClient = createClient({
    url: redisUrl
  });

  redisClient.on('error', (err: Error) => {
    console.error('Redis Client Error', err);
  });

  await redisClient.connect();
  
  return redisClient;
}

export async function getFromCache(key: string): Promise<string | null> {
  try {
    const client = await createRedisClient();
    return await client.get(key);
  } catch (error: any) {
    console.warn('Redis get error:', error);
    return null;
  }
}

export async function setCache(key: string, value: string, expireInSeconds?: number): Promise<boolean> {
  try {
    const client = await createRedisClient();
    if (expireInSeconds) {
      await client.setEx(key, expireInSeconds, value);
    } else {
      await client.set(key, value);
    }
    return true;
  } catch (error: any) {
    console.warn('Redis set error:', error);
    return false;
  }
}

export async function deleteFromCache(key: string): Promise<boolean> {
  try {
    const client = await createRedisClient();
    await client.del(key);
    return true;
  } catch (error: any) {
    console.warn('Redis delete error:', error);
    return false;
  }
}

export async function closeRedisConnection(): Promise<any> {
  if (redisClient) {
    await redisClient.quit();
    redisClient = null;
  }
}
