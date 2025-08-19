import Redis from 'ioredis';

export function createRedisConnection() {
  return new Redis({
    host: import.meta.env.REDIS_HOST || 'localhost',
    port: parseInt(import.meta.env.REDIS_PORT || '6379'),
    enableReadyCheck: true,
    maxRetriesPerRequest: null,
    // retryDelayOnFailover removed - not a valid ioredis option
  });
}
