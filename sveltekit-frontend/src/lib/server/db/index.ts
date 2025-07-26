import { db, sql } from './drizzle';

export { db, sql };

// Re-export performance optimizations
export { OptimizedQueries, CacheService } from '$lib/performance/optimizations';

// Database connection health check
export async function healthCheck() {
  try {
    await db.execute(sql`SELECT 1`);
    return { status: 'healthy', timestamp: new Date() };
  } catch (error) {
    return { status: 'unhealthy', error: error.message, timestamp: new Date() };
  }
}