import { getRedisService } from '$lib/server/redis/redis-service';
import type { RequestHandler } from './$types';


export const GET: RequestHandler = async () => {
  try {
    const redisService = getRedisService();
    
    const metrics = {
      connected: redisService.isConnectedToRedis(),
      status: redisService.isConnectedToRedis() ? 'healthy' : 'disconnected',
      timestamp: new Date().toISOString()
    };

    return new Response(JSON.stringify({ redis: metrics }), { 
      headers: { 'Content-Type': 'application/json' } 
    });
  } catch (error) {
    return new Response(JSON.stringify({ 
      redis: { 
        connected: false, 
        status: 'error', 
        error: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date().toISOString()
      } 
    }), { 
      status: 500,
      headers: { 'Content-Type': 'application/json' } 
    });
  }
};
