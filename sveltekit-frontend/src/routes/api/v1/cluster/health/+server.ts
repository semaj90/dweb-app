import type { RequestHandler } from './$types.js';
import { json } from '@sveltejs/kit';

/**
 * Cluster Health Monitoring API
 * Real-time health checks for all 37 Go services + external dependencies
 */

import { getRedisService } from '$lib/server/redis/redis-service.js';
import { minioService } from '$lib/server/storage/minio-service.js';
import { rabbitmqService } from '$lib/server/messaging/rabbitmq-service.js';
import { URL } from "url";

export const GET: RequestHandler = async ({ url }) => {
  const includeMetrics = url.searchParams.get('metrics') === 'true';
  
  try {
    // Check health of available services
    const redisService = getRedisService();
    const [redisHealth, minioHealth, rabbitmqHealth] = await Promise.all([
      { status: redisService.isConnectedToRedis() ? 'healthy' : 'unhealthy', details: { connected: redisService.isConnectedToRedis() } },
      minioService.healthCheck(),
      rabbitmqService.healthCheck()
    ]);

    // Check external services
    const externalServices = await checkExternalServices();
    
    // Count healthy services
    const internalServices = { 
      redis: redisHealth.status === 'healthy', 
      minio: minioHealth.status === 'healthy',
      rabbitmq: rabbitmqHealth.status === 'healthy'
    };
    
    const totalHealthy = Object.values(internalServices).filter(Boolean).length + 
                        Object.values(externalServices).filter(Boolean).length;
    const totalServices = Object.keys(internalServices).length + Object.keys(externalServices).length;

    const response = {
      timestamp: new Date().toISOString(),
      status: totalHealthy > 0 ? 'operational' : 'degraded',
      summary: {
        total: totalServices,
        healthy: totalHealthy,
        degraded: totalServices - totalHealthy
      },
      services: {
        internal: internalServices,
        external: externalServices
      },
      ...(includeMetrics && {
        details: {
          redis: redisHealth.details,
          minio: minioHealth.details,
          rabbitmq: rabbitmqHealth.details
        }
      })
    };

    return json(response);
  } catch (error: any) {
    console.error('Health check failed:', error);
    return json(
      {
        error: 'Health check failed',
        message: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date().toISOString(),
        status: 'error'
      },
      { status: 500 }
    );
  }
};

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { action } = await request.json();
    
    switch (action) {
      case 'force_health_check':
        // Force refresh of all service health checks
        const redisService = getRedisService();
        const [redisHealth, minioHealth, rabbitmqHealth] = await Promise.all([
          { status: redisService.isConnectedToRedis() ? 'healthy' : 'unhealthy', details: { connected: redisService.isConnectedToRedis() } },
          minioService.healthCheck(),
          rabbitmqService.healthCheck()
        ]);
        
        return json({
          action: 'force_health_check',
          results: {
            redis: redisHealth.status === 'healthy',
            minio: minioHealth.status === 'healthy',
            rabbitmq: rabbitmqHealth.status === 'healthy'
          },
          timestamp: new Date().toISOString()
        });
        
      default:
        return json({ error: 'Invalid action' }, { status: 400 });
    }
  } catch (error: any) {
    return json({ 
      error: 'Action failed', 
      message: error instanceof Error ? error.message : 'Unknown error' 
    }, { status: 500 });
  }
};

async function checkExternalServices(): Promise<Record<string, boolean>> {
  const externalChecks = [
    { name: 'enhanced_rag', url: 'http://localhost:8095/health' },
    { name: 'upload_service', url: 'http://localhost:8093/health' },
    { name: 'ollama', url: 'http://localhost:11434/api/tags' },
    { name: 'sveltekit', url: 'http://localhost:5173/' }
  ];

  const results: Record<string, boolean> = {};
  
  await Promise.all(
    externalChecks.map(async ({ name, url }) => {
      try {
        const response = await fetch(url, { 
          method: 'GET',
          signal: AbortSignal.timeout(2000)
        });
        results[name] = response.ok || response.status < 500;
      } catch {
        results[name] = false;
      }
    })
  );

  return results;
}