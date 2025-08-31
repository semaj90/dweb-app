

import type { RequestHandler } from './$types';

export const GET: RequestHandler = async () => {
  const health = {
    timestamp: new Date().toISOString(),
    services: {},
    overall: 'unknown'
  };

  const checks = [
    // Redis Health Check
    {
      name: 'redis',
      check: async () => {
        try {
          const { createClient } = await import('redis');
          const client = createClient({ url: 'redis://localhost:6379' });
          await client.connect();
          await client.ping();
          await client.quit();
          return { status: 'healthy', responseTime: Date.now() };
        } catch (error: any) {
          return { status: 'unhealthy', error: error.message };
        }
      }
    },
    
    // Qdrant Health Check
    {
      name: 'qdrant',
      check: async () => {
        try {
          const start = Date.now();
          const response = await fetch('http://localhost:6333', {
            signal: AbortSignal.timeout(5000)
          });
          const responseTime = Date.now() - start;
          
          if (response.ok) {
            return { status: 'healthy', responseTime };
          } else {
            return { status: 'unhealthy', error: `HTTP ${response.status}` };
          }
        } catch (error: any) {
          return { status: 'unhealthy', error: error.message };
        }
      }
    },
    
    // Ollama Health Check
    {
      name: 'ollama',
      check: async () => {
        try {
          const start = Date.now();
          const response = await fetch('http://localhost:11434/api/tags', {
            signal: AbortSignal.timeout(5000)
          });
          const responseTime = Date.now() - start;
          
          if (response.ok) {
            const data = await response.json();
            return { 
              status: 'healthy', 
              responseTime,
              models: data.models?.length || 0
            };
          } else {
            return { status: 'unhealthy', error: `HTTP ${response.status}` };
          }
        } catch (error: any) {
          return { status: 'unhealthy', error: error.message };
        }
      }
    },
    
    // SvelteKit App Health Check
    {
      name: 'sveltekit',
      check: async () => {
        try {
          const start = Date.now();
          // Test a simple API endpoint
          const response = await fetch('http://localhost:5173/api/test-crud', {
            signal: AbortSignal.timeout(5000)
          });
          const responseTime = Date.now() - start;
          
          if (response.ok) {
            return { status: 'healthy', responseTime };
          } else {
            return { status: 'unhealthy', error: `HTTP ${response.status}` };
          }
        } catch (error: any) {
          return { status: 'unhealthy', error: error.message };
        }
      }
    },
    
    // Cache Layer Health Check
    {
      name: 'cache_layers',
      check: async () => {
        try {
          const { cacheManager } = await import('$lib/services/cache-layer-manager');
          const stats = cacheManager.getLayerStats();
          
          const enabledLayers = Object.values(stats).filter((layer: any) => layer.enabled).length;
          const avgHitRate = Object.values(stats)
            .filter((layer: any) => layer.enabled)
            .reduce((sum, layer) => sum + layer.hitRate, 0) / enabledLayers;
          
          return {
            status: 'healthy',
            layers: enabledLayers,
            avgHitRate: Math.round(avgHitRate * 100),
            stats
          };
        } catch (error: any) {
          return { status: 'unhealthy', error: error.message };
        }
      }
    }
  ];

  // Run all health checks in parallel
  const results = await Promise.allSettled(
    checks.map(async ({ name, check }) => {
      const result = await check();
      return { name, ...result };
    })
  );

  // Process results
  let healthyCount = 0;
  let totalCount = 0;

  results.forEach((result) => {
    totalCount++;
    if (result.status === 'fulfilled') {
      const service = result.value;
      health.services[service.name] = service;
      if (service.status === 'healthy') {
        healthyCount++;
      }
    } else {
      // Handle rejected promises
      health.services[`unknown_${totalCount}`] = {
        status: 'unhealthy',
        error: result.reason?.message || 'Unknown error'
      };
    }
  });

  // Determine overall health
  if (healthyCount === totalCount) {
    health.overall = 'healthy';
  } else if (healthyCount >= totalCount * 0.7) {
    health.overall = 'degraded';
  } else {
    health.overall = 'unhealthy';
  }

  const statusCode = health.overall === 'healthy' ? 200 : 
                    health.overall === 'degraded' ? 206 : 503;

  return json(health, { status: statusCode });
};