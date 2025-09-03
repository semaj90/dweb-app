
import type { RequestHandler } from './$types';
import { json } from '@sveltejs/kit';
import { z } from 'zod';

// SIMD JSON Processing API - SvelteKit Integration with Go Microservice
import { simdRedisClient } from "$lib/services/simd-redis-client";

const SIMDProcessSchema = z.object({
  data: z.any(),
  operation: z.enum(['parse', 'cache', 'benchmark']).default('parse'),
  cache_key: z.string().optional(),
  iterations: z.number().min(1).max(10000).default(100).optional(),
});

// SIMD JSON Processing
export const POST: RequestHandler = async ({ request }) => {
  try {
    const body = await request.json();
    const parsed = SIMDProcessSchema.safeParse(body);

    if (!parsed.success) {
      return json(
        { error: 'Invalid request', details: parsed.error.errors },
        { status: 400 }
      );
    }

    const { data, operation, cache_key, iterations } = parsed.data;

    // Check if Go SIMD service is available
    try {
      await simdRedisClient.healthCheck();
    } catch (healthError) {
      return json(
        { 
          error: 'SIMD service unavailable', 
          message: 'Go microservice not running on localhost:8080',
          fallback: 'Using standard JSON processing',
          details: String(healthError)
        },
        { status: 503 }
      );
    }

    switch (operation) {
      case 'parse': {
        const result = await simdRedisClient.parseJSON(data);
        return json({
          success: true,
          operation: 'simd_parse',
          result,
          metadata: {
            service: 'go-simd-microservice',
            timestamp: new Date().toISOString()
          }
        });
      }

      case 'cache': {
        if (!cache_key) {
          return json(
            { error: 'cache_key required for cache operation' },
            { status: 400 }
          );
        }
        
        const result = await simdRedisClient.cacheJSON(cache_key, data);
        return json({
          success: true,
          operation: 'redis_cache',
          result,
          metadata: {
            service: 'go-simd-microservice',
            key: cache_key,
            timestamp: new Date().toISOString()
          }
        });
      }

      case 'benchmark': {
        const result = await simdRedisClient.benchmark(data, iterations);
        return json({
          success: true,
          operation: 'simd_benchmark',
          result,
          metadata: {
            service: 'go-simd-microservice',
            iterations: iterations || 100,
            timestamp: new Date().toISOString()
          }
        });
      }

      default:
        return json(
          { error: 'Invalid operation' },
          { status: 400 }
        );
    }

  } catch (error: any) {
    console.error('SIMD API error:', error);
    
    return json(
      { 
        error: 'SIMD processing failed',
        message: String(error),
        fallback_available: true
      },
      { status: 500 }
    );
  }
};

// Get SIMD service status
export const GET: RequestHandler = async () => {
  try {
    const health = await simdRedisClient.healthCheck();
    
    return json({
      success: true,
      service: 'simd-redis-microservice',
      status: health,
      endpoints: {
        'POST /api/simd': 'SIMD JSON processing',
        'GET /api/simd': 'Service health check'
      },
      performance_features: [
        'SIMD-optimized JSON parsing',
        'Redis JSON caching',
        'Sub-millisecond processing',
        'Goroutine worker pools',
        'Automatic fallback handling'
      ],
      timestamp: new Date().toISOString()
    });

  } catch (error: any) {
    return json(
      {
        success: false,
        service: 'simd-redis-microservice',
        status: 'unavailable',
        error: String(error),
        message: 'Go microservice not running. Start with: cd go-microservice && go run simd-server.go'
      },
      { status: 503 }
    );
  }
};