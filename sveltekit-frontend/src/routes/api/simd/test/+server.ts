import crypto from "crypto";
import type { RequestHandler } from './$types';


// SIMD + Redis Performance Testing API
import { simdRedisClient } from "$lib/services/simd-redis-client";

// Generate test data of various sizes
function generateTestData(size: 'small' | 'medium' | 'large' | 'xlarge') {
  const baseObj = {
    id: crypto.randomUUID(),
    timestamp: new Date().toISOString(),
    metadata: {
      version: '1.0',
      processed: false,
      tags: ['legal', 'document', 'test']
    }
  };

  switch (size) {
    case 'small':
      return { ...baseObj, data: 'x'.repeat(100) }; // ~200 bytes
    case 'medium':
      return { ...baseObj, data: 'x'.repeat(5000) }; // ~5KB
    case 'large':
      return { ...baseObj, data: 'x'.repeat(50000), chunks: Array(100).fill(baseObj) }; // ~50KB
    case 'xlarge':
      return { ...baseObj, data: 'x'.repeat(500000), chunks: Array(1000).fill(baseObj) }; // ~500KB
    default:
      return baseObj;
  }
}

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { test_type, iterations = 100 } = await request.json();
    const cacheService = comprehensiveCachingService;
    
    const results: any = {
      test_type,
      iterations,
      timestamp: new Date().toISOString(),
      results: {}
    };

    switch (test_type) {
      case 'simd_health': {
        try {
          const health = await simdRedisClient.healthCheck();
          results.results = {
            simd_available: true,
            health,
            message: 'SIMD service is operational'
          };
        } catch (error: any) {
          results.results = {
            simd_available: false,
            error: String(error),
            message: 'SIMD service unavailable - start with: cd go-microservice && go run simd-server.go'
          };
        }
        break;
      }

      case 'json_parsing_benchmark': {
        const testSizes = ['small', 'medium', 'large'] as const;
        const benchmarkResults: any = {};

        for (const size of testSizes) {
          const testData = generateTestData(size);
          
          // SIMD benchmark
          try {
            const simdResult = await simdRedisClient.benchmark(testData, Math.min(iterations, 100));
            benchmarkResults[size] = simdResult;
          } catch (error: any) {
            benchmarkResults[size] = {
              error: String(error),
              message: 'SIMD benchmark failed - service may be unavailable'
            };
          }
        }

        results.results = benchmarkResults;
        break;
      }

      case 'cache_performance': {
        const testData = generateTestData('large');
        const cacheKey = `test:${Date.now()}`;

        // Standard cache performance
        const standardStart = performance.now();
        await cacheService.set(cacheKey, testData);
        const cachedData = await cacheService.get(cacheKey);
        const standardTime = performance.now() - standardStart;

        // SIMD cache performance (if available)
        let simdTime = null;
        try {
          const simdStart = performance.now();
          await cacheService.setSIMD(`simd:${cacheKey}`, testData);
          const simdCachedData = await cacheService.getSIMD(`simd:${cacheKey}`);
          simdTime = performance.now() - simdStart;
        } catch (error: any) {
          console.warn('SIMD cache test failed:', error);
        }

        results.results = {
          data_size: JSON.stringify(testData).length,
          standard_cache_ms: standardTime,
          simd_cache_ms: simdTime,
          speedup_factor: simdTime ? standardTime / simdTime : null,
          cache_hit: cachedData !== null
        };
        break;
      }

      case 'redis_json_operations': {
        const testData = generateTestData('medium');
        const operations: any[] = [];

        try {
          // Test Redis JSON caching
          const cacheStart = performance.now();
          const cacheResult = await simdRedisClient.cacheJSON('test:redis-json', testData);
          const cacheTime = performance.now() - cacheStart;

          operations.push({
            operation: 'redis_json_set',
            time_ms: cacheTime,
            success: cacheResult.cached,
            data_size: JSON.stringify(testData).length
          });

          // Test SIMD parsing
          const parseStart = performance.now();
          const parseResult = await simdRedisClient.parseJSON(testData);
          const parseTime = performance.now() - parseStart;

          operations.push({
            operation: 'simd_json_parse',
            time_ms: parseTime,
            parse_time: parseResult.parse_time,
            fields_parsed: parseResult.fields,
            data_size: parseResult.size
          });

        } catch (error: any) {
          operations.push({
            operation: 'error',
            error: String(error),
            message: 'Redis/SIMD operations failed'
          });
        }

        results.results = { operations };
        break;
      }

      default:
        return json(
          { error: 'Invalid test_type. Use: simd_health, json_parsing_benchmark, cache_performance, redis_json_operations' },
          { status: 400 }
        );
    }

    return json(results);

  } catch (error: any) {
    console.error('SIMD test API error:', error);
    return json(
      { error: 'Test execution failed', details: String(error) },
      { status: 500 }
    );
  }
};

export const GET: RequestHandler = async () => {
  return json({
    message: 'SIMD + Redis Performance Testing API',
    available_tests: [
      'simd_health - Check Go microservice status',
      'json_parsing_benchmark - Compare SIMD vs standard JSON parsing',
      'cache_performance - Compare standard vs SIMD cache operations', 
      'redis_json_operations - Test Redis JSON and SIMD parsing'
    ],
    usage: {
      method: 'POST',
      body: {
        test_type: 'simd_health | json_parsing_benchmark | cache_performance | redis_json_operations',
        iterations: 100
      }
    },
    go_service: {
      start_command: 'cd go-microservice && go run simd-server.go',
      endpoints: [
        'http://localhost:8080/health',
        'http://localhost:8080/simd-parse',
        'http://localhost:8080/redis-json'
      ]
    }
  });
};