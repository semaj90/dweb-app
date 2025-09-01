import { json } from "@sveltejs/kit";
import { qdrantOptimized } from "$lib/server/vector/qdrant-optimized";
import { redisRateLimit, createRateLimitConfig } from "$lib/server/redisRateLimit";
import logger from '$lib/server/production-logger';
import { dev } from '$app/environment';
import type { RequestHandler } from './$types.js';
import { URL } from "url";


/**
 * Optimized Qdrant API Endpoints
 * Memory-efficient vector operations with cache-like logging system
 * 
 * GET /api/qdrant/optimized?action=search&query=...     - Optimized vector search with caching
 * GET /api/qdrant/optimized?action=metrics              - Performance and memory metrics
 * GET /api/qdrant/optimized?action=health               - Optimized health check
 * POST /api/qdrant/optimized { action: "batch_upsert" } - Memory-efficient batch operations
 * POST /api/qdrant/optimized { action: "clear_cache" }  - Cache management
 */

// Enhanced interfaces for optimized operations
export interface OptimizedSearchRequest {
  collection: string;
  query: string | number[];
  limit?: number;
  offset?: number;
  threshold?: number;
  useCache?: boolean;
  filter?: Record<string, any>;
}

export interface BatchUpsertRequest {
  collection: string;
  points: Array<{
    id: string;
    vector: number[];
    payload: any;
  }>;
}

export interface CacheManagementRequest {
  action: 'clear_cache' | 'get_stats' | 'optimize_memory';
  cacheType?: 'vector' | 'search' | 'query_history' | 'all';
}

export const GET: RequestHandler = async ({ url, locals, getClientAddress }) => {
  const clientIP = getClientAddress();
  const action = url.searchParams.get('action') || 'health';

  try {
    // Enhanced rate limiting for GET requests based on user role
    const isAuthenticated = !!locals.user;
    const rateLimitResult = await redisRateLimit({
      key: `qdrant_optimized_get:${clientIP}:${locals.user?.id || 'anonymous'}`,
      limit: isAuthenticated ? 200 : 100, // Higher limit for authenticated users
      windowSec: 60
    });

    if (!rateLimitResult.allowed) {
      return json({
        success: false,
        error: 'Rate limit exceeded',
        retryAfter: rateLimitResult.retryAfter
      }, { 
        status: 429,
        headers: {
          'Retry-After': rateLimitResult.retryAfter.toString()
        }
      });
    }

    switch (action) {
      case 'health': {
        const startTime = Date.now();
        const isHealthy = await qdrantOptimized.isHealthy();
        const memoryUsage = qdrantOptimized.getMemoryUsage();
        const performanceMetrics = qdrantOptimized.getPerformanceMetrics();

        return json({
          success: true,
          data: {
            status: isHealthy ? 'healthy' : 'unhealthy',
            processingTime: Date.now() - startTime,
            optimization: {
              memoryUsage,
              performanceMetrics,
              cacheEfficiency: performanceMetrics.totalQueries > 0 
                ? (performanceMetrics.cacheHits / performanceMetrics.totalQueries * 100).toFixed(1) + '%'
                : 'N/A',
              platform: process.platform,
              memoryBudget: process.platform === 'win32' ? '4MB' : '2MB'
            }
          },
          meta: {
            endpoint: 'optimized_health',
            timestamp: new Date().toISOString(),
            rateLimit: {
              remaining: rateLimitResult.remaining,
              resetTime: rateLimitResult.resetTime
            }
          }
        });
      }

      case 'metrics': {
        const memoryUsage = qdrantOptimized.getMemoryUsage();
        const performanceMetrics = qdrantOptimized.getPerformanceMetrics();
        const queryHistory = qdrantOptimized.getQueryHistory();

        return json({
          success: true,
          data: {
            memory: {
              usage: memoryUsage,
              efficiency: (memoryUsage.total / (process.platform === 'win32' ? 4096 : 2048) * 100).toFixed(1) + '%',
              budget: process.platform === 'win32' ? '4MB' : '2MB'
            },
            performance: {
              ...performanceMetrics,
              cacheHitRatio: performanceMetrics.totalQueries > 0 
                ? (performanceMetrics.cacheHits / performanceMetrics.totalQueries * 100).toFixed(1) + '%'
                : 'N/A'
            },
            queryHistory: {
              totalQueries: queryHistory.length,
              recentQueries: queryHistory.slice(-5).map(q => ({
                timestamp: q.timestamp,
                operation: q.operation,
                collection: q.collection,
                processingTime: q.processingTime,
                cacheHit: q.cacheHit,
                resultCount: q.resultCount
              }))
            },
            system: {
              platform: process.platform,
              arch: process.arch,
              nodeVersion: process.version,
              windowsOptimized: process.platform === 'win32'
            }
          },
          meta: {
            endpoint: 'optimized_metrics',
            timestamp: new Date().toISOString(),
            rateLimit: {
              remaining: rateLimitResult.remaining,
              resetTime: rateLimitResult.resetTime
            }
          }
        });
      }

      case 'search': {
        const collection = url.searchParams.get('collection') || 'legal_documents';
        const query = url.searchParams.get('query');
        const limit = Math.min(parseInt(url.searchParams.get('limit') || '10'), 50);
        const offset = parseInt(url.searchParams.get('offset') || '0');
        const threshold = parseFloat(url.searchParams.get('threshold') || '0.7');
        const useCache = url.searchParams.get('cache') !== 'false';

        if (!query) {
          return json({
            success: false,
            error: 'Query parameter required for search'
          }, { status: 400 });
        }

        const startTime = Date.now();

        try {
          // Parse query as vector or use as text
          let queryInput: string | number[];
          try {
            const vectorQuery = JSON.parse(query);
            if (Array.isArray(vectorQuery) && vectorQuery.every(n => typeof n === 'number')) {
              queryInput = vectorQuery;
            } else {
              queryInput = query;
            }
          } catch {
            queryInput = query;
          }

          const results = await qdrantOptimized.search(collection, queryInput, {
            limit,
            offset,
            threshold,
            useCache
          });

          const processingTime = Date.now() - startTime;
          const memoryUsage = qdrantOptimized.getMemoryUsage();
          const performanceMetrics = qdrantOptimized.getPerformanceMetrics();

          logger.info('Optimized vector search completed', {
            component: 'QdrantOptimizedAPI',
            service: 'qdrant'
          }, {
            collection,
            queryType: typeof queryInput,
            resultsCount: results.length,
            processingTime,
            cacheHit: performanceMetrics.cacheHits > 0
          });

          return json({
            success: true,
            data: {
              collection,
              query: {
                type: typeof queryInput,
                limit,
                offset,
                threshold,
                useCache
              },
              results,
              optimization: {
                processingTime,
                memoryUsage,
                cacheHit: useCache && performanceMetrics.cacheHits > 0,
                cacheEfficiency: performanceMetrics.totalQueries > 0 
                  ? (performanceMetrics.cacheHits / performanceMetrics.totalQueries * 100).toFixed(1) + '%'
                  : 'N/A'
              }
            },
            meta: {
              endpoint: 'optimized_search',
              timestamp: new Date().toISOString(),
              rateLimit: {
                remaining: rateLimitResult.remaining,
                resetTime: rateLimitResult.resetTime
              }
            }
          });

        } catch (searchError) {
          logger.error('Optimized search failed', searchError instanceof Error ? searchError : undefined, {
            component: 'QdrantOptimizedAPI',
            service: 'qdrant'
          }, {
            collection,
            queryType: typeof query,
            processingTime: Date.now() - startTime
          });
          throw searchError;
        }
      }

      case 'cache_stats': {
        const memoryUsage = qdrantOptimized.getMemoryUsage();
        const performanceMetrics = qdrantOptimized.getPerformanceMetrics();
        
        return json({
          success: true,
          data: {
            cacheStats: {
              memory: memoryUsage,
              performance: performanceMetrics,
              efficiency: {
                hitRatio: performanceMetrics.totalQueries > 0 
                  ? (performanceMetrics.cacheHits / performanceMetrics.totalQueries * 100).toFixed(1) + '%'
                  : 'N/A',
                memoryEfficiency: (memoryUsage.total / (process.platform === 'win32' ? 4096 : 2048) * 100).toFixed(1) + '%'
              }
            }
          }
        });
      }

      default:
        return json({
          success: false,
          error: 'Invalid action',
          availableActions: ['health', 'metrics', 'search', 'cache_stats']
        }, { status: 400 });
    }

  } catch (error: any) {
    logger.error('Optimized Qdrant GET operation failed', error instanceof Error ? error : undefined, {
      component: 'QdrantOptimizedAPI',
      service: 'qdrant'
    }, {
      action,
      clientIP
    });

    return json({
      success: false,
      error: 'Internal server error',
      details: dev ? (error instanceof Error ? error.message : 'Unknown error') : undefined
    }, { status: 500 });
  }
};

export const POST: RequestHandler = async ({ request, locals, getClientAddress }) => {
  const clientIP = getClientAddress();
  let body: any = null;

  try {
    // Parse body early for error handling
    body = await request.json();

    // Enhanced rate limiting based on user role
    const rateLimitConfig = createRateLimitConfig(locals.user?.role === 'admin' ? 'admin' : 'api');
    const rateLimitResult = await redisRateLimit({
      key: `qdrant_optimized_post:${clientIP}:${locals.user?.id || 'anonymous'}`,
      ...rateLimitConfig
    });

    if (!rateLimitResult.allowed) {
      return json({
        success: false,
        error: 'Rate limit exceeded',
        retryAfter: rateLimitResult.retryAfter
      }, { 
        status: 429,
        headers: {
          'Retry-After': rateLimitResult.retryAfter.toString()
        }
      });
    }

    const { action } = body;

    if (!action) {
      return json({
        success: false,
        error: 'Action parameter required'
      }, { status: 400 });
    }

    switch (action) {
      case 'batch_upsert': {
        // Admin only for batch operations  
        if (!locals.user || locals.user.role !== 'admin') {
          return json({
            success: false,
            error: 'Admin privileges required for batch operations'
          }, { status: 403 });
        }

        const { collection, points }: BatchUpsertRequest = body;
        
        if (!collection || !points || !Array.isArray(points)) {
          return json({
            success: false,
            error: 'Collection and points array required'
          }, { status: 400 });
        }

        const startTime = Date.now();
        
        try {
          await qdrantOptimized.upsertBatch(collection, points);
          
          const processingTime = Date.now() - startTime;
          const memoryUsage = qdrantOptimized.getMemoryUsage();

          logger.info('Batch upsert completed', {
            component: 'QdrantOptimizedAPI',
            service: 'qdrant'
          }, {
            collection,
            pointsCount: points.length,
            processingTime,
            memoryUsage
          });

          return json({
            success: true,
            data: {
              message: 'Batch upsert completed successfully',
              collection,
              pointsUpserted: points.length,
              processingTime,
              optimization: {
                memoryUsage,
                batchOptimized: true
              },
              rateLimit: {
                remaining: rateLimitResult.remaining,
                resetTime: rateLimitResult.resetTime
              }
            }
          });

        } catch (upsertError) {
          logger.error('Batch upsert failed', upsertError instanceof Error ? upsertError : undefined, {
            component: 'QdrantOptimizedAPI',
            service: 'qdrant'
          }, {
            collection,
            pointsCount: points.length
          });
          throw upsertError;
        }
      }

      case 'clear_cache': {
        // Admin only for cache management
        if (!locals.user || locals.user.role !== 'admin') {
          return json({
            success: false,
            error: 'Admin privileges required for cache operations'
          }, { status: 403 });
        }

        const { cacheType = 'all' }: CacheManagementRequest = body;
        const beforeMemory = qdrantOptimized.getMemoryUsage();
        
        // Clear caches
        qdrantOptimized.clearCaches();
        
        const afterMemory = qdrantOptimized.getMemoryUsage();
        
        logger.info('Cache cleared', {
          component: 'QdrantOptimizedAPI',
          service: 'qdrant'
        }, {
          cacheType,
          memoryFreed: beforeMemory.total - afterMemory.total
        });

        return json({
          success: true,
          data: {
            message: 'Cache cleared successfully',
            cacheType,
            memoryFreed: {
              before: beforeMemory,
              after: afterMemory,
              freedKB: beforeMemory.total - afterMemory.total
            }
          }
        });
      }

      case 'optimize_memory': {
        // Admin only for memory optimization
        if (!locals.user || locals.user.role !== 'admin') {
          return json({
            success: false,
            error: 'Admin privileges required for memory optimization'
          }, { status: 403 });
        }

        const beforeMemory = qdrantOptimized.getMemoryUsage();
        
        // Clear caches and trigger garbage collection
        qdrantOptimized.clearCaches();
        
        if (global.gc) {
          global.gc();
        }
        
        const afterMemory = qdrantOptimized.getMemoryUsage();
        
        logger.info('Memory optimization completed', {
          component: 'QdrantOptimizedAPI',
          service: 'qdrant'
        }, {
          memoryOptimization: {
            before: beforeMemory,
            after: afterMemory,
            freed: beforeMemory.total - afterMemory.total
          }
        });

        return json({
          success: true,
          data: {
            message: 'Memory optimization completed',
            optimization: {
              before: beforeMemory,
              after: afterMemory,
              memoryFreed: beforeMemory.total - afterMemory.total,
              efficiencyGain: beforeMemory.total > 0 
                ? ((beforeMemory.total - afterMemory.total) / beforeMemory.total * 100).toFixed(1) + '%'
                : '0%'
            }
          }
        });
      }

      default:
        return json({
          success: false,
          error: 'Invalid action',
          availableActions: ['batch_upsert', 'clear_cache', 'optimize_memory']
        }, { status: 400 });
    }

  } catch (error: any) {
    logger.error('Optimized Qdrant POST operation failed', error instanceof Error ? error : undefined, {
      component: 'QdrantOptimizedAPI',
      service: 'qdrant'
    }, {
      clientIP,
      action: body?.action || 'unknown'
    });

    return json({
      success: false,
      error: 'Internal server error',
      details: dev ? (error instanceof Error ? error.message : 'Unknown error') : undefined
    }, { status: 500 });
  }
};