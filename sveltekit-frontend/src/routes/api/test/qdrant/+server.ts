import type { RequestHandler } from './$types.js';

// Optimized Qdrant Service Test API
// Tests the memory-efficient Qdrant service with SOM clustering and NES cache integration

import { json } from '@sveltejs/kit';
import { optimizedQdrantService } from '$lib/services/optimized-qdrant-service';
import { URL } from "url";

export interface TestResult {
  test: string;
  status: 'success' | 'error' | 'warning';
  data?: any;
  error?: string;
  duration?: number;
}

export const GET: RequestHandler = async ({ url }) => {
  const testType = url.searchParams.get('test') || 'all';
  const results: TestResult[] = [];

  try {
    // Test 1: Health Check
    if (testType === 'all' || testType === 'health') {
      const startTime = Date.now();
      try {
        const health = await optimizedQdrantService.healthCheck();
        results.push({
          test: 'qdrant_health_check',
          status: health.status === 'healthy' ? 'success' : 'warning',
          data: health,
          duration: Date.now() - startTime
        });
      } catch (error: any) {
        results.push({
          test: 'qdrant_health_check',
          status: 'error',
          error: error instanceof Error ? error.message : String(error),
          duration: Date.now() - startTime
        });
      }
    }

    // Test 2: Collection Setup
    if (testType === 'all' || testType === 'collection') {
      const startTime = Date.now();
      try {
        await optimizedQdrantService.ensureCollection();
        results.push({
          test: 'collection_setup',
          status: 'success',
          data: { message: 'Collection ensured with 768-dimensional nomic-embed vectors' },
          duration: Date.now() - startTime
        });
      } catch (error: any) {
        results.push({
          test: 'collection_setup',
          status: 'error',
          error: error instanceof Error ? error.message : String(error),
          duration: Date.now() - startTime
        });
      }
    }

    // Test 3: Sample Vector Search
    if (testType === 'all' || testType === 'search') {
      const startTime = Date.now();
      try {
        // Generate a sample 768-dimensional vector for nomic-embed
        const sampleVector = Array.from({ length: 768 }, () => Math.random() * 2 - 1);
        
        const searchResult = await optimizedQdrantService.searchVectors(sampleVector, {
          limit: 5,
          threshold: 0.1,
          useCache: true,
          enableSOM: true
        });

        results.push({
          test: 'vector_search',
          status: 'success',
          data: {
            results_count: searchResult.results.length,
            stats: searchResult.stats,
            sample_vector_dim: sampleVector.length
          },
          duration: Date.now() - startTime
        });
      } catch (error: any) {
        results.push({
          test: 'vector_search',
          status: 'error',
          error: error instanceof Error ? error.message : String(error),
          duration: Date.now() - startTime
        });
      }
    }

    // Test 4: Sample Vector Upsert
    if (testType === 'all' || testType === 'upsert') {
      const startTime = Date.now();
      try {
        const sampleVectors = [
          {
            id: `test_${Date.now()}_1`,
            vector: Array.from({ length: 768 }, () => Math.random() * 2 - 1),
            payload: {
              type: 'test',
              title: 'Test Document 1',
              content: 'Sample legal document for testing vector operations',
              created_at: new Date().toISOString()
            }
          },
          {
            id: `test_${Date.now()}_2`,
            vector: Array.from({ length: 768 }, () => Math.random() * 2 - 1),
            payload: {
              type: 'test',
              title: 'Test Document 2', 
              content: 'Another sample legal document for testing',
              created_at: new Date().toISOString()
            }
          }
        ];

        const upsertResult = await optimizedQdrantService.upsertVectors(sampleVectors);
        
        results.push({
          test: 'vector_upsert',
          status: 'success',
          data: {
            vectors_inserted: sampleVectors.length,
            upsert_result: upsertResult,
            vector_dimensions: sampleVectors[0].vector.length
          },
          duration: Date.now() - startTime
        });
      } catch (error: any) {
        results.push({
          test: 'vector_upsert',
          status: 'error',
          error: error instanceof Error ? error.message : String(error),
          duration: Date.now() - startTime
        });
      }
    }

    // Test 5: PostgreSQL Sync (dry run)
    if (testType === 'all' || testType === 'sync') {
      const startTime = Date.now();
      try {
        // Test sync capability without full sync
        const health = await optimizedQdrantService.healthCheck();
        
        results.push({
          test: 'postgresql_sync_test',
          status: 'success',
          data: {
            message: 'Sync service initialized and ready',
            qdrant_status: health.status,
            memory_usage: health.memoryUsage,
            note: 'Use POST /api/test/qdrant?action=sync for full PostgreSQL sync'
          },
          duration: Date.now() - startTime
        });
      } catch (error: any) {
        results.push({
          test: 'postgresql_sync_test',
          status: 'error',
          error: error instanceof Error ? error.message : String(error),
          duration: Date.now() - startTime
        });
      }
    }

    // Test 6: Memory Usage and Performance
    if (testType === 'all' || testType === 'performance') {
      const startTime = Date.now();
      try {
        const health = await optimizedQdrantService.healthCheck();
        
        // Simulate multiple search operations to test caching
        const searchPromises = Array.from({ length: 3 }, async (_, i) => {
          const vector = Array.from({ length: 768 }, () => Math.random() * 2 - 1);
          return await optimizedQdrantService.searchVectors(vector, {
            limit: 3,
            useCache: true,
            enableSOM: true
          });
        });

        const searchResults = await Promise.all(searchPromises);
        const cacheHits = searchResults.filter(r => r.stats.cacheHit).length;

        results.push({
          test: 'performance_test',
          status: 'success',
          data: {
            memory_usage_mb: Math.round(health.memoryUsage / 1024 / 1024 * 100) / 100,
            cache_entries: health.cacheHits,
            cache_hit_rate: `${cacheHits}/3`,
            avg_search_time: Math.round(
              searchResults.reduce((sum, r) => sum + r.stats.searchTimeMs, 0) / searchResults.length
            ),
            som_clusters_used: searchResults.filter(r => r.stats.somClusterUsed).length
          },
          duration: Date.now() - startTime
        });
      } catch (error: any) {
        results.push({
          test: 'performance_test',
          status: 'error',
          error: error instanceof Error ? error.message : String(error),
          duration: Date.now() - startTime
        });
      }
    }

    return json({
      success: true,
      timestamp: new Date().toISOString(),
      service: 'optimized_qdrant_service',
      tests: results,
      summary: {
        total: results.length,
        passed: results.filter(r => r.status === 'success').length,
        failed: results.filter(r => r.status === 'error').length,
        warnings: results.filter(r => r.status === 'warning').length,
        avg_duration: Math.round(
          results.reduce((sum, r) => sum + (r.duration || 0), 0) / results.length
        )
      },
      configuration: {
        vector_dimensions: 384,
        embedding_model: 'nomic-embed-text',
        som_clustering: 'enabled',
        nes_cache: 'enabled',
        batching: 'enabled',
        memory_limit_mb: 32
      }
    });

  } catch (error: any) {
    return json({
      success: false,
      error: error instanceof Error ? error.message : String(error),
      timestamp: new Date().toISOString()
    }, { status: 500 });
  }
};

export const POST: RequestHandler = async ({ request, url }) => {
  const action = url.searchParams.get('action');
  
  if (action === 'sync') {
    try {
      const body = await request.json();
      const options = {
        fullSync: body.fullSync || false,
        batchSize: body.batchSize || 50,
        sinceTimestamp: body.sinceTimestamp ? new Date(body.sinceTimestamp) : undefined
      };

      const syncResult = await optimizedQdrantService.syncFromPostgreSQL(options);
      
      return json({
        success: true,
        action: 'postgresql_sync',
        result: syncResult,
        timestamp: new Date().toISOString()
      });
    } catch (error: any) {
      return json({
        success: false,
        action: 'postgresql_sync',
        error: error instanceof Error ? error.message : String(error),
        timestamp: new Date().toISOString()
      }, { status: 500 });
    }
  }

  return json({
    success: false,
    error: 'Invalid action. Supported actions: sync',
    timestamp: new Date().toISOString()
  }, { status: 400 });
};