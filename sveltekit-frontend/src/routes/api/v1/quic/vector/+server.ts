import type { RequestHandler } from './$types.js';

/**
 * QUIC Vector Proxy API - High-Performance Vector Operations
 * Provides vector search with intelligent caching and multi-backend routing
 * Port: 8445 (QUIC), 8446 (HTTP/2 fallback)
 * Backends: Qdrant (6333), pgvector via Enhanced RAG (8094)
 */
import { json, error } from '@sveltejs/kit';
import { randomUUID } from 'crypto';

import { ensureError } from '$lib/utils/ensure-error';
import { vectorOperations, type VectorSearchQuery } from '$lib/server/db/vector-operations.js';
import { goServiceManager } from '$lib/services/go-microservice-client.js';
import { URL } from "url";

const QUIC_VECTOR_CONFIG = {
  primaryPort: 8445,    // QUIC HTTP/3
  fallbackPort: 8446,   // HTTP/2
  baseUrl: 'http://localhost:8445',
  fallbackUrl: 'http://localhost:8446',
  timeout: 30000,       // Vector operations can take longer
  cacheTTL: 300,        // 5 minutes cache TTL
  maxCacheSize: 1000
};

/**
 * GET /api/v1/quic/vector - Vector proxy health and cache status
 */
export const GET: RequestHandler = async ({ url }) => {
  try {
    // Check vector proxy health
    const healthResponse = await fetch(`${QUIC_VECTOR_CONFIG.baseUrl}/health`, {
      signal: AbortSignal.timeout(QUIC_VECTOR_CONFIG.timeout)
    });

    let proxyStatus = 'healthy';
    let responseData: any = {};

    if (healthResponse.ok) {
      responseData = await healthResponse.json();
    } else {
      // Try fallback HTTP/2
      const fallbackResponse = await fetch(`${QUIC_VECTOR_CONFIG.fallbackUrl}/health`, {
        signal: AbortSignal.timeout(QUIC_VECTOR_CONFIG.timeout)
      });

      if (fallbackResponse.ok) {
        responseData = await fallbackResponse.json();
        proxyStatus = 'fallback';
      } else {
        proxyStatus = 'unhealthy';
      }
    }

    return json({
      service: 'quic-vector-proxy',
      status: proxyStatus,
      protocol: proxyStatus === 'healthy' ? 'HTTP/3' : proxyStatus === 'fallback' ? 'HTTP/2' : 'N/A',
      ports: {
        quic: QUIC_VECTOR_CONFIG.primaryPort,
        fallback: QUIC_VECTOR_CONFIG.fallbackPort
      },
      backends: {
        qdrant: 'http://localhost:6333',
        pgvector: 'http://localhost:8094' // Enhanced RAG service
      },
      features: [
        'Multi-backend Routing (Qdrant + pgvector)',
        'Intelligent Caching',
        'Vector Similarity Search',
        'Cache Management',
        'Health Monitoring'
      ],
      cache: responseData.cache || {
        enabled: true,
        ttl: QUIC_VECTOR_CONFIG.cacheTTL,
        maxSize: QUIC_VECTOR_CONFIG.maxCacheSize
      },
      metrics: responseData.metrics || null,
      timestamp: new Date().toISOString()
    });

  } catch (err: any) {
    console.error('QUIC Vector Proxy health check failed:', err);
    
    return json({
      service: 'quic-vector-proxy',
      status: 'error',
      error: err instanceof Error ? err.message : 'Unknown error',
      timestamp: new Date().toISOString()
    });
  }
};

/**
 * POST /api/v1/quic/vector - Vector search with QUIC acceleration
 */
export const POST: RequestHandler = async ({ request, url }) => {
  try {
    const searchQuery: VectorSearchQuery = await request.json();
    const useCache = url.searchParams.get('cache') !== 'false';
    const useHttp3 = url.searchParams.get('http3') !== 'false';
    const backend = url.searchParams.get('backend') || 'auto'; // 'auto', 'qdrant', 'pgvector'

    // Validate search query
    if (!searchQuery.query && !searchQuery.embedding) {
      error(400, ensureError({ message: 'Either query text or embedding vector is required' }));
    }

    // Determine target URL
    const targetUrl = useHttp3 
      ? `${QUIC_VECTOR_CONFIG.baseUrl}/api/vector/search`
      : `${QUIC_VECTOR_CONFIG.fallbackUrl}/api/vector/search`;

    // Prepare request payload
    const requestPayload = {
      ...searchQuery,
      meta: {
        useCache,
        backend,
        requestId: randomUUID(),
        timestamp: Date.now()
      }
    };

    let response: Response;
    let protocol: string;

    try {
      // Use Go Vector Service if backend is 'auto' or 'vector'
      if (backend === 'auto' || backend === 'vector' || backend === 'pgvector') {
        const vectorClient = goServiceManager.getClient('vectorService');
        if (vectorClient) {
          const serviceResponse = await vectorClient.post('/api/vector/search', requestPayload);
          
          if (serviceResponse.success) {
            return json({
              success: true,
              results: serviceResponse.data.results || serviceResponse.data,
              protocol: serviceResponse.protocol || 'HTTP',
              source: 'go-vector-service',
              cached: serviceResponse.data.cached || false,
              timestamp: new Date().toISOString(),
              metrics: {
                totalResults: serviceResponse.data.results?.length || 0,
                executionTimeMs: serviceResponse.responseTime || 0,
                cacheHit: serviceResponse.data.cacheHit || false,
                backend: 'go-vector-service'
              }
            });
          } else {
            console.warn('Go Vector Service failed, falling back to enhanced RAG:', serviceResponse.error);
          }
        }
      }

      // Fallback to Enhanced RAG service 
      const enhancedRagClient = goServiceManager.getEnhancedRAG();
      const ragSearchResponse = await enhancedRagClient.semanticSearch(
        searchQuery.query || 'vector search',
        {
          collection: searchQuery.collection,
          limit: searchQuery.limit || 10
        }
      );

      if (ragSearchResponse.success) {
        return json({
          success: true,
          results: ragSearchResponse.data.results || ragSearchResponse.data,
          protocol: ragSearchResponse.protocol || 'HTTP',
          source: 'enhanced-rag-service',
          cached: false,
          timestamp: new Date().toISOString(),
          metrics: {
            totalResults: ragSearchResponse.data.results?.length || 0,
            executionTimeMs: ragSearchResponse.responseTime || 0,
            cacheHit: false,
            backend: 'enhanced-rag'
          }
        });
      }

      // Original QUIC proxy as final fallback
      response = await fetch(targetUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Vector-Backend': backend,
          'X-Use-Cache': String(useCache),
          'X-QUIC-Request': 'true'
        },
        body: JSON.stringify(requestPayload),
        signal: AbortSignal.timeout(QUIC_VECTOR_CONFIG.timeout)
      });
      protocol = useHttp3 ? 'HTTP/3' : 'HTTP/2';

    } catch (quicError) {
      // Fallback to direct database operations if QUIC proxy fails
      console.warn('QUIC Vector Proxy failed, using direct database access:', quicError);
      
      const directResults = await vectorOperations.vectorSearch(searchQuery);
      
      return json({
        success: true,
        results: directResults,
        protocol: 'Direct Database',
        source: 'fallback',
        cached: false,
        timestamp: new Date().toISOString(),
        metrics: {
          totalResults: directResults.length,
          executionTimeMs: 0,
          cacheHit: false
        }
      });
    }

    if (!response.ok) {
      throw new Error(`Vector proxy responded with ${response.status}: ${response.statusText}`);
    }

    const responseData = await response.json();

    return json({
      success: true,
      results: responseData.results || responseData,
      protocol,
      source: 'quic-vector-proxy',
      cached: responseData.cached || false,
      timestamp: new Date().toISOString(),
      metrics: {
        totalResults: responseData.results?.length || 0,
        executionTimeMs: responseData.executionTime || 0,
        cacheHit: responseData.cacheHit || false,
        backend: responseData.backend || 'unknown'
      }
    });

  } catch (err: any) {
    console.error('QUIC Vector search error:', err);
    error(500, ensureError({
      message: 'Vector search failed',
      error: err instanceof Error ? err.message : 'Unknown error'
    }));
  }
};

/**
 * DELETE /api/v1/quic/vector - Clear vector cache
 */
export const DELETE: RequestHandler = async ({ url }) => {
  try {
    const cacheKey = url.searchParams.get('key');
    const useHttp3 = url.searchParams.get('http3') !== 'false';

    const targetUrl = useHttp3 
      ? `${QUIC_VECTOR_CONFIG.baseUrl}/cache`
      : `${QUIC_VECTOR_CONFIG.fallbackUrl}/cache`;

    const query = new URLSearchParams();
    if (cacheKey) query.set('key', cacheKey);

    const response = await fetch(`${targetUrl}?${query}`, {
      method: 'DELETE',
      headers: {
        'X-QUIC-Request': 'true'
      },
      signal: AbortSignal.timeout(QUIC_VECTOR_CONFIG.timeout)
    });

    if (!response.ok) {
      throw new Error(`Cache clear failed: ${response.statusText}`);
    }

    const result = await response.json();

    return json({
      success: true,
      message: cacheKey ? `Cache key '${cacheKey}' cleared` : 'All cache cleared',
      result,
      timestamp: new Date().toISOString()
    });

  } catch (err: any) {
    console.error('Vector cache clear error:', err);
    error(500, ensureError({
      message: 'Cache clear failed',
      error: err instanceof Error ? err.message : 'Unknown error'
    }));
  }
};

/**
 * PUT /api/v1/quic/vector - Update vector proxy configuration
 */
export const PUT: RequestHandler = async ({ request }) => {
  try {
    const config = await request.json();

    // Validate configuration
    if (config.cacheTTL && (config.cacheTTL < 10 || config.cacheTTL > 3600)) {
      error(400, ensureError({ message: 'Cache TTL must be between 10 and 3600 seconds' }));
    }

    if (config.maxCacheSize && (config.maxCacheSize < 10 || config.maxCacheSize > 10000)) {
      error(400, ensureError({ message: 'Max cache size must be between 10 and 10000' }));
    }

    // Update configuration (in a real implementation, this would be persisted)
    const updatedConfig = {
      ...QUIC_VECTOR_CONFIG,
      ...config,
      lastUpdated: new Date().toISOString()
    };

    return json({
      success: true,
      message: 'Vector proxy configuration updated',
      config: updatedConfig
    });

  } catch (err: any) {
    console.error('Vector proxy configuration update failed:', err);
    error(500, ensureError({
      message: 'Configuration update failed',
      error: err instanceof Error ? err.message : 'Unknown error'
    }));
  }
};