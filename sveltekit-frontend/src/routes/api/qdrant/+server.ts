import { json } from "@sveltejs/kit";
import { qdrant, optimizedQdrant, qdrantOptimized } from "$lib/server/vector/qdrant";
import { redisRateLimit, createRateLimitConfig } from "$lib/server/redisRateLimit";
import { productionLogger as logger } from '$lib/server/production-logger';
import { dev } from '$app/environment';
import type { RequestHandler } from './$types.js';
import { URL } from "url";


// Enhanced interfaces for better type safety and optimization
export interface QdrantSyncRequest {
  collection?: string;
  batchSize?: number;
  limit?: number;
  forceRecreate?: boolean;
  source?: 'postgres' | 'csv' | 'json';
  filters?: Record<string, any>;
  useOptimizedService?: boolean; // Use memory-optimized service
  enableCaching?: boolean;       // Enable vector caching
  memoryBudget?: number;        // Memory budget in KB
}

export interface CollectionRequest {
  name: string;
  vectorSize?: number;
  distance?: "Cosine" | "Euclid" | "Dot";
  optimizersConfig?: {
    deletedThreshold?: number;
    vacuumMinVectorNumber?: number;
    defaultSegmentNumber?: number;
    maxSegmentSize?: number;
    memmapThreshold?: number;
    indexingThreshold?: number;
    flushIntervalSec?: number;
  };
  walConfig?: {
    walCapacityMb?: number;
    walSegmentsAhead?: number;
  };
}

export interface SearchRequest {
  collection: string;
  query: number[] | string;
  limit?: number;
  offset?: number;
  filter?: Record<string, any>;
  params?: {
    hnsw_ef?: number;
    exact?: boolean;
  };
  with_payload?: boolean | string[];
  with_vector?: boolean | string[];
  useOptimizedService?: boolean; // Use memory-optimized service
  enableCaching?: boolean;       // Enable search result caching
  threshold?: number;           // Similarity threshold
}

// Enhanced error handling and logging
class QdrantAPIError extends Error {
  constructor(
    message: string,
    public statusCode: number = 500,
    public details?: any
  ) {
    super(message);
    this.name = 'QdrantAPIError';
  }
}

// Utility functions for Windows optimization
function getOptimizedQdrantConfig(vectorSize: number) {
  // Windows-specific optimizations for Qdrant
  const config: any = {
    vectors: {
      size: vectorSize,
      distance: "Cosine"
    },
    optimizers_config: {
      deleted_threshold: 0.2,
      vacuum_min_vector_number: 1000,
      default_segment_number: process.platform === 'win32' ? 2 : 0, // Fewer segments on Windows for better performance
      max_segment_size: process.platform === 'win32' ? 200000 : null, // Limit segment size on Windows
      memmap_threshold: process.platform === 'win32' ? 50000 : 20000, // Higher threshold on Windows
      indexing_threshold: process.platform === 'win32' ? 30000 : 20000, // Adjusted for Windows I/O
      flush_interval_sec: 10,
    },
    wal_config: {
      wal_capacity_mb: process.platform === 'win32' ? 64 : 32, // More WAL capacity on Windows
      wal_segments_ahead: 1,
    }
  };

  // Additional Windows optimizations
  if (process.platform === 'win32') {
    config.hnsw_config = {
      m: 16,
      ef_construct: 128,
      full_scan_threshold: 10000,
      max_indexing_threads: Math.max(1, Math.floor(require('os').cpus().length / 2))
    };
  }

  return config;
}

async function validateQdrantConnection(): Promise<void> {
  const isHealthy = await qdrant.isHealthy();
  if (!isHealthy) {
    throw new QdrantAPIError('Qdrant service is not available', 503);
  }
}

// Sync data from PostgreSQL to Qdrant with Windows optimization
export const POST: RequestHandler = async ({ request, locals, getClientAddress }) => {
  const clientIP = getClientAddress();

  try {
    // Rate limiting
    const rateLimitConfig = createRateLimitConfig(locals.user?.role === 'admin' ? 'admin' : 'api');
    const rateLimitResult = await redisRateLimit({
      key: `qdrant_sync:${clientIP}:${locals.user?.id || 'anonymous'}`,
      ...rateLimitConfig
    });

    if (!rateLimitResult.allowed) {
      return json(
        {
          success: false,
          error: 'Rate limit exceeded',
          retryAfter: rateLimitResult.retryAfter
        },
        { 
          status: 429,
          headers: {
            'Retry-After': rateLimitResult.retryAfter.toString()
          }
        }
      );
    }

    // Check admin permissions for sync operations
    if (!locals.user || locals.user.role !== "admin") {
      return json(
        {
          success: false,
          error: "Admin privileges required for sync operations",
        },
        { status: 403 }
      );
    }

    await validateQdrantConnection();

    const body: QdrantSyncRequest = await request.json();
    const {
      collection = "legal_documents",
      batchSize = process.platform === 'win32' ? 50 : 100, // Smaller batches on Windows
      limit = 1000,
      forceRecreate = false,
      source = 'postgres',
      filters = {}
    } = body;

    const startTime = Date.now();

    // Enhanced sync logic with Windows optimizations
    try {
      let syncResults;

      switch (source) {
        case 'postgres': {
          // Check if collection exists
          const collections = await qdrant.getCollections();
          const collectionExists = collections.some((c: any) => c.name === collection);

          if (!collectionExists || forceRecreate) {
            if (forceRecreate && collectionExists) {
              await qdrant.deleteCollection(collection);
            }

            // Create optimized collection for Windows
            const vectorSize = 384; // Default for sentence transformers
            const config = getOptimizedQdrantConfig(vectorSize);
            await qdrant.createCollection(collection, config);
          }

          // TODO: Implement actual PostgreSQL sync
          // For now, return success with stub data
          syncResults = {
            synced: 0,
            errors: 0,
            collection,
            message: "PostgreSQL sync implementation needed",
            batchSize,
            windowsOptimized: process.platform === 'win32'
          };
          break;
        }

        default:
          throw new QdrantAPIError(`Unsupported sync source: ${source}`, 400);
      }

      const executionTime = Date.now() - startTime;

      return json({
        success: true,
        data: {
          ...syncResults,
          executionTime,
          filters,
          performance: {
            platform: process.platform,
            batchSize,
            limit,
            rateLimit: {
              remaining: rateLimitResult.remaining,
              resetTime: rateLimitResult.resetTime
            }
          }
        },
      });

    } catch (syncError) {
      console.error("Qdrant sync operation failed:", syncError);
      throw new QdrantAPIError(
        'Sync operation failed',
        500,
        dev ? (syncError instanceof Error ? syncError.message : 'Unknown error') : undefined
      );
    }

  } catch (error: any) {
    console.error("Qdrant sync error:", error);
    
    if (error instanceof QdrantAPIError) {
      return json(
        {
          success: false,
          error: error.message,
          details: error.details,
        },
        { status: error.statusCode }
      );
    }

    return json(
      {
        success: false,
        error: "Failed to sync with Qdrant",
        details: dev ? (error instanceof Error ? error.message : "Unknown error") : undefined,
        timestamp: new Date().toISOString()
      },
      { status: 500 }
    );
  }
};

// Enhanced GET endpoint with search capabilities
export const GET: RequestHandler = async ({ url, locals, getClientAddress }) => {
  const clientIP = getClientAddress();

  try {
    // Rate limiting for GET requests
    const rateLimitResult = await redisRateLimit({
      key: `qdrant_get:${clientIP}`,
      limit: 200, // More generous for read operations
      windowSec: 60
    });

    if (!rateLimitResult.allowed) {
      return json(
        {
          success: false,
          error: 'Rate limit exceeded',
          retryAfter: rateLimitResult.retryAfter
        },
        { status: 429 }
      );
    }

    await validateQdrantConnection();

    const collection = url.searchParams.get("collection") || "legal_documents";
    const action = url.searchParams.get("action") || "status";

    switch (action) {
      case 'search': {
        const query = url.searchParams.get("query");
        const limit = Math.min(parseInt(url.searchParams.get("limit") || "10"), 100);
        const offset = parseInt(url.searchParams.get("offset") || "0");

        if (!query) {
          return json(
            { success: false, error: "Query parameter required for search" },
            { status: 400 }
          );
        }

        // Parse query as vector or text
        let vectorQuery: number[];
        try {
          vectorQuery = JSON.parse(query);
          if (!Array.isArray(vectorQuery) || !vectorQuery.every(n => typeof n === 'number')) {
            throw new Error('Invalid vector format');
          }
        } catch {
          return json(
            { success: false, error: "Query must be a valid JSON array of numbers" },
            { status: 400 }
          );
        }

        const searchParams = {
          collection,
          query: vectorQuery,
          limit,
          offset,
          with_payload: true,
          with_vector: false
        };

        const searchResults = await qdrant.search(searchParams);

        return json({
          success: true,
          data: {
            collection,
            query: {
              vector: vectorQuery.slice(0, 5), // Show first 5 dimensions for debugging
              dimensions: vectorQuery.length,
              limit,
              offset
            },
            results: searchResults,
            performance: {
              resultsCount: searchResults.length,
              rateLimit: {
                remaining: rateLimitResult.remaining,
                resetTime: rateLimitResult.resetTime
              }
            }
          }
        });
      }

      case 'status':
      default: {
        let collections, collectionInfo;
        try {
          collections = await qdrant.getCollections();
          collectionInfo = await qdrant.getCollection(collection);
        } catch (err: any) {
          console.error("Failed to get Qdrant collections/info:", err);
          throw new QdrantAPIError(
            'Failed to get Qdrant collections/info',
            500,
            dev ? (err instanceof Error ? err.message : 'Unknown error') : undefined
          );
        }

        // Get system information
        const systemInfo = {
          platform: process.platform,
          arch: process.arch,
          nodeVersion: process.version,
          windowsOptimizations: process.platform === 'win32',
          rateLimit: {
            remaining: rateLimitResult.remaining,
            resetTime: rateLimitResult.resetTime
          }
        };

        return json({
          success: true,
          data: {
            status: "healthy",
            collections,
            currentCollection: {
              name: collection,
              ...collectionInfo,
              windowsOptimized: process.platform === 'win32'
            },
            system: systemInfo,
            endpoints: {
              "POST /api/qdrant": "Sync PostgreSQL data to Qdrant",
              "PUT /api/qdrant": "Create or recreate collection",
              "DELETE /api/qdrant": "Delete collection", 
              "GET /api/qdrant": "Get status and collection info",
              "GET /api/qdrant?action=search&query=[...]": "Vector similarity search",
            },
            capabilities: {
              vectorSearch: true,
              bulkSync: true,
              windowsOptimized: process.platform === 'win32',
              rateLimiting: true,
              adminControls: locals.user?.role === 'admin'
            }
          },
        });
      }
    }

  } catch (error: any) {
    console.error("Qdrant GET error:", error);
    
    if (error instanceof QdrantAPIError) {
      return json(
        {
          success: false,
          error: error.message,
          details: error.details,
        },
        { status: error.statusCode }
      );
    }

    return json(
      {
        success: false,
        error: "Failed to get Qdrant status",
        details: dev ? (error instanceof Error ? error.message : "Unknown error") : undefined,
        timestamp: new Date().toISOString()
      },
      { status: 500 }
    );
  }
};

// Enhanced collection management with Windows optimizations
export const PUT: RequestHandler = async ({ request, locals, getClientAddress }) => {
  const clientIP = getClientAddress();

  try {
    // Rate limiting for collection operations
    const rateLimitResult = await redisRateLimit({
      key: `qdrant_collection:${clientIP}:${locals.user?.id || 'anonymous'}`,
      limit: 10, // Stricter for collection operations
      windowSec: 60
    });

    if (!rateLimitResult.allowed) {
      return json(
        {
          success: false,
          error: 'Rate limit exceeded',
          retryAfter: rateLimitResult.retryAfter
        },
        { status: 429 }
      );
    }

    // Check admin permissions
    if (!locals.user || locals.user.role !== "admin") {
      return json(
        {
          success: false,
          error: "Admin privileges required for collection management",
        },
        { status: 403 }
      );
    }

    await validateQdrantConnection();

    const body: CollectionRequest = await request.json();
    const {
      name,
      vectorSize = 384, // Default for sentence-transformers/all-MiniLM-L6-v2
      distance = "Cosine",
      optimizersConfig,
      walConfig
    } = body;

    if (!name || name.trim().length === 0) {
      return json(
        {
          success: false,
          error: "Collection name is required",
        },
        { status: 400 }
      );
    }

    // Create collection with Windows optimizations
    const collectionConfig = getOptimizedQdrantConfig(vectorSize);
    
    // Override with custom configs if provided
    if (optimizersConfig) {
      collectionConfig.optimizers_config = { ...collectionConfig.optimizers_config, ...optimizersConfig };
    }
    if (walConfig) {
      collectionConfig.wal_config = { ...collectionConfig.wal_config, ...walConfig };
    }

    // Override distance metric
    collectionConfig.vectors.distance = distance;

    const result = await qdrant.createCollection(name, collectionConfig);

    return json({
      success: true,
      data: {
        message: `Collection '${name}' created successfully with Windows optimizations`,
        collection: name,
        config: {
          vectorSize,
          distance,
          windowsOptimized: process.platform === 'win32',
          optimizations: {
            platform: process.platform,
            memoryMappingThreshold: collectionConfig.optimizers_config.memmap_threshold,
            segmentConfiguration: collectionConfig.optimizers_config.default_segment_number,
            flushInterval: collectionConfig.optimizers_config.flush_interval_sec
          }
        },
        result,
        rateLimit: {
          remaining: rateLimitResult.remaining,
          resetTime: rateLimitResult.resetTime
        }
      },
    });

  } catch (error: any) {
    console.error("Qdrant collection creation error:", error);
    
    if (error instanceof QdrantAPIError) {
      return json(
        {
          success: false,
          error: error.message,
          details: error.details,
        },
        { status: error.statusCode }
      );
    }

    return json(
      {
        success: false,
        error: "Failed to create collection",
        details: dev ? (error instanceof Error ? error.message : "Unknown error") : undefined,
        timestamp: new Date().toISOString()
      },
      { status: 500 }
    );
  }
};

// Enhanced collection deletion with safety checks
export const DELETE: RequestHandler = async ({ request, locals, getClientAddress }) => {
  const clientIP = getClientAddress();

  try {
    // Rate limiting for deletion operations
    const rateLimitResult = await redisRateLimit({
      key: `qdrant_delete:${clientIP}:${locals.user?.id || 'anonymous'}`,
      limit: 5, // Very strict for deletions
      windowSec: 300 // 5-minute window
    });

    if (!rateLimitResult.allowed) {
      return json(
        {
          success: false,
          error: 'Rate limit exceeded for deletion operations',
          retryAfter: rateLimitResult.retryAfter
        },
        { status: 429 }
      );
    }

    // Check admin permissions
    if (!locals.user || locals.user.role !== "admin") {
      return json(
        {
          success: false,
          error: "Admin privileges required for collection deletion",
        },
        { status: 403 }
      );
    }

    await validateQdrantConnection();

    const { collection, forceDelete = false, confirmationToken } = await request.json();

    if (!collection || collection.trim().length === 0) {
      return json(
        {
          success: false,
          error: "Collection name is required",
        },
        { status: 400 }
      );
    }

    // Safety checks for critical collections
    const protectedCollections = ['legal_documents', 'default', 'production'];
    if (protectedCollections.includes(collection) && !forceDelete) {
      return json(
        {
          success: false,
          error: `Cannot delete protected collection '${collection}'. Use forceDelete=true with confirmationToken to override.`,
          hint: 'Protected collections require explicit confirmation'
        },
        { status: 400 }
      );
    }

    // Additional confirmation for forced deletions
    if (forceDelete && !confirmationToken) {
      return json(
        {
          success: false,
          error: "Confirmation token required for force deletion",
          hint: 'Add confirmationToken with collection name to confirm'
        },
        { status: 400 }
      );
    }

    if (forceDelete && confirmationToken !== collection) {
      return json(
        {
          success: false,
          error: "Invalid confirmation token",
        },
        { status: 400 }
      );
    }

    // Check if collection exists before deletion
    try {
      await qdrant.getCollection(collection);
    } catch {
      return json(
        {
          success: false,
          error: `Collection '${collection}' does not exist`,
        },
        { status: 404 }
      );
    }

    const result = await qdrant.deleteCollection(collection);

    return json({
      success: true,
      data: {
        message: `Collection '${collection}' deleted successfully`,
        collection,
        forced: forceDelete,
        result,
        rateLimit: {
          remaining: rateLimitResult.remaining,
          resetTime: rateLimitResult.resetTime
        },
        timestamp: new Date().toISOString()
      },
    });

  } catch (error: any) {
    console.error("Qdrant collection deletion error:", error);
    
    if (error instanceof QdrantAPIError) {
      return json(
        {
          success: false,
          error: error.message,
          details: error.details,
        },
        { status: error.statusCode }
      );
    }

    return json(
      {
        success: false,
        error: "Failed to delete collection",
        details: dev ? (error instanceof Error ? error.message : "Unknown error") : undefined,
        timestamp: new Date().toISOString()
      },
      { status: 500 }
    );
  }
};