
// --- Qdrant passthroughs for admin API (enhanced with logging) ---
export async function getCollections(): Promise<any> {
  const wrapper = getQdrantWrapper();
  if (!wrapper) {
    logger.error("Qdrant not configured", undefined, {
      component: 'QdrantService',
      service: 'qdrant'
    });
    throw new Error("Qdrant not configured");
  }
  
  return wrapper.getCollections();
}

export async function getCollection(collection: string): Promise<any> {
  const wrapper = getQdrantWrapper();
  if (!wrapper) {
    logger.error("Qdrant not configured", undefined, {
      component: 'QdrantService',
      service: 'qdrant'
    });
    throw new Error("Qdrant not configured");
  }
  
  return wrapper.getCollection(collection);
}

export async function createCollection(name: string, config: any): Promise<any> {
  const wrapper = getQdrantWrapper();
  if (!wrapper) {
    logger.error("Qdrant not configured", undefined, {
      component: 'QdrantService',
      service: 'qdrant'
    });
    throw new Error("Qdrant not configured");
  }
  
  return wrapper.createCollection(name, config);
}

export async function deleteCollection(name: string): Promise<any> {
  const wrapper = getQdrantWrapper();
  if (!wrapper) {
    logger.error("Qdrant not configured", undefined, {
      component: 'QdrantService', 
      service: 'qdrant'
    });
    throw new Error("Qdrant not configured");
  }
  
  return wrapper.deleteCollection(name);
}
// Qdrant vector search service
// High-performance vector search with memory optimization
// Now using optimized service with cache-like logging system
import { optimizedQdrant, qdrantOptimized } from './qdrant-optimized.js';
import { createQdrantWrapper, QdrantApiWrapper } from './qdrant-api-wrapper.js';
import { generateEmbedding } from '../ai/embeddings-simple.js';
import { productionLogger as logger } from '../production-logger.js';

let qdrantWrapper: QdrantApiWrapper | null = null;

// Initialize Qdrant wrapper (enhanced API compatibility)
function getQdrantWrapper(): QdrantApiWrapper | null {
  if (!import.meta.env.QDRANT_URL) {
    return null;
  }
  if (!qdrantWrapper) {
    qdrantWrapper = createQdrantWrapper({
      url: import.meta.env.QDRANT_URL,
      apiKey: import.meta.env.QDRANT_API_KEY || undefined,
    });
  }
  return qdrantWrapper;
}

// Legacy compatibility - maintain old interface
function getQdrantClient() {
  return getQdrantWrapper();
}
// Collection names (optimized for memory efficiency)
const COLLECTIONS = {
  CASES: "prosecutor_cases",
  EVIDENCE: "prosecutor_evidence", 
  DOCUMENTS: "prosecutor_documents",
  // New collections for optimized storage
  LEGAL_DOCUMENTS: "legal_documents",
  CASE_SUMMARIES: "case_summaries",
  EVIDENCE_METADATA: "evidence_metadata"
} as const;

// Vector dimensions (adjusted for memory optimization)
// Using 384 dimensions (sentence-transformers/all-MiniLM-L6-v2) instead of 1536 for better memory usage
const VECTOR_DIMENSION = 384;
const LEGACY_VECTOR_DIMENSION = 1536; // For backward compatibility

// Initialize collections with memory optimization
export async function initializeCollections(): Promise<void> {
  const client = getQdrantClient();
  if (!client) {
    logger.warn("Qdrant not configured, skipping collection initialization", {
      component: 'QdrantService',
      service: 'qdrant'
    });
    return;
  }
  
  try {
    const isWindows = typeof process !== 'undefined' && process.platform === 'win32';
    
    for (const [name, collectionName] of Object.entries(COLLECTIONS)) {
      try {
        // Check if collection exists
        await client.getCollection(collectionName);
        logger.info(`Qdrant collection ${collectionName} already exists`, {
          component: 'QdrantService',
          service: 'qdrant'
        });
      } catch (error: any) {
        // Collection doesn't exist, create it with optimized config
        const config = {
          vectors: {
            size: VECTOR_DIMENSION,
            distance: "Cosine" as const,
            hnsw_config: {
              m: isWindows ? 16 : 12,                    // Slightly lower on non-Windows
              ef_construct: isWindows ? 100 : 80,        // Reduce construction effort
              full_scan_threshold: isWindows ? 10000 : 5000, // Lower threshold for better memory
              max_indexing_threads: 1,                   // Single thread for memory optimization
            },
          },
          optimizers_config: {
            default_segment_number: isWindows ? 2 : 1,  // Fewer segments on non-Windows
            max_segment_size: isWindows ? 20000 : 10000, // Smaller segments for memory
            indexing_threshold: isWindows ? 20000 : 10000,
            flush_interval_sec: 10,                     // More frequent flushes
            max_optimization_threads: 1,               // Single thread optimization
            memmap_threshold: isWindows ? 50000 : 20000, // Higher threshold on Windows
          },
          quantization_config: {
            scalar: {
              type: "int8" as const,    // Aggressive quantization for memory
              quantile: 0.99,
              always_ram: false         // Allow disk storage
            },
          },
          // Additional memory optimizations
          wal_config: {
            wal_capacity_mb: isWindows ? 32 : 16,  // Smaller WAL for memory
            wal_segments_ahead: 1
          }
        };
        
        await client.createCollection(collectionName, config);
        logger.info(`Created optimized Qdrant collection ${collectionName}`, {
          component: 'QdrantService',
          service: 'qdrant'
        }, {
          windowsOptimized: isWindows,
          vectorDimension: VECTOR_DIMENSION,
          quantizationEnabled: true
        });
      }
    }
  } catch (error: any) {
    logger.error("Failed to initialize Qdrant collections", error instanceof Error ? error : undefined, {
      component: 'QdrantService',
      service: 'qdrant'
    });
  }
}
// Search interface
export interface SearchOptions {
  limit?: number;
  offset?: number;
  filter?: unknown;
  withPayload?: boolean;
  scoreThreshold?: number;
}
// Search cases in Qdrant (using optimized service)
export async function searchCases(
  query: string,
  options: SearchOptions = {}
): Promise<any[]> {
  try {
    // Use optimized service with caching and memory efficiency
    return await qdrantOptimized.search(COLLECTIONS.CASES, query, {
      limit: options.limit || 20,
      offset: options.offset || 0,
      filter: options.filter,
      threshold: options.scoreThreshold || 0.7,
      useCache: true // Enable caching for better performance
    });
  } catch (error: any) {
    logger.error("Optimized case search failed", error instanceof Error ? error : undefined, {
      component: 'QdrantService',
      service: 'qdrant'
    }, {
      query: typeof query === 'string' ? query.substring(0, 100) : 'vector_query',
      options
    });
    return [];
  }
}
// Search evidence in Qdrant (using optimized service)
export async function searchEvidence(
  query: string,
  options: SearchOptions = {}
): Promise<any[]> {
  try {
    // Use optimized service with caching and memory efficiency
    return await qdrantOptimized.search(COLLECTIONS.EVIDENCE, query, {
      limit: options.limit || 20,
      offset: options.offset || 0,
      filter: options.filter,
      threshold: options.scoreThreshold || 0.7,
      useCache: true // Enable caching for better performance
    });
  } catch (error: any) {
    logger.error("Optimized evidence search failed", error instanceof Error ? error : undefined, {
      component: 'QdrantService',
      service: 'qdrant'
    }, {
      query: typeof query === 'string' ? query.substring(0, 100) : 'vector_query',
      options
    });
    return [];
  }
}
// Add or update a case in Qdrant (with optimized batching support)
export async function upsertCase(
  id: string,
  embedding: number[],
  payload: any
): Promise<void> {
  const wrapper = getQdrantWrapper();
  if (!wrapper) {
    logger.warn("Qdrant not configured for case upsert", {
      component: 'QdrantService',
      service: 'qdrant'
    });
    return;
  }
  
  try {
    // Convert to regular array for API compatibility
    const vectorArray = Array.from(embedding);
    
    await wrapper.upsert(COLLECTIONS.CASES, {
      wait: true,
      points: [
        {
          id,
          vector: vectorArray,
          payload,
        },
      ],
    });
    
    logger.debug('Case upserted successfully', {
      component: 'QdrantService',
      service: 'qdrant'
    }, {
      caseId: id,
      vectorDimension: embedding.length,
      payloadSize: JSON.stringify(payload).length
    });
  } catch (error: any) {
    logger.error("Failed to upsert case in Qdrant", error instanceof Error ? error : undefined, {
      component: 'QdrantService',
      service: 'qdrant'
    }, {
      caseId: id,
      vectorDimension: embedding.length
    });
  }
}
// Add or update evidence in Qdrant (with optimized batching support)
export async function upsertEvidence(
  id: string,
  embedding: number[],
  payload: any
): Promise<void> {
  const wrapper = getQdrantWrapper();
  if (!wrapper) {
    logger.warn("Qdrant not configured for evidence upsert", {
      component: 'QdrantService',
      service: 'qdrant'
    });
    return;
  }
  
  try {
    // Convert to regular array for API compatibility
    const vectorArray = Array.from(embedding);
    
    await wrapper.upsert(COLLECTIONS.EVIDENCE, {
      wait: true,
      points: [
        {
          id,
          vector: vectorArray,
          payload,
        },
      ],
    });
    
    logger.debug('Evidence upserted successfully', {
      component: 'QdrantService',
      service: 'qdrant'
    }, {
      evidenceId: id,
      vectorDimension: embedding.length,
      payloadSize: JSON.stringify(payload).length
    });
  } catch (error: any) {
    logger.error("Failed to upsert evidence in Qdrant", error instanceof Error ? error : undefined, {
      component: 'QdrantService',
      service: 'qdrant'
    }, {
      evidenceId: id,
      vectorDimension: embedding.length
    });
  }
}
// Delete a point from Qdrant (with enhanced logging)
export async function deletePoint(
  collection: string,
  id: string
): Promise<void> {
  const wrapper = getQdrantWrapper();
  if (!wrapper) {
    logger.warn("Qdrant not configured for point deletion", {
      component: 'QdrantService',
      service: 'qdrant'
    });
    return;
  }
  
  try {
    await wrapper.delete(collection, {
      wait: true,
      points: [id],
    });
    
    logger.info('Point deleted successfully', {
      component: 'QdrantService',
      service: 'qdrant'
    }, {
      collection,
      pointId: id
    });
  } catch (error: any) {
    logger.error("Failed to delete point from Qdrant", error instanceof Error ? error : undefined, {
      component: 'QdrantService',
      service: 'qdrant'
    }, {
      collection,
      pointId: id
    });
  }
}
// Health check (using wrapper and optimized service)
export async function isQdrantHealthy(): Promise<boolean> {
  try {
    // Try optimized service first
    if (qdrantOptimized && qdrantOptimized.isHealthy) {
      return await qdrantOptimized.isHealthy();
    }
    
    // Fallback to wrapper health check
    const wrapper = getQdrantWrapper();
    if (wrapper) {
      const health = await wrapper.healthCheck();
      return health.status === 'healthy';
    }
    
    return false;
  } catch (error: any) {
    logger.error("Qdrant health check failed", error instanceof Error ? error : undefined, {
      component: 'QdrantService',
      service: 'qdrant'
    });
    return false;
  }
}
// Export for easier usage (enhanced with optimized service)
export const qdrant = {
  // Optimized search methods
  searchCases,
  searchEvidence,
  
  // Legacy methods (maintained for compatibility)
  upsertCase,
  upsertEvidence,
  deletePoint,
  
  // Enhanced methods
  isHealthy: isQdrantHealthy,
  initializeCollections,
  
  // Collection and client management
  collections: COLLECTIONS,
  getCollections,
  getCollection,
  createCollection,
  deleteCollection,
  
  // New optimized methods
  optimized: qdrantOptimized,
  
  // Memory and performance monitoring
  getMemoryUsage: () => qdrantOptimized.getMemoryUsage(),
  getPerformanceMetrics: () => qdrantOptimized.getPerformanceMetrics(),
  getQueryHistory: () => qdrantOptimized.getQueryHistory(),
  clearCaches: () => qdrantOptimized.clearCaches(),
  
  // Batch operations for memory efficiency
  upsertBatch: (collection: string, points: Array<{ id: string; vector: number[]; payload: any }>) =>
    qdrantOptimized.upsertBatch(collection, points),
    
  // Search with advanced options
  searchOptimized: (collection: string, query: string | number[], options = {}) => 
    qdrantOptimized.search(collection, query, options)
};
