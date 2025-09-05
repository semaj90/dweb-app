// NES GPU Integration - RTX Tensor Core Acceleration for Legal AI
// Integrates Go microservices with SvelteKit 2 + Svelte 5 reactive patterns
// Features: GPU embeddings, multi-layer caching, batch processing, XState orchestration

import { writable, derived, get } from 'svelte/store';
import { createMachine, interpret, type ActorRefFrom } from 'xstate';
import Loki from 'lokijs';
import Fuse from 'fuse.js';

// Types for RTX GPU Integration
export interface GPUEmbeddingRequest {
  texts: string[];
  use_cache?: boolean;
  normalize?: boolean;
  model_name?: string;
  fp16_precision?: boolean;
}

export interface GPUEmbeddingResponse {
  embeddings: EmbeddingVector[];
  metrics: ProcessingMetrics;
  success: boolean;
  error?: string;
}

export interface EmbeddingVector {
  values: number[];
  dimensions: number;
  text_hash: string;
}

export interface ProcessingMetrics {
  processing_time_ms: number;
  batch_size: number;
  cache_hits: number;
  gpu_used: boolean;
  fp16_used: boolean;
  tensor_cores_used: boolean;
  method: string;
  gpu_info?: GPUInfo;
}

export interface GPUInfo {
  gpu_name: string;
  memory_allocated_gb: number;
  memory_total_gb: number;
  memory_utilization: number;
  compute_capability: number;
}

export interface Neo4jSearchRequest {
  query: string;
  query_vector?: number[];
  practice_area: string;
  document_type: string;
  max_results?: number;
  min_confidence?: number;
  search_radius?: number;
  use_gpu?: boolean;
  use_fp16?: boolean;
  use_cache?: boolean;
  batch_optimization?: boolean;
  metadata?: Record<string, any>;
}

export interface Neo4jSearchResponse {
  results: EnhancedResult[];
  total_found: number;
  processing_info: ProcessingInfo;
  performance_info: PerformanceInfo;
  timestamp: string;
}

export interface EnhancedResult {
  node_id: string;
  document_id: string;
  title: string;
  content?: string;
  similarity_score: number;
  distance: number;
  confidence: number;
  practice_area: string;
  document_type: string;
  embedding?: number[];
  related_nodes: RelatedNodeInfo[];
  graph_path?: GraphPathNode[];
  metadata: Record<string, any>;
  processing_source: 'GPU' | 'CPU' | 'CACHE';
}

export interface RelatedNodeInfo {
  node_id: string;
  relation_type: string;
  weight: number;
  distance: number;
  properties: Record<string, any>;
}

export interface GraphPathNode {
  node_id: string;
  node_type: string;
  similarity: number;
}

export interface ProcessingInfo {
  query_embedding_generated: boolean;
  embedding_method: string;
  similarity_method: string;
  graph_traversal_depth: number;
  nodes_processed: number;
  batches_processed: number;
  cache_operations: number;
  gpu_utilization: boolean;
  tensor_cores_used: boolean;
  fp16_precision: boolean;
}

export interface PerformanceInfo {
  total_time_ms: number;
  embedding_time_ms: number;
  database_time_ms: number;
  similarity_time_ms: number;
  cache_time_ms: number;
  network_time_ms: number;
  batch_processing_time_ms: number;
  memory_usage_mb: number;
}

// Service URLs Configuration
export const GPU_SERVICE_URLS = {
  NEO4J_SIMD: 'http://localhost:8091/api/neo4j-simd',
  NEO4J_GPU_INTEGRATED: 'http://localhost:8092/api/neo4j-gpu',
  GPU_TENSOR_WORKER: 'http://localhost:50051',
  ENHANCED_RAG: 'http://localhost:8094/api/rag',
} as const;

// Loki.js Database for Client-Side Caching
export class LegalAIDatabase {
  private db: Loki;
  private embeddings: Collection<any>;
  private searchResults: Collection<any>;
  private documents: Collection<any>;
  private fuseIndex: Fuse<any> | null = null;

  constructor() {
    this.db = new Loki('legalAI.db', {
      autosave: true,
      autoload: true,
      autoloadCallback: this.initCollections.bind(this),
    });
  }

  private initCollections() {
    this.embeddings = this.db.getCollection('embeddings') || this.db.addCollection('embeddings', {
      indices: ['text_hash', 'model_name', 'created_at'],
      ttl: 1800000, // 30 minutes
    });

    this.searchResults = this.db.getCollection('searchResults') || this.db.addCollection('searchResults', {
      indices: ['query_hash', 'practice_area', 'document_type'],
      ttl: 900000, // 15 minutes
    });

    this.documents = this.db.getCollection('documents') || this.db.addCollection('documents', {
      indices: ['document_id', 'title', 'practice_area'],
    });

    // Initialize Fuse.js for fuzzy search
    this.setupFuseIndex();
  }

  private setupFuseIndex() {
    const allDocs = this.documents.find();
    if (allDocs.length > 0) {
      this.fuseIndex = new Fuse(allDocs, {
        keys: ['title', 'content', 'practice_area', 'document_type'],
        threshold: 0.4,
        includeScore: true,
      });
    }
  }

  // Cache embedding
  cacheEmbedding(textHash: string, embedding: EmbeddingVector, modelName: string = 'default') {
    this.embeddings.insertOne({
      text_hash: textHash,
      embedding: embedding.values,
      dimensions: embedding.dimensions,
      model_name: modelName,
      created_at: Date.now(),
    });
  }

  // Get cached embedding
  getCachedEmbedding(textHash: string, modelName: string = 'default'): EmbeddingVector | null {
    const cached = this.embeddings.findOne({ 
      text_hash: textHash, 
      model_name: modelName 
    });

    if (cached) {
      return {
        values: cached.embedding,
        dimensions: cached.dimensions,
        text_hash: textHash,
      };
    }

    return null;
  }

  // Cache search results
  cacheSearchResults(queryHash: string, results: Neo4jSearchResponse) {
    this.searchResults.insertOne({
      query_hash: queryHash,
      results: results.results,
      total_found: results.total_found,
      processing_info: results.processing_info,
      performance_info: results.performance_info,
      cached_at: Date.now(),
    });
  }

  // Get cached search results
  getCachedSearchResults(queryHash: string): Neo4jSearchResponse | null {
    const cached = this.searchResults.findOne({ query_hash: queryHash });

    if (cached) {
      return {
        results: cached.results,
        total_found: cached.total_found,
        processing_info: cached.processing_info,
        performance_info: cached.performance_info,
        timestamp: new Date(cached.cached_at).toISOString(),
      };
    }

    return null;
  }

  // Fuzzy search over cached documents
  fuzzySearch(query: string, limit: number = 10) {
    if (!this.fuseIndex) {
      this.setupFuseIndex();
    }

    if (!this.fuseIndex) return [];

    const results = this.fuseIndex.search(query, { limit });
    return results.map(result => ({
      ...result.item,
      fuzzy_score: 1 - (result.score || 0),
    }));
  }

  // Store document for fuzzy search
  storeDocument(doc: EnhancedResult) {
    const existing = this.documents.findOne({ document_id: doc.document_id });
    if (!existing) {
      this.documents.insertOne(doc);
      this.setupFuseIndex(); // Refresh Fuse index
    }
  }

  // Get database statistics
  getStats() {
    return {
      embeddings_count: this.embeddings.count(),
      search_results_count: this.searchResults.count(),
      documents_count: this.documents.count(),
      database_size: this.db.serialize().length,
    };
  }
}

// XState Machine for Legal AI Workflow
export const legalAIMachine = createMachine({
  id: 'legalAI',
  initial: 'idle',
  context: {
    isAuthenticated: false,
    messageHistory: [],
    documents: [],
    searchResults: [],
    isProcessing: false,
    gpuAvailable: false,
    currentQuery: '',
    selectedPracticeArea: '',
    selectedDocumentType: '',
    processingMetrics: null,
    errorMessage: null,
    batchQueue: [],
    cacheStats: {
      hits: 0,
      misses: 0,
      total: 0,
    },
  },
  states: {
    idle: {
      on: {
        LOGIN: 'authenticating',
        SEARCH: {
          target: 'searching',
          actions: 'setQuery',
        },
        UPLOAD: 'uploading',
        BATCH_PROCESS: 'batchProcessing',
        CHECK_GPU: 'checkingGPU',
      },
    },
    authenticating: {
      on: {
        SUCCESS: {
          target: 'idle',
          actions: 'setAuthenticated',
        },
        FAILURE: {
          target: 'idle',
          actions: 'setError',
        },
      },
    },
    checkingGPU: {
      invoke: {
        src: 'checkGPUAvailability',
        onDone: {
          target: 'idle',
          actions: 'setGPUStatus',
        },
        onError: {
          target: 'idle',
          actions: 'setGPUUnavailable',
        },
      },
    },
    searching: {
      invoke: {
        src: 'performSearch',
        onDone: {
          target: 'idle',
          actions: 'setSearchResults',
        },
        onError: {
          target: 'idle',
          actions: 'setError',
        },
      },
    },
    uploading: {
      invoke: {
        src: 'uploadDocument',
        onDone: {
          target: 'idle',
          actions: 'addDocument',
        },
        onError: {
          target: 'idle',
          actions: 'setError',
        },
      },
    },
    batchProcessing: {
      invoke: {
        src: 'processBatch',
        onDone: {
          target: 'idle',
          actions: 'setBatchResults',
        },
        onError: {
          target: 'idle',
          actions: 'setError',
        },
      },
    },
  },
}, {
  actions: {
    setQuery: (context, event) => {
      context.currentQuery = event.query;
      context.selectedPracticeArea = event.practiceArea || '';
      context.selectedDocumentType = event.documentType || '';
    },
    setAuthenticated: (context) => {
      context.isAuthenticated = true;
      context.errorMessage = null;
    },
    setError: (context, event) => {
      context.errorMessage = event.data?.message || 'An error occurred';
      context.isProcessing = false;
    },
    setGPUStatus: (context, event) => {
      context.gpuAvailable = event.data.available;
    },
    setGPUUnavailable: (context) => {
      context.gpuAvailable = false;
    },
    setSearchResults: (context, event) => {
      context.searchResults = event.data.results;
      context.processingMetrics = event.data.processing_info;
      context.isProcessing = false;
      
      // Update cache stats
      if (event.data.processing_info?.cache_operations > 0) {
        context.cacheStats.hits += event.data.processing_info.cache_operations;
      } else {
        context.cacheStats.misses += 1;
      }
      context.cacheStats.total += 1;
    },
    addDocument: (context, event) => {
      context.documents.push(event.data);
      context.isProcessing = false;
    },
    setBatchResults: (context, event) => {
      context.batchQueue = [];
      context.isProcessing = false;
    },
  },
  services: {
    checkGPUAvailability: async () => {
      const response = await fetch(`${GPU_SERVICE_URLS.NEO4J_GPU_INTEGRATED}/health`);
      const health = await response.json();
      return { available: health.gpu_worker_available };
    },
    performSearch: async (context) => {
      const request: Neo4jSearchRequest = {
        query: context.currentQuery,
        practice_area: context.selectedPracticeArea,
        document_type: context.selectedDocumentType,
        max_results: 20,
        min_confidence: 0.1,
        use_gpu: context.gpuAvailable,
        use_fp16: true,
        use_cache: true,
        batch_optimization: true,
      };

      const response = await fetch(`${GPU_SERVICE_URLS.NEO4J_GPU_INTEGRATED}/search/enhanced`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`);
      }

      return await response.json();
    },
    uploadDocument: async (context, event) => {
      // Document upload logic
      return { success: true };
    },
    processBatch: async (context) => {
      const texts = context.batchQueue.map((item: any) => item.text);
      
      const request: GPUEmbeddingRequest = {
        texts,
        use_cache: true,
        normalize: true,
        fp16_precision: context.gpuAvailable,
      };

      const response = await fetch(`${GPU_SERVICE_URLS.NEO4J_GPU_INTEGRATED}/batch/embeddings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      });

      return await response.json();
    },
  },
});

// Reactive Stores for SvelteKit Integration
export const legalAIService = interpret(legalAIMachine).start();

export const legalAIState = writable(legalAIService.state);
export const legalAIContext = writable(legalAIService.getSnapshot().context);

// Update stores when machine state changes
legalAIService.subscribe((state) => {
  legalAIState.set(state);
  legalAIContext.set(state.context);
});

// Database instance
export const legalDB = new LegalAIDatabase();

// Reactive store for database stats
export const dbStats = writable(legalDB.getStats());

// Update stats periodically
setInterval(() => {
  dbStats.set(legalDB.getStats());
}, 30000); // Every 30 seconds

// Derived stores for computed values
export const isGPUAvailable = derived(legalAIContext, ($context) => $context.gpuAvailable);
export const isProcessing = derived(legalAIContext, ($context) => $context.isProcessing);
export const searchResults = derived(legalAIContext, ($context) => $context.searchResults);
export const cacheHitRatio = derived(legalAIContext, ($context) => {
  const { hits, total } = $context.cacheStats;
  return total > 0 ? (hits / total) * 100 : 0;
});

// High-level API functions
export class LegalAIGPUService {
  private static instance: LegalAIGPUService;
  
  static getInstance(): LegalAIGPUService {
    if (!LegalAIGPUService.instance) {
      LegalAIGPUService.instance = new LegalAIGPUService();
    }
    return LegalAIGPUService.instance;
  }

  // Generate embeddings with GPU acceleration
  async generateEmbeddings(
    texts: string[],
    options: Partial<GPUEmbeddingRequest> = {}
  ): Promise<GPUEmbeddingResponse> {
    const request: GPUEmbeddingRequest = {
      texts,
      use_cache: true,
      normalize: true,
      fp16_precision: get(isGPUAvailable),
      ...options,
    };

    // Check local cache first
    const cachedResults: EmbeddingVector[] = [];
    const uncachedTexts: string[] = [];

    texts.forEach((text) => {
      const textHash = this.hashString(text);
      const cached = legalDB.getCachedEmbedding(textHash);
      if (cached) {
        cachedResults.push(cached);
      } else {
        uncachedTexts.push(text);
        cachedResults.push(null as any);
      }
    });

    // Generate uncached embeddings
    if (uncachedTexts.length > 0) {
      const response = await fetch(`${GPU_SERVICE_URLS.NEO4J_GPU_INTEGRATED}/batch/embeddings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...request,
          texts: uncachedTexts,
        }),
      });

      if (!response.ok) {
        throw new Error(`Embedding generation failed: ${response.statusText}`);
      }

      const result: GPUEmbeddingResponse = await response.json();
      
      // Cache new embeddings and merge results
      let uncachedIndex = 0;
      for (let i = 0; i < cachedResults.length; i++) {
        if (cachedResults[i] === null) {
          const embedding = result.embeddings[uncachedIndex];
          cachedResults[i] = embedding;
          legalDB.cacheEmbedding(embedding.text_hash, embedding);
          uncachedIndex++;
        }
      }

      return {
        embeddings: cachedResults,
        metrics: result.metrics,
        success: true,
      };
    }

    return {
      embeddings: cachedResults,
      metrics: {
        processing_time_ms: 0,
        batch_size: texts.length,
        cache_hits: cachedResults.length,
        gpu_used: false,
        fp16_used: false,
        tensor_cores_used: false,
        method: 'CACHE',
      },
      success: true,
    };
  }

  // Perform enhanced Neo4j search with GPU acceleration
  async enhancedSearch(request: Neo4jSearchRequest): Promise<Neo4jSearchResponse> {
    const queryHash = this.hashString(JSON.stringify(request));
    
    // Check cache first
    const cached = legalDB.getCachedSearchResults(queryHash);
    if (cached && request.use_cache !== false) {
      return cached;
    }

    // Send search request
    legalAIService.send({ type: 'SEARCH', ...request });

    const response = await fetch(`${GPU_SERVICE_URLS.NEO4J_GPU_INTEGRATED}/search/enhanced`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`Search failed: ${response.statusText}`);
    }

    const result: Neo4jSearchResponse = await response.json();

    // Cache results and store documents for fuzzy search
    legalDB.cacheSearchResults(queryHash, result);
    result.results.forEach(doc => legalDB.storeDocument(doc));

    return result;
  }

  // Fuzzy search over local cache
  fuzzySearch(query: string, limit: number = 10) {
    return legalDB.fuzzySearch(query, limit);
  }

  // Get service health
  async getHealth() {
    const responses = await Promise.allSettled([
      fetch(`${GPU_SERVICE_URLS.NEO4J_SIMD}/health`),
      fetch(`${GPU_SERVICE_URLS.NEO4J_GPU_INTEGRATED}/health`),
    ]);

    return {
      simd_service: responses[0].status === 'fulfilled' && responses[0].value.ok,
      gpu_service: responses[1].status === 'fulfilled' && responses[1].value.ok,
      database_stats: legalDB.getStats(),
    };
  }

  // Get performance metrics
  async getMetrics() {
    const responses = await Promise.allSettled([
      fetch(`${GPU_SERVICE_URLS.NEO4J_SIMD}/metrics`),
      fetch(`${GPU_SERVICE_URLS.NEO4J_GPU_INTEGRATED}/metrics`),
    ]);

    const metrics = {
      simd_metrics: null,
      gpu_metrics: null,
      cache_stats: get(legalAIContext).cacheStats,
      local_db_stats: legalDB.getStats(),
    };

    if (responses[0].status === 'fulfilled' && responses[0].value.ok) {
      metrics.simd_metrics = await responses[0].value.json();
    }

    if (responses[1].status === 'fulfilled' && responses[1].value.ok) {
      metrics.gpu_metrics = await responses[1].value.json();
    }

    return metrics;
  }

  // Clear all caches
  async clearCache() {
    // Clear local Loki.js cache
    legalDB['embeddings'].clear();
    legalDB['searchResults'].clear();

    // Clear remote caches
    await Promise.allSettled([
      fetch(`${GPU_SERVICE_URLS.NEO4J_SIMD}/cache`, { method: 'DELETE' }),
      fetch(`${GPU_SERVICE_URLS.NEO4J_GPU_INTEGRATED}/cache`, { method: 'DELETE' }),
    ]);

    // Update stats
    dbStats.set(legalDB.getStats());
  }

  private hashString(str: string): string {
    let hash = 0;
    if (str.length === 0) return hash.toString();
    
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    
    return hash.toString();
  }
}

// Global service instance
export const gpuService = LegalAIGPUService.getInstance();

// Utility functions for SvelteKit integration
export const legalAIHelpers = {
  // Send search query
  search: (query: string, practiceArea: string, documentType: string) => {
    legalAIService.send({
      type: 'SEARCH',
      query,
      practiceArea,
      documentType,
    });
  },

  // Check GPU availability
  checkGPU: () => {
    legalAIService.send({ type: 'CHECK_GPU' });
  },

  // Add to batch queue
  addToBatch: (text: string) => {
    const currentContext = get(legalAIContext);
    currentContext.batchQueue.push({ text, timestamp: Date.now() });
    legalAIContext.set(currentContext);
  },

  // Process batch queue
  processBatch: () => {
    legalAIService.send({ type: 'BATCH_PROCESS' });
  },

  // Get current state
  getCurrentState: () => get(legalAIState),
  getCurrentContext: () => get(legalAIContext),
};

// Export for use in SvelteKit components
export default {
  // Services
  gpuService,
  legalDB,
  legalAIService,
  
  // Stores
  legalAIState,
  legalAIContext,
  dbStats,
  isGPUAvailable,
  isProcessing,
  searchResults,
  cacheHitRatio,
  
  // Helpers
  legalAIHelpers,
  
  // Classes
  LegalAIDatabase,
  LegalAIGPUService,
  
  // Constants
  GPU_SERVICE_URLS,
};