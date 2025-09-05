// @ts-nocheck
/**
 * Enhanced BVector Store
 * Multi-layered GPU cache system for contextual prompting and reinforcement learning
 * Integrates with Go binaries, embedding workers, and SOM RAG system
 */

import { EmbeddingWorkerManager } from '../workers/embedding-worker';
import { createSOMRAGSystem, type SelfOrganizingMapRAG } from '../ai/som-rag-system';
import { createEnhancedNeo4jReranker, type EnhancedNeo4jReranker } from '../ai/enhanced-neo4j-reranker';
import { db } from '../server/db/schema-postgres';
import { aiHistory, users, cases } from '../server/db/schema-postgres';
import { eq, and, sql, desc } from 'drizzle-orm';
import Redis from 'ioredis';

export interface BVectorConfig {
  // Go binary integration
  goBinaries: {
    vectorService: string;     // vector-service.exe
    cudaService: string;       // cuda-ai-service.exe  
    enhancedRAG: string;       // enhanced-rag.exe
    gpuOrchestrator: string;   // gpu-orchestrator.exe
  };
  
  // GPU cache configuration
  gpuCache: {
    layers: number;            // Multi-layer cache depth
    maxMemoryMB: number;       // RTX 3060 Ti memory limit
    batchSize: number;         // GPU batch processing size
    enableQuantization: boolean;
  };
  
  // Embedding configuration
  embedding: {
    dimensions: number;        // 384 for nomic-embed-text
    model: string;
    workerThreads: number;
    batchSize: number;
  };
  
  // Reinforcement learning
  reinforcementLearning: {
    enabled: boolean;
    learningRate: number;
    decayFactor: number;
    feedbackThreshold: number;
  };
  
  // Storage configuration
  storage: {
    redis: {
      host: string;
      port: number;
      password?: string;
    };
    postgresql: {
      enabled: boolean;
      tableName: string;
    };
    neo4j: {
      enabled: boolean;
      uri: string;
    };
  };
}

export interface BVectorEntry {
  id: string;
  content: string;
  embedding: number[];
  metadata: {
    userId: string;
    caseId?: string;
    timestamp: number;
    conversationId?: string;
    userRole: 'prosecutor' | 'detective' | 'admin';
    intent?: string;
    confidence: number;
    
    // Legal context
    legalDomain?: string;
    jurisdiction?: string;
    caseType?: string;
    
    // Reinforcement learning
    rlMetrics?: {
      userSatisfaction?: number;
      responseQuality?: number;
      followUpSuccess?: boolean;
      contextRelevance?: number;
      correctionsMade?: string[];
    };
  };
}

export interface SearchResult extends BVectorEntry {
  similarity: number;
  rank: number;
  contextualBoost: number;
  rlWeight: number;
  explanation: string;
}

export interface GPUCacheLayer {
  level: number;
  name: string;
  capacity: number;
  currentSize: number;
  hitRate: number;
  avgAccessTime: number;
}

export class EnhancedBVectorStore {
  private config: BVectorConfig;
  private embeddingWorker: EmbeddingWorkerManager;
  private somRAG: SelfOrganizingMapRAG;
  private neo4jReranker: EnhancedNeo4jReranker;
  private redis: Redis;
  
  // GPU cache layers
  private gpuCacheLayers: Map<string, GPUCacheLayer> = new Map();
  private gpuMemoryUsage = 0;
  
  // Reinforcement learning state
  private userPreferences: Map<string, Record<string, number>> = new Map();
  private contextualWeights: Map<string, number[]> = new Map();
  
  // Performance metrics
  private metrics = {
    totalQueries: 0,
    cacheHits: 0,
    gpuAccelerated: 0,
    averageLatency: 0,
    reinforcementUpdates: 0
  };

  constructor(config: BVectorConfig) {
    this.config = config;
    this.initialize();
  }

  private async initialize(): Promise<void> {
    console.log('🚀 Initializing Enhanced BVector Store...');

    // Initialize embedding worker
    this.embeddingWorker = new EmbeddingWorkerManager();

    // Initialize SOM RAG system
    this.somRAG = createSOMRAGSystem({
      mapWidth: 20,
      mapHeight: 20,
      dimensions: this.config.embedding.dimensions,
      clusterCount: 12
    });

    // Initialize Neo4j reranker
    this.neo4jReranker = createEnhancedNeo4jReranker({
      enable_neo4j_paths: true,
      enable_boolean_patterns: true,
      accuracy_threshold: 0.95,
      legal_weight_multiplier: 1.5
    });

    // Initialize Redis
    this.redis = new Redis({
      host: this.config.storage.redis.host,
      port: this.config.storage.redis.port,
      password: this.config.storage.redis.password,
      retryDelayOnFailover: 100,
      maxRetriesPerRequest: 3
    });

    // Initialize GPU cache layers
    await this.initializeGPUCache();

    // Start Go binary health monitoring
    await this.startGoBinaryMonitoring();

    console.log('✅ Enhanced BVector Store initialized');
  }

  /**
   * Initialize multi-layered GPU cache system
   */
  private async initializeGPUCache(): Promise<void> {
    const layers = [
      { level: 1, name: 'L1_FREQUENT', capacity: 1000, avgAccessTime: 0.1 },
      { level: 2, name: 'L2_RECENT', capacity: 5000, avgAccessTime: 0.5 },
      { level: 3, name: 'L3_CONTEXTUAL', capacity: 10000, avgAccessTime: 1.0 },
      { level: 4, name: 'L4_ARCHIVAL', capacity: 50000, avgAccessTime: 2.0 }
    ];

    for (const layer of layers) {
      this.gpuCacheLayers.set(layer.name, {
        ...layer,
        currentSize: 0,
        hitRate: 0
      });
    }

    console.log('🎮 GPU cache layers initialized:', this.gpuCacheLayers.size);
  }

  /**
   * Start monitoring Go binary services
   */
  private async startGoBinaryMonitoring(): Promise<void> {
    const services = Object.entries(this.config.goBinaries);
    
    setInterval(async () => {
      for (const [name, binary] of services) {
        try {
          const response = await fetch(`http://localhost:${this.getServicePort(name)}/health`);
          const isHealthy = response.ok;
          
          if (!isHealthy) {
            console.warn(`⚠️ Go service ${name} (${binary}) is unhealthy`);
          }
        } catch (error) {
          console.error(`❌ Go service ${name} connection failed:`, error);
        }
      }
    }, 30000); // Check every 30 seconds
  }

  /**
   * Store vector with contextual metadata and RL tracking
   */
  async store(entry: BVectorEntry): Promise<void> {
    try {
      // Generate embedding if not provided
      if (!entry.embedding || entry.embedding.length === 0) {
        entry.embedding = await this.generateEmbedding(entry.content, entry.metadata);
      }

      // Store in multiple layers based on importance and recency
      await Promise.all([
        this.storeInPostgreSQL(entry),
        this.storeInRedis(entry),
        this.storeInGPUCache(entry),
        this.storeInSOMRAG(entry)
      ]);

      // Update reinforcement learning weights if metrics available
      if (entry.metadata.rlMetrics) {
        await this.updateReinforcementWeights(entry);
      }

      console.log(`📦 Stored vector ${entry.id} in all layers`);
    } catch (error) {
      console.error('Failed to store vector:', error);
      throw error;
    }
  }

  /**
   * Enhanced contextual search with multi-layer caching and RL
   */
  async search(
    query: string,
    options: {
      userId: string;
      caseId?: string;
      userRole: 'prosecutor' | 'detective' | 'admin';
      limit?: number;
      threshold?: number;
      useReinforcementLearning?: boolean;
      enableGPUAcceleration?: boolean;
      contextualPrompting?: boolean;
    }
  ): Promise<SearchResult[]> {
    const startTime = Date.now();
    this.metrics.totalQueries++;

    try {
      // Generate query embedding
      const queryEmbedding = await this.generateEmbedding(query, {
        userId: options.userId,
        userRole: options.userRole,
        caseId: options.caseId
      });

      // Multi-layer search with caching
      let results = await this.performMultiLayerSearch(
        queryEmbedding,
        query,
        options
      );

      // Apply SOM clustering and contextual enhancement
      if (options.contextualPrompting) {
        results = await this.enhanceWithSOMContext(results, query, options);
      }

      // Apply Neo4j reranking with legal context
      results = await this.applyNeo4jReranking(results, query, options);

      // Apply reinforcement learning weights
      if (options.useReinforcementLearning) {
        results = await this.applyReinforcementWeights(results, options.userId);
      }

      // Final ranking and truncation
      results = results
        .sort((a, b) => (b.similarity + b.contextualBoost + b.rlWeight) - (a.similarity + a.contextualBoost + a.rlWeight))
        .slice(0, options.limit || 10);

      // Update performance metrics
      const latency = Date.now() - startTime;
      this.updateMetrics(latency, results.length > 0);

      console.log(`🔍 Search completed: ${results.length} results in ${latency}ms`);
      return results;
    } catch (error) {
      console.error('Search failed:', error);
      return [];
    }
  }

  /**
   * Multi-layer search with GPU acceleration
   */
  private async performMultiLayerSearch(
    queryEmbedding: number[],
    query: string,
    options: any
  ): Promise<SearchResult[]> {
    const searchTasks = [];

    // L1: GPU Cache - Frequent items
    searchTasks.push(this.searchGPUCache('L1_FREQUENT', queryEmbedding, options));

    // L2: Redis Cache - Recent items  
    searchTasks.push(this.searchRedisCache(queryEmbedding, options));

    // L3: PostgreSQL - Persistent storage with pgvector
    searchTasks.push(this.searchPostgreSQL(queryEmbedding, options));

    // Execute searches in parallel
    const layerResults = await Promise.allSettled(searchTasks);
    
    // Combine and deduplicate results
    const allResults: SearchResult[] = [];
    const seenIds = new Set<string>();

    for (const result of layerResults) {
      if (result.status === 'fulfilled' && result.value) {
        for (const item of result.value) {
          if (!seenIds.has(item.id)) {
            seenIds.add(item.id);
            allResults.push(item);
          }
        }
      }
    }

    return allResults;
  }

  /**
   * Search GPU cache layer
   */
  private async searchGPUCache(
    layer: string,
    queryEmbedding: number[],
    options: any
  ): Promise<SearchResult[]> {
    const cacheLayer = this.gpuCacheLayers.get(layer);
    if (!cacheLayer) return [];

    try {
      // Simulate GPU-accelerated similarity search
      // In production, this would use CUDA or WebGPU
      const cacheKey = `gpu_cache:${layer}:search`;
      const cachedData = await this.redis.get(cacheKey);
      
      if (cachedData) {
        this.metrics.cacheHits++;
        cacheLayer.hitRate = (cacheLayer.hitRate + 1) / 2; // Moving average
        
        const entries = JSON.parse(cachedData);
        return this.calculateSimilarities(entries, queryEmbedding, options);
      }

      return [];
    } catch (error) {
      console.warn(`GPU cache search failed for ${layer}:`, error);
      return [];
    }
  }

  /**
   * Search Redis cache with compression
   */
  private async searchRedisCache(
    queryEmbedding: number[],
    options: any
  ): Promise<SearchResult[]> {
    try {
      const pattern = `bvector:user:${options.userId}:*`;
      const keys = await this.redis.keys(pattern);
      
      if (keys.length === 0) return [];

      const pipeline = this.redis.pipeline();
      keys.forEach(key => pipeline.get(key));
      const results = await pipeline.exec();

      const entries: BVectorEntry[] = [];
      if (results) {
        for (const result of results) {
          if (result && result[1]) {
            try {
              entries.push(JSON.parse(result[1] as string));
            } catch (e) {
              console.warn('Failed to parse Redis cache entry');
            }
          }
        }
      }

      return this.calculateSimilarities(entries, queryEmbedding, options);
    } catch (error) {
      console.warn('Redis cache search failed:', error);
      return [];
    }
  }

  /**
   * Search PostgreSQL with pgvector
   */
  private async searchPostgreSQL(
    queryEmbedding: number[],
    options: any
  ): Promise<SearchResult[]> {
    try {
      const embeddingString = `[${queryEmbedding.join(',')}]`;
      
      const results = await db
        .select({
          id: aiHistory.id,
          prompt: aiHistory.prompt,
          response: aiHistory.response,
          embedding: aiHistory.embedding,
          userId: aiHistory.userId,
          caseId: aiHistory.caseId,
          metadata: aiHistory.metadata,
          similarity: sql<number>`1 - (embedding <-> ${embeddingString}::vector)`,
          createdAt: aiHistory.createdAt
        })
        .from(aiHistory)
        .where(
          and(
            eq(aiHistory.userId, options.userId),
            options.caseId ? eq(aiHistory.caseId, options.caseId) : undefined,
            sql`1 - (embedding <-> ${embeddingString}::vector) > ${options.threshold || 0.7}`
          )
        )
        .orderBy(sql`embedding <-> ${embeddingString}::vector`)
        .limit(options.limit || 50);

      return results.map((result): SearchResult => ({
        id: result.id.toString(),
        content: result.prompt,
        embedding: result.embedding as number[],
        metadata: {
          userId: result.userId || '',
          caseId: result.caseId || undefined,
          timestamp: result.createdAt ? result.createdAt.getTime() : Date.now(),
          userRole: 'admin' as const,
          confidence: result.similarity,
          ...((result.metadata as any) || {})
        },
        similarity: result.similarity,
        rank: 0,
        contextualBoost: 0,
        rlWeight: 0,
        explanation: 'PostgreSQL pgvector search result'
      }));
    } catch (error) {
      console.warn('PostgreSQL search failed:', error);
      return [];
    }
  }

  /**
   * Generate embedding using worker or Go binary
   */
  private async generateEmbedding(
    content: string,
    metadata: any
  ): Promise<number[]> {
    try {
      // Try Go CUDA service first for GPU acceleration
      if (this.config.goBinaries.cudaService && this.gpuMemoryUsage < this.config.gpuCache.maxMemoryMB * 0.8) {
        const embedding = await this.generateEmbeddingGoCUDA(content, metadata);
        if (embedding) {
          this.metrics.gpuAccelerated++;
          return embedding;
        }
      }

      // Fallback to embedding worker
      const result = await this.embeddingWorker.processEmbeddings({
        texts: [content],
        batchSize: 1,
        model: this.config.embedding.model,
        dimensions: this.config.embedding.dimensions
      });

      if (result.results && result.results.length > 0) {
        return result.results[0].embedding;
      }

      throw new Error('No embedding generated');
    } catch (error) {
      console.error('Embedding generation failed:', error);
      // Return zero vector as fallback
      return new Array(this.config.embedding.dimensions).fill(0);
    }
  }

  /**
   * Generate embedding using Go CUDA service
   */
  private async generateEmbeddingGoCUDA(
    content: string,
    metadata: any
  ): Promise<number[] | null> {
    try {
      const port = this.getServicePort('cudaService');
      const response = await fetch(`http://localhost:${port}/api/embed`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: content,
          model: this.config.embedding.model,
          legal_mode: true,
          user_context: metadata
        })
      });

      if (response.ok) {
        const data = await response.json();
        return data.embedding;
      }

      return null;
    } catch (error) {
      console.warn('Go CUDA service embedding failed:', error);
      return null;
    }
  }

  /**
   * Calculate similarities with contextual boosting
   */
  private calculateSimilarities(
    entries: BVectorEntry[],
    queryEmbedding: number[],
    options: any
  ): SearchResult[] {
    return entries.map((entry, index): SearchResult => {
      const similarity = this.cosineSimilarity(queryEmbedding, entry.embedding);
      
      return {
        ...entry,
        similarity,
        rank: index,
        contextualBoost: 0,
        rlWeight: 0,
        explanation: 'Cosine similarity calculation'
      };
    });
  }

  /**
   * Enhance results with SOM contextual clustering
   */
  private async enhanceWithSOMContext(
    results: SearchResult[],
    query: string,
    options: any
  ): Promise<SearchResult[]> {
    try {
      // Convert results to SOM document format
      const documents = results.map(r => ({
        id: r.id,
        content: r.content,
        embedding: r.embedding,
        metadata: {
          ...r.metadata,
          timestamp: r.metadata.timestamp
        }
      }));

      // Get SOM-enhanced results
      const somResults = await this.somRAG.semanticSearch(
        query,
        options.queryEmbedding || [],
        results.length
      );

      // Merge SOM insights with original results
      return results.map(result => {
        const somMatch = somResults.find(s => s.id === result.id);
        return {
          ...result,
          contextualBoost: somMatch ? 0.2 : 0,
          explanation: result.explanation + (somMatch ? ' + SOM contextual enhancement' : '')
        };
      });
    } catch (error) {
      console.warn('SOM context enhancement failed:', error);
      return results;
    }
  }

  /**
   * Apply Neo4j legal reranking
   */
  private async applyNeo4jReranking(
    results: SearchResult[],
    query: string,
    options: any
  ): Promise<SearchResult[]> {
    try {
      const documents = results.map(r => ({
        id: r.id,
        content: r.content,
        embedding: r.embedding,
        metadata: r.metadata
      }));

      const rerankedResults = await this.neo4jReranker.enhancedRerank(
        query,
        documents,
        {
          user_id: options.userId,
          case_id: options.caseId,
          role: options.userRole,
          search_intent: 'evidence'
        }
      );

      // Apply reranking scores
      return results.map(result => {
        const reranked = rerankedResults.find(r => r.document_id === result.id);
        if (reranked) {
          return {
            ...result,
            similarity: reranked.enhanced_score,
            contextualBoost: result.contextualBoost + reranked.neo4j_boost,
            explanation: result.explanation + ` + ${reranked.explanation}`
          };
        }
        return result;
      });
    } catch (error) {
      console.warn('Neo4j reranking failed:', error);
      return results;
    }
  }

  /**
   * Apply reinforcement learning weights
   */
  private async applyReinforcementWeights(
    results: SearchResult[],
    userId: string
  ): Promise<SearchResult[]> {
    const userPrefs = this.userPreferences.get(userId) || {};
    
    return results.map(result => {
      const rlWeight = this.calculateRLWeight(result, userPrefs);
      return {
        ...result,
        rlWeight,
        explanation: result.explanation + ` + RL weight: ${rlWeight.toFixed(3)}`
      };
    });
  }

  /**
   * Store in various layers
   */
  private async storeInPostgreSQL(entry: BVectorEntry): Promise<void> {
    if (!this.config.storage.postgresql.enabled) return;

    try {
      await db.insert(aiHistory).values({
        prompt: entry.content,
        response: '', // Will be filled by chat system
        embedding: entry.embedding,
        userId: entry.metadata.userId,
        caseId: entry.metadata.caseId,
        metadata: entry.metadata as any,
        createdAt: new Date(entry.metadata.timestamp)
      });
    } catch (error) {
      console.warn('PostgreSQL storage failed:', error);
    }
  }

  private async storeInRedis(entry: BVectorEntry): Promise<void> {
    const key = `bvector:user:${entry.metadata.userId}:${entry.id}`;
    await this.redis.setex(key, 86400, JSON.stringify(entry)); // 24 hour TTL
  }

  private async storeInGPUCache(entry: BVectorEntry): Promise<void> {
    // Determine appropriate cache layer based on metadata
    const layer = this.selectGPUCacheLayer(entry);
    const cacheLayer = this.gpuCacheLayers.get(layer);
    
    if (cacheLayer && cacheLayer.currentSize < cacheLayer.capacity) {
      const key = `gpu_cache:${layer}:${entry.id}`;
      await this.redis.setex(key, 3600, JSON.stringify(entry));
      cacheLayer.currentSize++;
    }
  }

  private async storeInSOMRAG(entry: BVectorEntry): Promise<void> {
    // Add to SOM training data
    const documents = [{
      id: entry.id,
      content: entry.content,
      embedding: entry.embedding,
      metadata: entry.metadata
    }];

    // Periodic SOM retraining
    if (Math.random() < 0.01) { // 1% chance
      await this.somRAG.trainSOM(documents);
    }
  }

  /**
   * Helper methods
   */
  private cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) return 0;
    
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    
    const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
    return magnitude > 0 ? dotProduct / magnitude : 0;
  }

  private selectGPUCacheLayer(entry: BVectorEntry): string {
    // Determine cache layer based on importance and recency
    const age = Date.now() - entry.metadata.timestamp;
    const confidence = entry.metadata.confidence || 0;
    
    if (confidence > 0.9 && age < 3600000) return 'L1_FREQUENT'; // 1 hour
    if (confidence > 0.7 && age < 86400000) return 'L2_RECENT'; // 1 day
    if (age < 604800000) return 'L3_CONTEXTUAL'; // 1 week
    return 'L4_ARCHIVAL';
  }

  private calculateRLWeight(result: SearchResult, userPrefs: Record<string, number>): number {
    let weight = 0;
    
    // Apply learned user preferences
    Object.entries(userPrefs).forEach(([key, value]) => {
      if (result.metadata[key as keyof typeof result.metadata]) {
        weight += value * 0.1;
      }
    });
    
    return Math.max(0, Math.min(1, weight));
  }

  private async updateReinforcementWeights(entry: BVectorEntry): Promise<void> {
    if (!this.config.reinforcementLearning.enabled || !entry.metadata.rlMetrics) return;

    const userId = entry.metadata.userId;
    const userPrefs = this.userPreferences.get(userId) || {};
    
    // Update preferences based on feedback
    if (entry.metadata.rlMetrics.userSatisfaction && entry.metadata.rlMetrics.userSatisfaction > 4) {
      // Positive feedback - boost similar patterns
      Object.keys(entry.metadata).forEach(key => {
        if (key !== 'rlMetrics') {
          userPrefs[key] = (userPrefs[key] || 0) + this.config.reinforcementLearning.learningRate;
        }
      });
    }

    this.userPreferences.set(userId, userPrefs);
    this.metrics.reinforcementUpdates++;
  }

  private getServicePort(serviceName: string): number {
    const portMap: Record<string, number> = {
      vectorService: 8091,
      cudaService: 8090,
      enhancedRAG: 8094,
      gpuOrchestrator: 8089
    };
    return portMap[serviceName] || 8080;
  }

  private updateMetrics(latency: number, successful: boolean): void {
    this.metrics.averageLatency = (this.metrics.averageLatency + latency) / 2;
    if (successful) {
      this.metrics.cacheHits++;
    }
  }

  /**
   * Public API methods
   */
  async getMetrics(): Promise<typeof this.metrics & { gpuCacheLayers: GPUCacheLayer[] }> {
    return {
      ...this.metrics,
      gpuCacheLayers: Array.from(this.gpuCacheLayers.values())
    };
  }

  async optimizeMemory(): Promise<void> {
    // Clear old entries from GPU cache
    for (const [name, layer] of this.gpuCacheLayers) {
      if (layer.currentSize > layer.capacity * 0.8) {
        const pattern = `gpu_cache:${name}:*`;
        const keys = await this.redis.keys(pattern);
        if (keys.length > 0) {
          await this.redis.del(...keys.slice(0, Math.floor(keys.length * 0.3)));
          layer.currentSize *= 0.7;
        }
      }
    }

    console.log('🧹 Memory optimization completed');
  }

  async trainUserModel(userId: string, conversationHistory: BVectorEntry[]): Promise<void> {
    if (!this.config.reinforcementLearning.enabled) return;

    // Train personalized model based on conversation patterns
    const documents = conversationHistory.map(entry => ({
      id: entry.id,
      content: entry.content,
      embedding: entry.embedding,
      metadata: entry.metadata
    }));

    await this.somRAG.trainSOM(documents);
    console.log(`🎯 User model trained for ${userId} with ${documents.length} conversations`);
  }

  async close(): Promise<void> {
    this.embeddingWorker.terminate();
    await this.redis.quit();
    console.log('🔒 Enhanced BVector Store closed');
  }
}

// Factory function
export function createEnhancedBVectorStore(config: Partial<BVectorConfig> = {}): EnhancedBVectorStore {
  const defaultConfig: BVectorConfig = {
    goBinaries: {
      vectorService: 'vector-service.exe',
      cudaService: 'cuda-ai-service.exe',
      enhancedRAG: 'enhanced-rag.exe',
      gpuOrchestrator: 'gpu-orchestrator.exe'
    },
    gpuCache: {
      layers: 4,
      maxMemoryMB: 6144, // RTX 3060 Ti limit
      batchSize: 32,
      enableQuantization: true
    },
    embedding: {
      dimensions: 384,
      model: 'nomic-embed-text',
      workerThreads: 4,
      batchSize: 16
    },
    reinforcementLearning: {
      enabled: true,
      learningRate: 0.01,
      decayFactor: 0.95,
      feedbackThreshold: 3.5
    },
    storage: {
      redis: {
        host: 'localhost',
        port: 6379
      },
      postgresql: {
        enabled: true,
        tableName: 'ai_history'
      },
      neo4j: {
        enabled: true,
        uri: 'neo4j://localhost:7687'
      }
    }
  };

  const finalConfig = { ...defaultConfig, ...config };
  return new EnhancedBVectorStore(finalConfig);
}

export default EnhancedBVectorStore;