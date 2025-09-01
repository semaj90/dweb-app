
/**
 * Advanced Memory Optimizer with LOD, k-means, SOM, and multi-layer caching
 * Optimizes memory usage across the entire legal AI system
 */

import { Worker } from "worker_threads";
import { SelfOrganizingMapRAG } from '../ai/som-rag-system';
// Docker dependency removed - using native memory optimization
import crypto from "crypto";
import {
  SIMDJSONParser,
  type ParsedLegalDocument,
} from '../parsers/simd-json-parser';

// Initialize SIMD parser instance
const simdParser = new SIMDJSONParser({
  batchSize: 2048,
  enableSIMD: true,
  memoryLimit: 512 * 1024 * 1024, // 512MB
  parallelChunks: 4,
  validateStructure: true,
});

// Native memory optimizer mock (Docker dependency removed)
class DockerResourceOptimizer {
  constructor(config?: any) {
    // Mock implementation - no actual Docker optimization
  }

  async cacheWithCompression(key: string, data: any) {
    // Simple compression mock
    const jsonData = JSON.stringify(data);
    return {
      key,
      data: jsonData,
      compressed: true,
      size: jsonData.length * 0.7 // Simulate 30% compression
    };
  }

  dispose() {
    // Mock cleanup
  }
}

export interface LODLevel {
  id: string;
  detail: "low" | "medium" | "high" | "ultra";
  maxMemoryMB: number;
  maxObjects: number;
  quality: number; // 0-1
  compressionRatio: number;
}

export interface ClusterMetrics {
  id: string;
  centroid: number[];
  size: number;
  cohesion: number;
  separability: number;
  memoryUsage: number;
  processingTime: number;
}

export interface MemoryPool {
  id: string;
  type: "embedding" | "vector" | "cache" | "som" | "cluster";
  current: number;
  max: number;
  items: Map<string, any>;
  lastAccessed: number;
  priority: number;
}

export interface CacheLayer {
  name: string;
  type:
    | "loki"
    | "redis"
    | "qdrant"
    | "postgres"
    | "neo4j"
    | "rabbitmq"
    | "memory";
  size: number;
  hitRate: number;
  avgResponseTime: number;
  ttl: number;
  priority: number;
  enabled: boolean;
}

export class AdvancedMemoryOptimizer {
  private memoryPools = new Map<string, MemoryPool>();
  private clusters = new Map<string, ClusterMetrics>();
  private cacheLayers = new Map<string, CacheLayer>();
  private lodLevels: LODLevel[] = [];
  private somNetwork: SelfOrganizingMapRAG | null = null;
  private dockerOptimizer: DockerResourceOptimizer;
  private currentLOD: LODLevel;
  private memoryPressure = 0;
  private optimizationHistory: any[] = [];
  private workerPool: Map<string, Worker> = new Map();
  private maxWorkers = 4;

  constructor() {
    this.dockerOptimizer = new DockerResourceOptimizer();

    this.initializeLODLevels();
    this.initializeCacheLayers();
    this.initializeMemoryPools();
    this.initializeWorkerPool();
    this.currentLOD = this.lodLevels[1]; // Start with medium
    this.startMemoryMonitoring();
  }

  /**
   * Initialize Level of Detail configurations
   */
  private initializeLODLevels(): void {
    this.lodLevels = [
      {
        id: "low",
        detail: "low",
        maxMemoryMB: 512,
        maxObjects: 1000,
        quality: 0.3,
        compressionRatio: 0.1,
      },
      {
        id: "medium",
        detail: "medium",
        maxMemoryMB: 1024,
        maxObjects: 5000,
        quality: 0.6,
        compressionRatio: 0.4,
      },
      {
        id: "high",
        detail: "high",
        maxMemoryMB: 2048,
        maxObjects: 10000,
        quality: 0.8,
        compressionRatio: 0.7,
      },
      {
        id: "ultra",
        detail: "ultra",
        maxMemoryMB: 4096,
        maxObjects: 25000,
        quality: 1.0,
        compressionRatio: 1.0,
      },
    ];
  }

  /**
   * Initialize cache layers with priorities
   */
  private initializeCacheLayers(): void {
    const layers: CacheLayer[] = [
      {
        name: "memory",
        type: "memory",
        size: 0,
        hitRate: 0,
        avgResponseTime: 1,
        ttl: 300,
        priority: 1,
        enabled: true,
      },
      {
        name: "loki",
        type: "loki",
        size: 0,
        hitRate: 0,
        avgResponseTime: 5,
        ttl: 300,
        priority: 2,
        enabled: true,
      },
      {
        name: "redis",
        type: "redis",
        size: 0,
        hitRate: 0,
        avgResponseTime: 10,
        ttl: 3600,
        priority: 3,
        enabled: true,
      },
      {
        name: "qdrant",
        type: "qdrant",
        size: 0,
        hitRate: 0,
        avgResponseTime: 25,
        ttl: 7200,
        priority: 4,
        enabled: true,
      },
      {
        name: "postgres",
        type: "postgres",
        size: 0,
        hitRate: 0,
        avgResponseTime: 50,
        ttl: 86400,
        priority: 5,
        enabled: true,
      },
      {
        name: "neo4j",
        type: "neo4j",
        size: 0,
        hitRate: 0,
        avgResponseTime: 75,
        ttl: 43200,
        priority: 6,
        enabled: true,
      },
    ];

    layers.forEach((layer) => {
      this.cacheLayers.set(layer.name, layer);
    });
  }

  /**
   * Initialize memory pools for different data types
   */
  private initializeMemoryPools(): void {
    const pools: MemoryPool[] = [
      {
        id: "embeddings",
        type: "embedding",
        current: 0,
        max: 512 * 1024 * 1024, // 512MB
        items: new Map(),
        lastAccessed: Date.now(),
        priority: 1,
      },
      {
        id: "vectors",
        type: "vector",
        current: 0,
        max: 256 * 1024 * 1024, // 256MB
        items: new Map(),
        lastAccessed: Date.now(),
        priority: 2,
      },
      {
        id: "cache",
        type: "cache",
        current: 0,
        max: 1024 * 1024 * 1024, // 1GB
        items: new Map(),
        lastAccessed: Date.now(),
        priority: 3,
      },
      {
        id: "som",
        type: "som",
        current: 0,
        max: 128 * 1024 * 1024, // 128MB
        items: new Map(),
        lastAccessed: Date.now(),
        priority: 4,
      },
      {
        id: "clusters",
        type: "cluster",
        current: 0,
        max: 64 * 1024 * 1024, // 64MB
        items: new Map(),
        lastAccessed: Date.now(),
        priority: 5,
      },
    ];

    pools.forEach((pool) => {
      this.memoryPools.set(pool.id, pool);
    });
  }

  /**
   * Initialize worker thread pool for CPU-intensive operations
   */
  private initializeWorkerPool(): void {
    console.log(`🧵 Initializing worker pool with ${this.maxWorkers} workers`);

    // Don't create workers immediately - create them on demand
    // This helps with memory usage and allows for better error handling
  }

  /**
   * Get or create a worker for k-means clustering
   */
  private async getKMeansWorker(): Promise<Worker> {
    const workerId = "kmeans-worker";

    if (this.workerPool.has(workerId)) {
      return this.workerPool.get(workerId)!;
    }

    // Create new worker
    const workerPath = new URL("../workers/kmeans-worker.js", import.meta.url);
    const worker = new Worker(workerPath);

    // Set up error handling
    worker.on("error", (error) => {
      console.error("🚨 K-means worker error:", error);
      this.workerPool.delete(workerId);
    });

    worker.on("exit", (code) => {
      if (code !== 0) {
        console.warn(`⚠️ K-means worker exited with code ${code}`);
      }
      this.workerPool.delete(workerId);
    });

    this.workerPool.set(workerId, worker);
    return worker;
  }

  /**
   * Intelligent cache layer selection based on data type and access patterns
   */
  async selectOptimalCacheLayer(
    key: string,
    dataType: string,
    size: number,
    accessFrequency: number
  ): Promise<CacheLayer[]> {
    const enabledLayers = Array.from(this.cacheLayers.values())
      .filter((layer) => layer.enabled)
      .sort((a, b) => a.priority - b.priority);

    // Score each layer based on multiple factors
    const scoredLayers = enabledLayers.map((layer) => {
      let score = 0;

      // Size factor (prefer layers with appropriate capacity)
      if (size < 1024)
        score += 10; // Small data - prefer fast layers
      else if (size < 1024 * 1024) score += layer.priority <= 3 ? 8 : 4;
      else score += layer.priority >= 4 ? 8 : 2; // Large data - prefer persistent layers

      // Access frequency factor
      if (accessFrequency > 0.8) score += layer.priority <= 2 ? 10 : 5;
      else if (accessFrequency > 0.5) score += layer.priority <= 4 ? 8 : 6;
      else score += layer.priority >= 4 ? 10 : 3; // Low frequency - prefer slower but persistent

      // Hit rate factor
      score += layer.hitRate * 10;

      // Response time factor (inverse)
      score += Math.max(0, 100 - layer.avgResponseTime);

      return { layer, score };
    });

    // Return top 3 layers
    return scoredLayers
      .sort((a, b) => b.score - a.score)
      .slice(0, 3)
      .map((item) => item.layer);
  }

  /**
   * Adaptive LOD management based on memory pressure
   */
  async adaptiveLODManagement(): Promise<void> {
    const currentMemory = await this.getCurrentMemoryUsage();
    const maxMemory = this.currentLOD.maxMemoryMB * 1024 * 1024;
    this.memoryPressure = currentMemory / maxMemory;

    console.log(
      `🧠 Memory pressure: ${(this.memoryPressure * 100).toFixed(1)}%`
    );

    if (this.memoryPressure > 0.9) {
      // High pressure - reduce LOD
      await this.reduceLOD();
    } else if (
      this.memoryPressure < 0.5 &&
      this.currentLOD.detail !== "ultra"
    ) {
      // Low pressure - potentially increase LOD
      await this.increaseLOD();
    }

    // Adjust object limits based on pressure
    await this.adjustObjectLimits();
  }

  /**
   * K-means clustering using worker threads for CPU-intensive operations
   */
  async performKMeansClustering(
    data: any[],
    k: number = 5
  ): Promise<ClusterMetrics[]> {
    console.log(
      `🔄 Performing k-means clustering (k=${k}) on ${data.length} items using worker threads...`
    );

    const startTime = Date.now();

    // Determine if we should use worker threads
    const useWorkerThread = data.length > 1000 && this.enableWorkerThreads();

    if (useWorkerThread) {
      return this.performKMeansWithWorker(data, k);
    } else {
      // Use original in-process clustering for smaller datasets
      return this.performKMeansInProcess(data, k);
    }
  }

  /**
   * Perform k-means clustering using worker thread
   */
  private async performKMeansWithWorker(
    data: any[],
    k: number
  ): Promise<ClusterMetrics[]> {
    const worker = await this.getKMeansWorker();

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error("K-means clustering timeout"));
      }, 300000); // 5 minute timeout

      // Set up progress monitoring
      const progressHandler = (message: any) => {
        if (message.type === "progress") {
          console.log(`📊 K-means progress: iteration ${message.iteration}`);
        }
      };

      // Set up result handler
      const messageHandler = (message: any) => {
        try {
          if (message.type === "result") {
            clearTimeout(timeout);
            worker.off("message", messageHandler);
            worker.off("message", progressHandler);

            // Store cluster results
            message.clusters.forEach((cluster: ClusterMetrics) => {
              this.clusters.set(cluster.id, cluster);
            });

            console.log(
              `✅ Worker k-means clustering completed in ${message.processingTime}ms`
            );
            resolve(message.clusters);
          } else if (message.type === "error") {
            clearTimeout(timeout);
            worker.off("message", messageHandler);
            worker.off("message", progressHandler);
            reject(new Error(`Worker error: ${message.error}`));
          }
        } catch (error: any) {
          clearTimeout(timeout);
          reject(error);
        }
      };

      worker.on("message", progressHandler);
      worker.on("message", messageHandler);

      // Send clustering task to worker
      worker.postMessage({
        data: data,
        k: k,
        dimensions: data[0]?.embedding?.length || 384,
        options: {
          maxIterations: 100,
          convergenceThreshold: 0.001,
        },
      });
    });
  }

  /**
   * Original in-process k-means for smaller datasets
   */
  private async performKMeansInProcess(
    data: any[],
    k: number
  ): Promise<ClusterMetrics[]> {
    // Keep original implementation for small datasets
    console.log(
      `🔄 Performing in-process k-means clustering (k=${k}) on ${data.length} items...`
    );

    const startTime = Date.now();

    // Initialize centroids randomly
    const centroids: number[][] = [];
    const dimensions = data[0]?.embedding?.length || 384;

    for (let i = 0; i < k; i++) {
      const centroid = Array.from(
        { length: dimensions },
        () => Math.random() - 0.5
      );
      centroids.push(centroid);
    }

    let hasConverged = false;
    let iteration = 0;
    const maxIterations = 100;
    const clusters: number[][] = Array.from({ length: k }, () => []);

    while (!hasConverged && iteration < maxIterations) {
      hasConverged = true;

      // Clear previous assignments
      clusters.forEach((cluster) => (cluster.length = 0));

      // Assign points to nearest centroid
      data.forEach((item, index) => {
        if (!item.embedding) return;

        let minDistance = Infinity;
        let bestCluster = 0;

        for (let i = 0; i < centroids.length; i++) {
          const distance = this.euclideanDistance(item.embedding, centroids[i]);
          if (distance < minDistance) {
            minDistance = distance;
            bestCluster = i;
          }
        }

        clusters[bestCluster].push(index);

        if (item.clusterId !== bestCluster) {
          item.clusterId = bestCluster;
          hasConverged = false;
        }
      });

      // Update centroids
      for (let i = 0; i < centroids.length; i++) {
        if (clusters[i].length === 0) continue;

        const newCentroid = new Array(dimensions).fill(0);

        clusters[i].forEach((dataIndex) => {
          const embedding = data[dataIndex].embedding;
          for (let j = 0; j < dimensions; j++) {
            newCentroid[j] += embedding[j];
          }
        });

        for (let j = 0; j < dimensions; j++) {
          newCentroid[j] /= clusters[i].length;
        }

        centroids[i] = newCentroid;
      }

      iteration++;
    }

    const processingTime = Date.now() - startTime;

    // Create cluster metrics
    const clusterMetrics: ClusterMetrics[] = [];

    for (let i = 0; i < k; i++) {
      const clusterData = clusters[i].map((index) => data[index]);
      const cohesion = this.calculateCohesion(clusterData, centroids[i]);
      const memoryUsage = this.estimateClusterMemoryUsage(clusterData);

      clusterMetrics.push({
        id: `cluster_${i}`,
        centroid: centroids[i],
        size: clusters[i].length,
        cohesion,
        separability: 0, // Calculate later
        memoryUsage,
        processingTime: processingTime / k,
      });

      this.clusters.set(`cluster_${i}`, clusterMetrics[i]);
    }

    console.log(
      `✅ In-process k-means clustering completed in ${processingTime}ms`
    );
    return clusterMetrics;
  }

  /**
   * Check if worker threads should be enabled
   */
  private enableWorkerThreads(): boolean {
    try {
      // Check if worker_threads is available
      require.resolve("worker_threads");
      return true;
    } catch {
      console.warn(
        "⚠️ Worker threads not available, falling back to in-process clustering"
      );
      return false;
    }
  }

  /**
   * Initialize Self-Organizing Map for advanced clustering
   */
  async initializeSOM(): Promise<void> {
    if (!this.somNetwork) {
      this.somNetwork = new SelfOrganizingMapRAG({
        mapWidth: 10,
        mapHeight: 10,
        dimensions: 384,
        learningRate: 0.1,
        neighborhoodRadius: 3.0,
        maxEpochs: 1000,
        clusterCount: 8,
      });

      console.log("🧠 Self-Organizing Map initialized");
    }
  }

  /**
   * Optimized memory allocation using SOM clusters
   */
  async optimizedMemoryAllocation(request: any): Promise<string> {
    await this.initializeSOM();

    if (!this.somNetwork) {
      throw new Error("SOM network not initialized");
    }

    // Create document embedding for SOM processing
    const documentEmbedding = {
      id: request.id || `doc_${Date.now()}`,
      content: request.content || "",
      embedding: request.embedding || [],
      metadata: {
        case_id: request.caseId,
        evidence_type: request.type || "general",
        legal_category: request.category,
        confidence: request.priority || 0.5,
        timestamp: Date.now(),
      },
    };

    // Use semantic search to find optimal cluster
    const searchResults = await this.somNetwork.semanticSearch(
      request.content || "",
      request.embedding || [],
      1
    );

    // Use first result if available, otherwise create new cluster
    const clusterId =
      searchResults.length > 0 ? searchResults[0].id.split("_")[1] || "0" : "0";

    // Allocate memory based on cluster characteristics
    const cluster = this.clusters.get(clusterId);
    if (cluster) {
      const pool = this.selectOptimalMemoryPool(cluster);
      await this.allocateToPool(pool.id, request, cluster.memoryUsage);
      return pool.id;
    }

    // Fallback to default allocation
    return this.allocateToDefaultPool(request);
  }

  /**
   * Dynamic cache warm-up based on access patterns
   */
  async intelligentCacheWarmup(patterns: any[]): Promise<void> {
    console.log("🔥 Starting intelligent cache warm-up...");

    // Analyze access patterns using SOM
    const clusteredPatterns = await this.performKMeansClustering(patterns, 3);

    for (const cluster of clusteredPatterns) {
      const priority = this.calculateCachePriority(cluster);
      const layers = await this.selectOptimalCacheLayer(
        cluster.id,
        "pattern",
        cluster.memoryUsage,
        priority
      );

      // Pre-load data to selected layers
      for (const layer of layers) {
        await this.preloadToLayer(layer, cluster);
      }
    }

    console.log("✅ Cache warm-up completed");
  }

  /**
   * Memory pressure response system
   */
  private async handleMemoryPressure(pressure: number): Promise<void> {
    if (pressure > 0.95) {
      console.warn("🚨 Critical memory pressure detected!");
      await this.emergencyCleanup();
    } else if (pressure > 0.85) {
      console.warn("⚠️ High memory pressure detected");
      await this.aggressiveOptimization();
    } else if (pressure > 0.7) {
      console.log("📊 Moderate memory pressure detected");
      await this.standardOptimization();
    }
  }

  /**
   * Emergency cleanup procedures
   */
  private async emergencyCleanup(): Promise<void> {
    console.log("🧹 Performing emergency cleanup...");

    // Clear least important caches
    for (const [poolId, pool] of this.memoryPools) {
      if (pool.priority > 3) {
        await this.clearPool(poolId, 0.8); // Clear 80%
      }
    }

    // Force garbage collection
    if (typeof global !== 'undefined' && (global as any).gc) {
      (global as any).gc();
    }

    // Reduce LOD to minimum
    this.currentLOD = this.lodLevels[0];

    // Compress all remaining data
    await this.compressAllPools();

    console.log("✅ Emergency cleanup completed");
  }

  /**
   * Predictive memory management using ML
   */
  async predictiveMemoryManagement(): Promise<void> {
    const history = this.optimizationHistory.slice(-100); // Last 100 optimizations

    if (history.length < 10) return;

    // Simple trend analysis
    const memoryTrend = this.calculateMemoryTrend(history);
    const accessTrend = this.calculateAccessTrend(history);

    if (memoryTrend > 0.1) {
      console.log("📈 Predicting memory increase - pre-emptive optimization");
      await this.preemptiveOptimization();
    }

    if (accessTrend > 0.2) {
      console.log("🚀 Predicting high access - cache warm-up");
      await this.predictiveCacheWarmup();
    }
  }

  /**
   * K-means clustering using worker threads for parallel processing
   */
  async performKMeansClusteringWithWorkers(
    data: any[],
    k: number = 5
  ): Promise<ClusterMetrics[]> {
    console.log(
      `🔄 Performing worker-based k-means clustering (k=${k}) on ${data.length} items...`
    );

    const startTime = Date.now();

    // For small datasets, use main thread to avoid worker overhead
    if (data.length < 1000) {
      return this.performKMeansClustering(data, k);
    }

    try {
      const worker = await this.getKMeansWorker();

      // Prepare data for worker (extract only necessary fields)
      const workerData = data.map((item, index) => ({
        id: item.id || `item_${index}`,
        embedding: item.embedding || [],
        metadata: item.metadata || {},
      }));

      // Send clustering task to worker
      const result = await this.sendWorkerMessage(worker, {
        action: "cluster",
        data: workerData,
        k,
        dimensions: workerData[0]?.embedding?.length || 384,
      });

      const processingTime = Date.now() - startTime;

      // Convert worker result to ClusterMetrics format
      const clusterMetrics: ClusterMetrics[] = result.clusters.map(
        (cluster: any, index: number) => {
          const metrics: ClusterMetrics = {
            id: `cluster_${index}`,
            centroid: cluster.centroid,
            size: cluster.size,
            cohesion: cluster.cohesion,
            separability: cluster.separability || 0,
            memoryUsage: cluster.memoryUsage,
            processingTime: processingTime / k,
          };

          this.clusters.set(`cluster_${index}`, metrics);
          return metrics;
        }
      );

      console.log(
        `✅ Worker-based k-means clustering completed in ${processingTime}ms`
      );
      return clusterMetrics;
    } catch (error: any) {
      console.error(
        "❌ Worker clustering failed, falling back to main thread:",
        error
      );
      return this.performKMeansClustering(data, k);
    }
  }

  /**
   * Send message to worker and wait for response
   */
  private sendWorkerMessage(worker: Worker, message: any): Promise<any> {
    return new Promise((resolve, reject) => {
      const messageId = crypto.randomUUID();
      const messageWithId = { ...message, messageId };

      const timeout = setTimeout(() => {
        worker.off("message", messageHandler);
        reject(new Error("Worker timeout"));
      }, 30000); // 30 second timeout

      const messageHandler = (response: any) => {
        if (response.messageId === messageId) {
          clearTimeout(timeout);
          worker.off("message", messageHandler);

          if (response.error) {
            reject(new Error(response.error));
          } else {
            resolve(response);
          }
        }
      };

      worker.on("message", messageHandler);
      worker.postMessage(messageWithId);
    });
  }

  /**
   * Parallel document processing using SIMD parser and worker threads
   */
  async processDocumentsBatch(documents: string[]): Promise<any[]> {
    console.log(
      `📄 Processing ${documents.length} documents with worker threads and SIMD`
    );

    if (documents.length === 0) return [];

    const batchSize = Math.ceil(documents.length / this.maxWorkers);
    const promises: Promise<any[]>[] = [];

    // Split documents into chunks for parallel processing
    for (
      let i = 0;
      i < this.maxWorkers && i * batchSize < documents.length;
      i++
    ) {
      const start = i * batchSize;
      const end = Math.min(start + batchSize, documents.length);
      const chunk = documents.slice(start, end);

      promises.push(this.processDocumentChunk(chunk, i));
    }

    try {
      const results = await Promise.all(promises);
      const flatResults = results.flat();

      console.log(`✅ Processed ${flatResults.length} documents successfully`);
      return flatResults;
    } catch (error: any) {
      console.error("❌ Batch processing failed:", error);
      throw error;
    }
  }

  /**
   * Process a chunk of documents in a worker thread
   */
  private async processDocumentChunk(
    documents: string[],
    chunkIndex: number
  ): Promise<any[]> {
    const workerId = `document-processor-${chunkIndex}`;

    try {
      // Create or get worker for document processing
      let worker = this.workerPool.get(workerId);

      if (!worker) {
        // Create SIMD document processor worker
        const workerPath = new URL(
          "../workers/simd-document-processor.js",
          import.meta.url
        );
        worker = new Worker(workerPath);

        worker.on("error", (error) => {
          console.error(
            `🚨 Document processor worker ${chunkIndex} error:`,
            error
          );
          this.workerPool.delete(workerId);
        });

        worker.on("exit", (code) => {
          if (code !== 0) {
            console.warn(
              `⚠️ Document processor worker ${chunkIndex} exited with code ${code}`
            );
          }
          this.workerPool.delete(workerId);
        });

        this.workerPool.set(workerId, worker);
      }

      // Send documents to worker for SIMD processing
      const result = await this.sendWorkerMessage(worker, {
        action: "processDocuments",
        documents,
        options: {
          batchSize: 1024,
          enableSIMD: true,
          memoryLimit: 256 * 1024 * 1024, // 256MB per worker
          validateStructure: true,
        },
      });

      return result.processedDocuments || [];
    } catch (error: any) {
      console.error(
        `❌ Document chunk processing failed for chunk ${chunkIndex}:`,
        error
      );

      // Fallback to main thread processing
      const { SIMDJSONParser } = await import("../parsers/simd-json-parser.js");
      const parser = new SIMDJSONParser({
        batchSize: 512,
        enableSIMD: false, // Disable SIMD in fallback
        memoryLimit: 128 * 1024 * 1024,
      });

      return parser.parseDocumentsBatch(documents);
    }
  }

  /**
   * Optimized memory allocation using worker threads for complex calculations
   */
  async optimizedMemoryAllocationWithWorkers(request: any): Promise<string> {
    await this.initializeSOM();

    if (!this.somNetwork) {
      throw new Error("SOM network not initialized");
    }

    try {
      // Use worker for complex memory optimization calculations
      const optimizationWorker = await this.getOptimizationWorker();

      const result = await this.sendWorkerMessage(optimizationWorker, {
        action: "optimizeMemoryAllocation",
        request,
        currentState: {
          memoryPools: this.serializeMemoryPools(),
          clusters: this.serializeClusters(),
          memoryPressure: this.memoryPressure,
          lodLevel: this.currentLOD,
        },
      });

      // Apply optimizations from worker result
      if (result.recommendations) {
        await this.applyOptimizationRecommendations(result.recommendations);
      }

      return result.allocatedPool || this.allocateToDefaultPool(request);
    } catch (error: any) {
      console.error(
        "❌ Worker-based optimization failed, using fallback:",
        error
      );
      return this.optimizedMemoryAllocation(request);
    }
  }

  /**
   * Get or create optimization worker
   */
  private async getOptimizationWorker(): Promise<Worker> {
    const workerId = "memory-optimizer";

    if (this.workerPool.has(workerId)) {
      return this.workerPool.get(workerId)!;
    }

    const workerPath = new URL(
      "../workers/memory-optimizer-worker.js",
      import.meta.url
    );
    const worker = new Worker(workerPath);

    worker.on("error", (error) => {
      console.error("🚨 Memory optimizer worker error:", error);
      this.workerPool.delete(workerId);
    });

    worker.on("exit", (code) => {
      if (code !== 0) {
        console.warn(`⚠️ Memory optimizer worker exited with code ${code}`);
      }
      this.workerPool.delete(workerId);
    });

    this.workerPool.set(workerId, worker);
    return worker;
  }

  /**
   * Serialize memory pools for worker communication
   */
  private serializeMemoryPools(): unknown[] {
    return Array.from(this.memoryPools.entries()).map(([id, pool]) => ({
      id,
      type: pool.type,
      current: pool.current,
      max: pool.max,
      itemCount: pool.items.size,
      lastAccessed: pool.lastAccessed,
      priority: pool.priority,
    }));
  }

  /**
   * Serialize clusters for worker communication
   */
  private serializeClusters(): unknown[] {
    return Array.from(this.clusters.values());
  }

  /**
   * Apply optimization recommendations from worker
   */
  private async applyOptimizationRecommendations(
    recommendations: any
  ): Promise<void> {
    console.log("🔧 Applying worker optimization recommendations");

    if (recommendations.adjustLOD) {
      this.currentLOD =
        this.lodLevels.find((lod) => lod.id === recommendations.targetLOD) ||
        this.currentLOD;
      await this.applyLODSettings();
    }

    if (recommendations.clearPools) {
      for (const poolClear of recommendations.clearPools) {
        await this.clearPool(poolClear.poolId, poolClear.ratio);
      }
    }

    if (recommendations.compressPools) {
      await this.compressAllPools();
    }

    if (recommendations.redistributeMemory) {
      await this.redistributeMemoryPools(recommendations.redistributeMemory);
    }
  }

  /**
   * Redistribute memory between pools based on usage patterns
   */
  private async redistributeMemoryPools(redistribution: any): Promise<void> {
    console.log("🔄 Redistributing memory pools");

    for (const adjustment of redistribution) {
      const pool = this.memoryPools.get(adjustment.poolId);
      if (pool) {
        const oldMax = pool.max;
        pool.max = Math.max(pool.current, adjustment.newMax);

        console.log(
          `📊 Pool ${adjustment.poolId}: ${oldMax} -> ${pool.max} bytes`
        );
      }
    }
  }

  // Helper methods
  private euclideanDistance(a: number[], b: number[]): number {
    if (a.length !== b.length) return Infinity;

    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      sum += Math.pow(a[i] - b[i], 2);
    }
    return Math.sqrt(sum);
  }

  private calculateCohesion(clusterData: any[], centroid: number[]): number {
    if (clusterData.length === 0) return 0;

    let totalDistance = 0;
    let validItems = 0;

    clusterData.forEach((item) => {
      if (item.embedding) {
        totalDistance += this.euclideanDistance(item.embedding, centroid);
        validItems++;
      }
    });

    return validItems > 0 ? 1 / (1 + totalDistance / validItems) : 0;
  }

  private estimateClusterMemoryUsage(clusterData: any[]): number {
    return clusterData.reduce((total, item) => {
      const itemSize = JSON.stringify(item).length;
      return total + itemSize;
    }, 0);
  }

  private selectOptimalMemoryPool(cluster: ClusterMetrics): MemoryPool {
    // Select pool based on cluster characteristics
    const pools = Array.from(this.memoryPools.values())
      .filter((pool) => pool.current + cluster.memoryUsage <= pool.max)
      .sort((a, b) => a.priority - b.priority);

    return pools[0] || this.memoryPools.get("cache")!;
  }

  private async allocateToPool(
    poolId: string,
    request: any,
    size: number
  ): Promise<void> {
    const pool = this.memoryPools.get(poolId);
    if (!pool) return;

    pool.items.set(request.id || Date.now().toString(), request);
    pool.current += size;
    pool.lastAccessed = Date.now();
  }

  private allocateToDefaultPool(request: any): string {
    const defaultPool = this.memoryPools.get("cache")!;
    const size = JSON.stringify(request).length;

    defaultPool.items.set(request.id || Date.now().toString(), request);
    defaultPool.current += size;
    defaultPool.lastAccessed = Date.now();

    return "cache";
  }

  private calculateCachePriority(cluster: ClusterMetrics): number {
    // Higher cohesion = higher priority
    return cluster.cohesion;
  }

  private async preloadToLayer(
    layer: CacheLayer,
    cluster: ClusterMetrics
  ): Promise<void> {
    // Implementation depends on layer type
    console.log(`Pre-loading cluster ${cluster.id} to ${layer.name}`);
  }

  private async getCurrentMemoryUsage(): Promise<number> {
    let total = 0;
    for (const pool of this.memoryPools.values()) {
      total += pool.current;
    }
    return total;
  }

  private async reduceLOD(): Promise<void> {
    const currentIndex = this.lodLevels.findIndex(
      (lod) => lod.id === this.currentLOD.id
    );
    if (currentIndex > 0) {
      this.currentLOD = this.lodLevels[currentIndex - 1];
      console.log(`📉 Reduced LOD to ${this.currentLOD.detail}`);
      await this.applyLODSettings();
    }
  }

  private async increaseLOD(): Promise<void> {
    const currentIndex = this.lodLevels.findIndex(
      (lod) => lod.id === this.currentLOD.id
    );
    if (currentIndex < this.lodLevels.length - 1) {
      this.currentLOD = this.lodLevels[currentIndex + 1];
      console.log(`📈 Increased LOD to ${this.currentLOD.detail}`);
      await this.applyLODSettings();
    }
  }

  private async applyLODSettings(): Promise<void> {
    // Adjust memory pools based on LOD
    for (const pool of this.memoryPools.values()) {
      pool.max = Math.floor(pool.max * this.currentLOD.compressionRatio);
    }

    // Trigger cleanup if necessary
    if (this.memoryPressure > 0.8) {
      await this.aggressiveOptimization();
    }
  }

  private async adjustObjectLimits(): Promise<void> {
    const limit = Math.floor(
      this.currentLOD.maxObjects * (1 - this.memoryPressure * 0.5)
    );

    for (const pool of this.memoryPools.values()) {
      if (pool.items.size > limit) {
        await this.clearPool(pool.id, 0.3); // Clear 30% of items
      }
    }
  }

  private async clearPool(poolId: string, ratio: number): Promise<void> {
    const pool = this.memoryPools.get(poolId);
    if (!pool) return;

    const itemsToRemove = Math.floor(pool.items.size * ratio);
    const items = Array.from(pool.items.entries()).sort(
      ([, a], [, b]) => (a.lastAccessed || 0) - (b.lastAccessed || 0)
    );

    for (let i = 0; i < itemsToRemove && i < items.length; i++) {
      const [key, item] = items[i];
      pool.items.delete(key);
      pool.current -= JSON.stringify(item).length;
    }

    console.log(`🧹 Cleared ${itemsToRemove} items from pool ${poolId}`);
  }

  private async aggressiveOptimization(): Promise<void> {
    console.log("⚡ Performing aggressive optimization...");

    // Clear low-priority pools
    await this.clearPool("cache", 0.5);
    await this.clearPool("som", 0.3);

    // Compress remaining data
    await this.compressAllPools();

    // Reduce batch sizes
    this.dockerOptimizer = new DockerResourceOptimizer({
      maxMemoryMB: this.currentLOD.maxMemoryMB,
      batchSize: 50,
      parallelism: 2,
    });
  }

  private async standardOptimization(): Promise<void> {
    console.log("📊 Performing standard optimization...");

    // Clear oldest items from low-priority pools
    await this.clearPool("cache", 0.2);

    // Trigger garbage collection
    if (typeof global !== 'undefined' && (global as any).gc) {
      (global as any).gc();
    }
  }

  private async compressAllPools(): Promise<void> {
    console.log("🗜️ Compressing all memory pools...");

    for (const [poolId, pool] of this.memoryPools) {
      const compressedItems = new Map();
      let newSize = 0;

      for (const [key, item] of pool.items) {
        const compressed = await this.dockerOptimizer.cacheWithCompression(
          `${poolId}_${key}`,
          item
        );
        compressedItems.set(key, compressed);
        newSize += JSON.stringify(compressed).length * 0.3; // Assume 70% compression
      }

      pool.items = compressedItems;
      pool.current = newSize;
    }
  }

  private calculateMemoryTrend(history: any[]): number {
    if (history.length < 5) return 0;

    const recentMemory = history.slice(-5).map((h) => h.memoryUsage);
    const earlyMemory = history.slice(0, 5).map((h) => h.memoryUsage);

    const recentAvg =
      recentMemory.reduce((a, b) => a + b, 0) / recentMemory.length;
    const earlyAvg =
      earlyMemory.reduce((a, b) => a + b, 0) / earlyMemory.length;

    return (recentAvg - earlyAvg) / earlyAvg;
  }

  private calculateAccessTrend(history: any[]): number {
    if (history.length < 5) return 0;

    const recentAccess = history.slice(-5).map((h) => h.accessCount);
    const earlyAccess = history.slice(0, 5).map((h) => h.accessCount);

    const recentAvg =
      recentAccess.reduce((a, b) => a + b, 0) / recentAccess.length;
    const earlyAvg =
      earlyAccess.reduce((a, b) => a + b, 0) / earlyAccess.length;

    return (recentAvg - earlyAvg) / earlyAvg;
  }

  private async preemptiveOptimization(): Promise<void> {
    console.log("🔮 Performing preemptive optimization...");

    // Reduce LOD proactively
    if (this.currentLOD.detail !== "low") {
      await this.reduceLOD();
    }

    // Clear predictively unused items
    await this.clearPool("cache", 0.15);
  }

  private async predictiveCacheWarmup(): Promise<void> {
    console.log("🔥 Performing predictive cache warm-up...");

    // Warm up high-priority caches
    const fastLayers = Array.from(this.cacheLayers.values())
      .filter((layer) => layer.priority <= 3)
      .sort((a, b) => a.priority - b.priority);

    // Implementation would pre-load frequently accessed data
  }

  /**
   * Start monitoring system
   */
  private startMemoryMonitoring(): void {
    setInterval(async () => {
      await this.adaptiveLODManagement();
      await this.handleMemoryPressure(this.memoryPressure);
      await this.predictiveMemoryManagement();

      // Record optimization history
      this.optimizationHistory.push({
        timestamp: Date.now(),
        memoryUsage: await this.getCurrentMemoryUsage(),
        memoryPressure: this.memoryPressure,
        lodLevel: this.currentLOD.detail,
        accessCount: Array.from(this.memoryPools.values()).reduce(
          (total, pool) => total + pool.items.size,
          0
        ),
      });

      // Keep history manageable
      if (this.optimizationHistory.length > 200) {
        this.optimizationHistory = this.optimizationHistory.slice(-100);
      }
    }, 10000); // Every 10 seconds
  }

  /**
   * Get optimization status
   */
  getOptimizationStatus() {
    return {
      currentLOD: this.currentLOD,
      memoryPressure: this.memoryPressure,
      pools: Array.from(this.memoryPools.entries()).map(([id, pool]) => ({
        id,
        usage: `${pool.current} / ${pool.max}`,
        percentage: (pool.current / pool.max) * 100,
        items: pool.items.size,
      })),
      clusters: Array.from(this.clusters.values()),
      cacheLayers: Array.from(this.cacheLayers.values()),
    };
  }

  /**
   * Cleanup resources including worker threads
   */
  dispose(): void {
    console.log("🧹 Disposing advanced memory optimizer...");

    // Terminate all worker threads
    for (const [workerId, worker] of this.workerPool) {
      console.log(`🛑 Terminating worker: ${workerId}`);
      worker.terminate();
    }
    this.workerPool.clear();

    // Clean up memory pools
    this.memoryPools.clear();
    this.clusters.clear();
    this.cacheLayers.clear();
    this.optimizationHistory = [];

    // Clean up Docker optimizer
    this.dockerOptimizer.dispose();

    // Clean up SOM network
    if (this.somNetwork) {
      this.somNetwork = null;
    }

    // Clean up SIMD parser
    simdParser.dispose();

    console.log("✅ Advanced memory optimizer disposed");
  }

  /**
   * SIMD-optimized document processing and memory allocation
   */
  async processBatchDocumentsSIMD(
    jsonDocuments: string[]
  ): Promise<ParsedLegalDocument[]> {
    console.log(
      `📄 Processing ${jsonDocuments.length} documents with SIMD optimization...`
    );

    const startTime = performance.now();

    // Use SIMD parser for high-performance document processing
    const parsedDocuments = await simdParser.parseDocumentsBatch(jsonDocuments);

    // Process embeddings and allocate memory efficiently
    const processedDocuments = await Promise.all(
      parsedDocuments.map(async (doc) => {
        // Generate embeddings if not present
        if (!doc.embeddings && doc.content) {
          doc.embeddings = await this.generateDocumentEmbeddings(doc.content);
        }

        // Allocate optimized memory based on document characteristics
        const memoryPool = await this.optimizedMemoryAllocation({
          id: doc.id,
          content: doc.content,
          embedding: doc.embeddings ? Array.from(doc.embeddings) : [],
          type: doc.documentType,
          caseId: doc.caseNumber,
          category: doc.metadata?.category,
          priority: this.calculateDocumentPriority(doc),
        });

        // Add memory allocation info to metadata
        doc.metadata = {
          ...doc.metadata,
          memoryPool,
          processedAt: Date.now(),
          simdOptimized: true,
        };

        return doc;
      })
    );

    const processingTime = performance.now() - startTime;
    console.log(
      `✅ SIMD document processing completed in ${processingTime.toFixed(2)}ms`
    );

    // Update optimization history
    this.optimizationHistory.push({
      timestamp: Date.now(),
      operation: "simd_batch_processing",
      documentCount: processedDocuments.length,
      processingTime,
      memoryUsage: await this.getCurrentMemoryUsage(),
      memoryPressure: this.memoryPressure,
    });

    return processedDocuments;
  }

  /**
   * Generate embeddings for document content (placeholder for actual embedding service)
   */
  private async generateDocumentEmbeddings(
    content: string
  ): Promise<Float32Array> {
    // This would typically call your embedding service (Ollama, OpenAI, etc.)
    // For now, return a mock embedding
    const dimensions = 384; // nomic-embed-text dimensions
    const embedding = new Float32Array(dimensions);

    // Simple hash-based mock embedding
    const hash = this.simpleHash(content);
    for (let i = 0; i < dimensions; i++) {
      embedding[i] = (Math.sin(hash + i) + 1) / 2; // Normalize to [0,1]
    }

    return embedding;
  }

  /**
   * Calculate document priority for memory allocation
   */
  private calculateDocumentPriority(doc: ParsedLegalDocument): number {
    let priority = 0.5; // Base priority

    // Higher priority for recent documents
    const age = Date.now() - (doc.metadata?.timestamp || Date.now());
    const ageScore = Math.max(0, 1 - age / (30 * 24 * 60 * 60 * 1000)); // 30 days
    priority += ageScore * 0.3;

    // Higher priority for legal documents with case numbers
    if (doc.caseNumber) {
      priority += 0.2;
    }

    // Higher priority for larger documents
    const sizeScore = Math.min(1, doc.content.length / 10000); // Normalize to 10k chars
    priority += sizeScore * 0.2;

    // Higher priority for certain document types
    const importantTypes = ["evidence", "testimony", "contract", "ruling"];
    if (importantTypes.includes(doc.documentType.toLowerCase())) {
      priority += 0.3;
    }

    return Math.min(1, priority);
  }

  /**
   * Simple hash function for mock embeddings
   */
  private simpleHash(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
  }
}
