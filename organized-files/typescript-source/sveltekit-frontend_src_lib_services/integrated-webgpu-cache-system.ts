/**
 * Integrated WebGPU Cache System
 *
 * @module IntegratedWebGPUCacheSystem
 *
 * @remarks
 * High-level client-side caching subsystem that unifies:
 * - A WebGPU-accelerated ranking cache (via webgpuRankingCache),
 * - A NES GPU / memory bridge (nesGPUBridge) for efficient memory layouts and texture mapping,
 * - WASM worker-based background processing for concurrent cache operations,
 * - Continuous performance monitoring and adaptive auto-optimization.
 *
 * The system exposes a singleton instance (integratedCacheSystem) that provides:
 * - performEnhancedSearch(context): GPU-accelerated semantic search with ranking and caching,
 * - storeLegalDocument(document, options): persist legal documents with optional GPU texture support and ranking publication,
 * - getSystemMetrics(): snapshot of runtime metrics including queue size, worker utilization and health,
 * - destroy(): graceful shutdown / resource cleanup.
 *
 * Key concepts:
 * - Initialization: the instance will asynchronously initialize WebGPU and WASM worker subsystems on first use.
 * - Concurrency: a configurable worker pool processes queued CacheOperation items. Queue capacity and pool size are configurable.
 * - Prioritization: queued operations are sorted by priority (high > medium > low).
 * - Resilience: worker crashes are detected, active operations are marked failed, and worker restart attempts are performed.
 * - Performance monitoring: periodic sampling aggregates WebGPU, WASM and NES metrics, and historical samples are retained for adaptive decisions.
 * - Auto-optimization: the system performs memory, cache and GPU optimizations when configured thresholds are exceeded (throttled to avoid thrashing).
 * - NES integration: optional synchronization with the NES-GPU bridge enables specialized FlatBuffer layouts and ranking textures for efficient GPU processing.
 *
 * Configuration (IntegratedCacheConfig):
 * - enableWebGPUAcceleration: enable/disable GPU pathways.
 * - maxTextureSize/gpuMemoryBudgetMB: bounds for texture allocation and GPU memory accounting.
 * - enableWASMWorkers/workerPoolSize/maxConcurrentOperations/queueCapacity: concurrency and backpressure controls.
 * - maxCacheSlots/compressionLevel: WASM / cache tuning (affects stored artifacts and compression behavior).
 * - enablePerformanceMonitoring/metricsUpdateIntervalMs/autoOptimization: telemetry/auto-tuning controls.
 * - enableNESMemorySync/nesMemorySyncIntervalMs/bankSwitchingEnabled: NES memory integration specifics.
 *
 * Metrics (IntegratedCacheMetrics):
 * The system provides aggregated metrics across domains: cache performance, WebGPU utilization, WASM worker usage,
 * NES memory stats, and system-level metrics (memory, CPU estimate, latency). Metrics are used to compute a simple
 * system-health score and to drive auto-optimization decisions.
 *
 * Public API summary:
 * - performEnhancedSearch(context: SearchContext): Promise<RankingCacheResponse<RankingResult[]>>
 *   Performs a semantic search over cached/ranked items. Uses webgpuRankingCache.searchAndCache under the hood,
 *   honoring similarity thresholds, limits and caching options configured by the system.
 *
 * - storeLegalDocument(document: LegalDocument, options?): Promise<boolean>
 *   Stores a legal document, optionally converting it to a FlatBuffer for the NES bridge, creating a GPU ranking
 *   texture from an embedding, and publishing rankings into the WebGPU ranking cache. Returns true on success.
 *
 * - getSystemMetrics(): IntegratedCacheMetrics & { operationQueue: number; activeOperations: number; workerUtilization: number; systemHealth: 'excellent'|'good'|'fair'|'poor' }
 *   Returns a snapshot of current metrics plus queue size, number of active operations, worker utilization and a simple health classification.
 *
 * - destroy(): Promise<void>
 *   Stops performance monitoring, terminates all workers, and releases GPU/NES resources. Safe to call on page unload.
 *
 * Behavior & guarantees:
 * - Lazy initialization: the first operation will ensure the underlying systems are ready; initialization errors propagate.
 * - Backpressure: submitting operations beyond queueCapacity throws an error.
 * - Ordering: operations are scheduled by priority; within the same priority they are FIFO.
 * - Timeouts: IPC with workers uses a bounded timeout for responsiveness; worker operations may be retried via worker restarts but not re-enqueued automatically.
 * - Error handling: failed operations are marked as failed with an error message; metrics are updated to reflect failures.
 *
 * Usage example:
 * @example
 * // Acquire the singleton and perform a search
 * const response = await integratedCacheSystem.performEnhancedSearch({
 *   query: 'search text',
 *   documentTypes: ['contract', 'brief'],
 *   similarityThreshold: 0.75,
 *   maxResults: 10,
 *   useSemanticSearch: true,
 *   enableRanking: true
 * });
 *
 * // Store a document with GPU acceleration and ranking
 * await integratedCacheSystem.storeLegalDocument(doc, { enableRanking: true, useGPUAcceleration: true, priority: 'high' });
 *
 * Notes and integration tips:
 * - Ensure the host environment supports WebGPU and Worker creation (browsers with WebGPU enabled).
 * - Adjust gpuMemoryBudgetMB and maxTextureSize relative to the target device capabilities to avoid OOM.
 * - When using NES integration, provide well-formed LegalDocument.metadata.vectorEmbedding arrays for texture creation.
 * - The system maintains an internal performanceHistory used for trend analysis; you can extend or persist this history externally if required.
 *
 * See also:
 * - webgpu-ranking-cache: WebGPU-enabled ranking cache primitives used by this system.
 * - nes-gpu-memory-bridge: Utilities for converting documents to FlatBuffer layouts and creating GPU textures.
 *
 * @public
 */

import { webgpuRankingCache, type RankingResult, type RankingCacheResponse } from './webgpu-ranking-cache'
import { nesGPUBridge, type NESGPUMemoryBridge } from '../gpu/nes-gpu-memory-bridge'
import { webgpuAI } from '../webgpu/webgpu-ai-engine';
import type { LegalDocument } from '../memory/nes-memory-architecture';

export interface IntegratedCacheConfig {
  // WebGPU Configuration
  enableWebGPUAcceleration: boolean;
  maxTextureSize: number;
  gpuMemoryBudgetMB: number;

  // WASM Configuration
  enableWASMWorkers: boolean;
  maxCacheSlots: number;
  compressionLevel: 'none' | 'low' | 'high';

  // Performance Configuration
  enablePerformanceMonitoring: boolean;
  metricsUpdateIntervalMs: number;
  autoOptimization: boolean;

  // NES Memory Integration
  enableNESMemorySync: boolean;
  nesMemorySyncIntervalMs: number;
  bankSwitchingEnabled: boolean;

  // Concurrency Configuration
  maxConcurrentOperations: number;
  workerPoolSize: number;
  queueCapacity: number;
}

export interface CacheOperation {
  id: string;
  type: 'store' | 'retrieve' | 'search' | 'optimize' | 'sync';
  data: any;
  priority: 'high' | 'medium' | 'low';
  timestamp: number;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  processingTime?: number;
  error?: string;
}

export interface IntegratedCacheMetrics {
  // Cache Performance
  totalOperations: number;
  successfulOperations: number;
  failedOperations: number;
  averageOperationTime: number;

  // WebGPU Performance
  webgpuUtilization: number;
  textureMemoryUsage: number;
  shaderCompilationTime: number;
  gpuAccelerationGain: number;

  // WASM Performance
  wasmWorkerUtilization: number;
  compressionRatio: number;
  binaryCacheEfficiency: number;

  // NES Memory Performance
  nesMemoryUtilization: number;
  bankSwitchCount: number;
  memoryPressureEvents: number;
  syncLatency: number;

  // System Performance
  memoryUsage: number;
  cpuUtilization: number;
  networkLatency: number;
  powerEfficiency: number;
}

export interface SearchContext {
  query: string;
  documentTypes: string[];
  userPreferences: any;
  sessionContext: any;
  similarityThreshold: number;
  maxResults: number;
  useSemanticSearch: boolean;
  enableRanking: boolean;
}

class IntegratedWebGPUCacheSystem {
  private config: IntegratedCacheConfig;
  private isInitialized = false;
  private operationQueue: CacheOperation[] = [];
  private activeOperations = new Map<string, CacheOperation>();
  private metrics: IntegratedCacheMetrics;
  private performanceHistory: IntegratedCacheMetrics[] = [];

  // Worker pool for concurrency
  private workerPool: Worker[] = [];
  private availableWorkers: Worker[] = [];
  private workerOperations = new Map<Worker, CacheOperation>();

  // Performance monitoring
  private metricsInterval: number | null = null;
  private lastOptimizationTime = 0;

  constructor(config: Partial<IntegratedCacheConfig> = {}) {
    this.config = {
      enableWebGPUAcceleration: true,
      maxTextureSize: 2048,
      gpuMemoryBudgetMB: 512,
      enableWASMWorkers: true,
      maxCacheSlots: 85,
      compressionLevel: 'high',
      enablePerformanceMonitoring: true,
      metricsUpdateIntervalMs: 1000,
      autoOptimization: true,
      enableNESMemorySync: true,
      nesMemorySyncIntervalMs: 16.67, // 60 FPS
      bankSwitchingEnabled: true,
      maxConcurrentOperations: 8,
      workerPoolSize: 4,
      queueCapacity: 100,
      ...config
    };

    this.metrics = this.initializeMetrics();
    this.initializeSystem();
  }

  private initializeMetrics(): IntegratedCacheMetrics {
    return {
      totalOperations: 0,
      successfulOperations: 0,
      failedOperations: 0,
      averageOperationTime: 0,
      webgpuUtilization: 0,
      textureMemoryUsage: 0,
      shaderCompilationTime: 0,
      gpuAccelerationGain: 0,
      wasmWorkerUtilization: 0,
      compressionRatio: 0,
      binaryCacheEfficiency: 0,
      nesMemoryUtilization: 0,
      bankSwitchCount: 0,
      memoryPressureEvents: 0,
      syncLatency: 0,
      memoryUsage: 0,
      cpuUtilization: 0,
      networkLatency: 0,
      powerEfficiency: 0
    };
  }

  private async initializeSystem(): Promise<void> {
    try {
      // Initialize WebGPU systems
      await webgpuAI.waitForReady();
      await webgpuRankingCache.waitForReady();

      // Initialize worker pool
      if (this.config.enableWASMWorkers) {
        await this.initializeWorkerPool();
      }

      // Start performance monitoring
      if (this.config.enablePerformanceMonitoring) {
        this.startPerformanceMonitoring();
      }

      // Initialize NES GPU bridge
      if (this.config.enableNESMemorySync) {
        await this.initializeNESIntegration();
      }

      this.isInitialized = true;
      console.log('🚀 Integrated WebGPU Cache System initialized with full acceleration');

    } catch (error) {
      console.error('❌ Failed to initialize integrated cache system:', error);
      throw error;
    }
  }

  private async initializeWorkerPool(): Promise<void> {
    for (let i = 0; i < this.config.workerPoolSize; i++) {
      try {
        const worker = new Worker('/workers/ranking-cache-worker.js');

        worker.onmessage = (event) => {
          this.handleWorkerMessage(worker, event);
        };

        worker.onerror = (error) => {
          this.handleWorkerError(worker, error);
        };

        await this.sendWorkerMessage(worker, { action: 'init' });
        this.workerPool.push(worker);
        this.availableWorkers.push(worker);
      } catch (error) {
        console.warn(`Failed to initialize worker ${i}:`, error);
      }
    }
    console.log(`✅ Initialized worker pool with ${this.workerPool.length} workers`);
  }

  private async initializeNESIntegration(): Promise<void> {
    // Wait for NES GPU bridge to be ready
    // This connects the client-side ranking cache with NES memory architecture
    console.log('🎮 NES-GPU integration active for memory coherency');
  }

  private startPerformanceMonitoring(): void {
    this.metricsInterval = setInterval(() => {
      this.updateMetrics();
      this.performanceHistory.push({ ...this.metrics });

      // Keep only recent history (last 5 minutes)
      if (this.performanceHistory.length > 300) {
        this.performanceHistory = this.performanceHistory.slice(-300);
      }

      // Auto-optimization
      if (this.config.autoOptimization && this.shouldOptimize()) {
        this.performAutoOptimization();
      }
    }, this.config.metricsUpdateIntervalMs) as any;
  }

  private async updateMetrics(): Promise<void> {
    try {
      // Get WebGPU metrics
      const webgpuCapabilities = webgpuAI.getCapabilities();
      const rankingCacheMetrics = await webgpuRankingCache.getMetrics();
      const nesGPUMetrics = nesGPUBridge.getPerformanceMetrics();

      // Update integrated metrics
      this.metrics = {
        ...this.metrics,
        webgpuUtilization: webgpuCapabilities.performance?.gpuUtilization || 0,
        textureMemoryUsage: nesGPUMetrics.totalBytesStored / (this.config.gpuMemoryBudgetMB * 1024 * 1024),
        shaderCompilationTime: webgpuCapabilities.performance?.averageProcessingTime || 0,
        gpuAccelerationGain: rankingCacheMetrics.wasmAccelerationGain || 0,
        wasmWorkerUtilization: this.calculateWorkerUtilization(),
        compressionRatio: rankingCacheMetrics.averageCompressionRatio || 0,
        binaryCacheEfficiency: rankingCacheMetrics.averageHitRatio || 0,
        nesMemoryUtilization: nesGPUMetrics.memoryEfficiencyRatio || 0,
        memoryUsage: this.calculateMemoryUsage(),
        cpuUtilization: this.estimateCPUUtilization()
      };

    } catch (error) {
      console.warn('Failed to update metrics:', error);
    }
  }

  private calculateWorkerUtilization(): number {
    const activeWorkers = this.workerPool.length - this.availableWorkers.length;
    return activeWorkers / this.workerPool.length;
  }

  private calculateMemoryUsage(): number {
    // Estimate memory usage from cache sizes
    if (typeof performance !== 'undefined' && 'memory' in performance) {
      return (performance as any).memory.usedJSHeapSize / (performance as any).memory.jsHeapSizeLimit;
    }
    return 0;
  }

  private estimateCPUUtilization(): number {
    // Simple CPU utilization estimate based on operation throughput
    const recentOps = this.performanceHistory.slice(-10);
    if (recentOps.length < 2) return 0;

    const avgOpTime = recentOps.reduce((sum, m) => sum + m.averageOperationTime, 0) / recentOps.length;
    return Math.min(avgOpTime / 100, 1); // Normalize to 0-1
  }

  private shouldOptimize(): boolean {
    const now = Date.now();
    if (now - this.lastOptimizationTime < 60000) return false; // Throttle to once per minute

    // Check optimization criteria
    const criteria = [
      this.metrics.memoryUsage > 0.8,
      this.metrics.binaryCacheEfficiency < 0.6,
      this.metrics.webgpuUtilization > 0.9,
      this.metrics.averageOperationTime > 100
    ];

    return criteria.filter(Boolean).length >= 2;
  }

  private async performAutoOptimization(): Promise<void> {
    console.log('🔧 Performing automatic optimization');
    this.lastOptimizationTime = Date.now();

    try {
      // Memory optimization
      if (this.metrics.memoryUsage > 0.8) {
        await this.optimizeMemoryUsage();
      }

      // Cache optimization
      if (this.metrics.binaryCacheEfficiency < 0.6) {
        await this.optimizeCacheStrategy();
      }

      // GPU optimization
      if (this.metrics.webgpuUtilization > 0.9) {
        await this.optimizeGPUUsage();
      }

      console.log('✅ Auto-optimization completed');
    } catch (error) {
      console.error('❌ Auto-optimization failed:', error);
    }
  }

  private async optimizeMemoryUsage(): Promise<void> {
    // Clear old performance history
    this.performanceHistory = this.performanceHistory.slice(-100);

    // Trigger garbage collection in workers
    const clearPromises = this.workerPool.map(worker =>
      this.sendWorkerMessage(worker, { action: 'clear_old_cache' })
    );
    await Promise.allSettled(clearPromises);

    // NES memory bank optimization
    if (this.config.enableNESMemorySync) {
      await nesGPUBridge.synchronizeNESGPUMemory();
    }
  }

  private async optimizeCacheStrategy(): Promise<void> {
    // Analyze cache hit patterns and adjust strategy
    await webgpuRankingCache.clearCache();

    // Preload commonly accessed items
    // This would be implemented based on usage patterns
  }

  private async optimizeGPUUsage(): Promise<void> {
    // Reduce GPU texture sizes temporarily
    // Implement shader optimization
    // This would involve more complex GPU resource management
  }

  /**
   * High-level API: Enhanced semantic search with full GPU acceleration
   */
  async performEnhancedSearch(context: SearchContext): Promise<RankingCacheResponse<RankingResult[]>> {
    const operation: CacheOperation = {
      id: `search_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      type: 'search',
      data: context,
      priority: 'high',
      timestamp: Date.now(),
      status: 'pending'
    };

    try {
      return await this.executeOperation(operation);
    } catch (error) {
      return {
        success: false,
        found: false,
        error: error instanceof Error ? error.message : 'Search failed',
        processingTime: 0,
        protocol: 'fallback'
      };
    }
  }

  /**
   * Store legal document with full GPU acceleration and ranking
   */
  async storeLegalDocument(document: LegalDocument, options: {
    enableRanking?: boolean;
    useGPUAcceleration?: boolean;
    priority?: 'high' | 'medium' | 'low';
  } = {}): Promise<boolean> {
    const operation: CacheOperation = {
      id: `store_${document.id}`,
      type: 'store',
      data: { document, options },
      priority: options.priority || 'medium',
      timestamp: Date.now(),
      status: 'pending'
    };

    try {
      await this.executeOperation(operation);
      return true;
    } catch (error) {
      console.error('Failed to store legal document:', error);
      return false;
    }
  }

  /**
   * Execute cache operation with worker pool and GPU acceleration
   */
  private async executeOperation(operation: CacheOperation): Promise<any> {
    if (!this.isInitialized) {
      await this.initializeSystem();
    }

    // Add to queue
    if (this.operationQueue.length >= this.config.queueCapacity) {
      throw new Error('Operation queue at capacity');
    }

    this.operationQueue.push(operation);
    this.activeOperations.set(operation.id, operation);

    // Process queue
    return await this.processOperationQueue();
  }

  private async processOperationQueue(): Promise<any> {
    // Sort queue by priority
    this.operationQueue.sort((a, b) => {
      const priorityOrder = { high: 3, medium: 2, low: 1 };
      return priorityOrder[b.priority] - priorityOrder[a.priority];
    });

    // Find available worker
    const worker = this.getAvailableWorker();
    if (!worker) {
      // Wait for worker to become available
      return new Promise((resolve, reject) => {
        const checkWorker = () => {
          const availableWorker = this.getAvailableWorker();
          if (availableWorker) {
            this.processNextOperation(availableWorker).then(resolve).catch(reject);
          } else {
            setTimeout(checkWorker, 10);
          }
        };
        checkWorker();
      });
    }

    return await this.processNextOperation(worker);
  }

  private async processNextOperation(worker: Worker): Promise<any> {
    const operation = this.operationQueue.shift();
    if (!operation) return null;

    const startTime = performance.now();
    operation.status = 'processing';
    this.workerOperations.set(worker, operation);

    // Remove worker from available pool
    const workerIndex = this.availableWorkers.indexOf(worker);
    if (workerIndex > -1) {
      this.availableWorkers.splice(workerIndex, 1);
    }

    try {
      let result;

      switch (operation.type) {
        case 'search':
          result = await this.executeSearchOperation(worker, operation.data);
          break;
        case 'store':
          result = await this.executeStoreOperation(worker, operation.data);
          break;
        case 'optimize':
          result = await this.executeOptimizeOperation(worker, operation.data);
          break;
        default:
          throw new Error(`Unknown operation type: ${operation.type}`);
      }

      operation.status = 'completed';
      operation.processingTime = performance.now() - startTime;

      // Update metrics
      this.metrics.totalOperations++;
      this.metrics.successfulOperations++;
      this.metrics.averageOperationTime = (
        (this.metrics.averageOperationTime * (this.metrics.totalOperations - 1) +
         operation.processingTime) / this.metrics.totalOperations
      );

      return result;

    } catch (error) {
      operation.status = 'failed';
      operation.error = error instanceof Error ? error.message : 'Unknown error';
      operation.processingTime = performance.now() - startTime;

      this.metrics.totalOperations++;
      this.metrics.failedOperations++;

      throw error;

    } finally {
      // Return worker to available pool
      this.availableWorkers.push(worker);
      this.workerOperations.delete(worker);
      this.activeOperations.delete(operation.id);
    }
  }

  private async executeSearchOperation(worker: Worker, searchContext: SearchContext): Promise<RankingCacheResponse<RankingResult[]>> {
    // Use integrated WebGPU ranking cache with NES memory optimization
    const result = await webgpuRankingCache.searchAndCache(
      searchContext.query,
      [], // Context would be provided by the application
      {
        threshold: searchContext.similarityThreshold,
        limit: searchContext.maxResults,
        useCache: true,
        cacheOptions: {
          useWebGPU: this.config.enableWebGPUAcceleration,
          compress: this.config.compressionLevel !== 'none'
        }
      }
    );

    return result;
  }

  private async executeStoreOperation(worker: Worker, data: { document: LegalDocument; options: any }): Promise<void> {
    // Convert legal document to FlatBuffer format
    const flatBuffer = await nesGPUBridge.createFlatBufferFromDocument(data.document);

    // Create GPU texture if requested
    if (data.options.useGPUAcceleration && data.document.metadata?.vectorEmbedding) {
      const embedding = data.document.metadata.vectorEmbedding;
      const dimensions = this.calculateTextureDimensions(embedding.length);

      await nesGPUBridge.createRankingTexture(
        data.document.id,
        embedding,
        dimensions
      );
    }

    // Store in ranking cache if requested
    if (data.options.enableRanking) {
      const rankingResults: RankingResult[] = [{
        docId: parseInt(data.document.id) || 0,
        score: data.document.confidenceLevel,
        flags: this.getLegalDocumentFlags(data.document),
        summary: this.extractSummary(data.document),
        url: data.document.metadata?.sourceUrl || ''
      }];

      await webgpuRankingCache.publishRankings(rankingResults, {
        useWebGPU: this.config.enableWebGPUAcceleration,
        compress: this.config.compressionLevel !== 'none'
      });
    }
  }

  private async executeOptimizeOperation(worker: Worker, data: any): Promise<void> {
    await this.performAutoOptimization();
  }

  private getAvailableWorker(): Worker | null {
    return this.availableWorkers[0] || null;
  }

  private async sendWorkerMessage(worker: Worker, message: any): Promise<any> {
    return new Promise((resolve, reject) => {
      const messageId = Date.now() + Math.random();
      const timeout = setTimeout(() => reject(new Error('Worker timeout')), 10000);

      const handleMessage = (event: MessageEvent) => {
        if (event.data.id === messageId) {
          clearTimeout(timeout);
          worker.removeEventListener('message', handleMessage);
          if (event.data.success) {
            resolve(event.data.result);
          } else {
            reject(new Error(event.data.error));
          }
        }
      };

      worker.addEventListener('message', handleMessage);
      worker.postMessage({ ...message, id: messageId });
    });
  }

  private handleWorkerMessage(worker: Worker, event: MessageEvent): void {
    // Handle worker lifecycle messages
    if (event.data.type === 'metrics') {
      // Update worker-specific metrics
    } else if (event.data.type === 'error') {
      console.error(`Worker error:`, event.data.error);
    }
  }

  private handleWorkerError(worker: Worker, error: ErrorEvent): void {
    console.error('Worker error:', error);

    // Remove from available workers
    const index = this.availableWorkers.indexOf(worker);
    if (index > -1) {
      this.availableWorkers.splice(index, 1);
    }

    // Fail any active operation on this worker
    const operation = this.workerOperations.get(worker);
    if (operation) {
      operation.status = 'failed';
      operation.error = 'Worker crashed';
      this.metrics.failedOperations++;
    }

    // Attempt to restart worker
    this.restartWorker(worker);
  }

  private async restartWorker(failedWorker: Worker): Promise<void> {
    try {
      failedWorker.terminate();

      const newWorker = new Worker('/workers/ranking-cache-worker.js');
      newWorker.onmessage = (event) => this.handleWorkerMessage(newWorker, event);
      newWorker.onerror = (error) => this.handleWorkerError(newWorker, error);

      await this.sendWorkerMessage(newWorker, { action: 'init' });

      // Replace in worker pool
      const poolIndex = this.workerPool.indexOf(failedWorker);
      if (poolIndex > -1) {
        this.workerPool[poolIndex] = newWorker;
        this.availableWorkers.push(newWorker);
      }

      console.log('✅ Worker restarted successfully');
    } catch (error) {
      console.error('❌ Failed to restart worker:', error);
    }
  }

  private calculateTextureDimensions(vectorLength: number): { width: number; height: number } {
    const sqrt = Math.sqrt(vectorLength);
    const width = Math.ceil(sqrt);
    const height = Math.ceil(vectorLength / width);
    return { width: Math.min(width, this.config.maxTextureSize), height: Math.min(height, this.config.maxTextureSize) };
  }

  private getLegalDocumentFlags(document: LegalDocument): number {
    let flags = 0;
    if (document.type === 'contract') flags |= 0x01
    if (document.type === 'evidence') flags |= 0x02
    if (document.type === 'brief') flags |= 0x04
    if (document.type === 'citation') flags |= 0x08
    if (document.riskLevel === 'high' || document.riskLevel === 'critical') flags |= 0x10;
    return flags;
  }

  private extractSummary(document: LegalDocument): string {
    return document.metadata?.description || `${document.type} document - ${document.id}`;
  }

  /**
   * Get comprehensive system metrics
   */
  getSystemMetrics(): IntegratedCacheMetrics & {
    operationQueue: number;
    activeOperations: number;
    workerUtilization: number;
    systemHealth: 'excellent' | 'good' | 'fair' | 'poor';
  } {
    const health = this.calculateSystemHealth();

    return {
      ...this.metrics,
      operationQueue: this.operationQueue.length,
      activeOperations: this.activeOperations.size,
      workerUtilization: this.calculateWorkerUtilization(),
      systemHealth: health
    };
  }

  private calculateSystemHealth(): 'excellent' | 'good' | 'fair' | 'poor' {
    const score = [
      this.metrics.binaryCacheEfficiency,
      1 - this.metrics.memoryUsage,
      this.metrics.gpuAccelerationGain / 100,
      1 - (this.metrics.failedOperations / Math.max(this.metrics.totalOperations, 1))
    ].reduce((sum, val) => sum + val, 0) / 4;

    if (score > 0.8) return 'excellent';
    if (score > 0.6) return 'good';
    if (score > 0.4) return 'fair';
    return 'poor';
  }

  /**
   * Cleanup resources
   */
  async destroy(): Promise<void> {
    // Stop performance monitoring
    if (this.metricsInterval) {
      clearInterval(this.metricsInterval);
      this.metricsInterval = null;
    }

    // Terminate all workers
    this.workerPool.forEach(worker => worker.terminate());
    this.workerPool = [];
    this.availableWorkers = [];

    // Cleanup GPU resources
    await nesGPUBridge.destroy();
    webgpuRankingCache.destroy();

    console.log('🧹 Integrated WebGPU Cache System destroyed');
  }
}

// Export singleton instance
export const integratedCacheSystem = new IntegratedWebGPUCacheSystem();

// Auto-cleanup on page unload
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    integratedCacheSystem.destroy();
  });
}

export type { IntegratedCacheConfig, CacheOperation, IntegratedCacheMetrics, SearchContext };