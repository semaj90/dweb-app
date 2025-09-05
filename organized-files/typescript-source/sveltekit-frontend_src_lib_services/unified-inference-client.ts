/**
 * Unified Inference Client
 * Connects XState + Loki.js client caching with Redis-backed inference pipeline
 */

import { inferenceCacheService, type InferenceRequest } from '$lib/stores/inference-cache';
import { writable, derived, get } from 'svelte/store';
import { browser } from '$app/environment';

// Enhanced types for unified pipeline
interface UnifiedInferenceRequest extends Omit<InferenceRequest, 'id'> {
  type: 'tokenize' | 'embed' | 'generate' | 'similarity' | 'legal_analysis';
  cacheStrategy?: 'client-only' | 'server-only' | 'hybrid' | 'no-cache';
  priority?: 'low' | 'medium' | 'high';
}

interface InferenceResponse {
  requestId: string;
  cached: boolean;
  processingTime: number;
  data: any;
  cacheLevel?: 'client' | 'server' | 'none';
}

interface PipelineHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  services: {
    redis: boolean;
    goGateway: boolean;
    pythonGpu: boolean;
    clientCache: boolean;
  };
  cache: {
    clientHitRate: number;
    serverHitRate: number;
    totalRequests: number;
  };
}

// Pipeline statistics
interface PipelineStats {
  requests: {
    total: number;
    cached: number;
    errors: number;
  };
  performance: {
    averageLatency: number;
    cacheHitRate: number;
    errorRate: number;
  };
  usage: {
    tokenizations: number;
    embeddings: number;
    generations: number;
    similarities: number;
    legalAnalyses: number;
  };
}

export class UnifiedInferenceClient {
  private static instance: UnifiedInferenceClient;
  private apiEndpoint = '/api/ai/inference-pipeline';
  private requestCount = 0;
  private errorCount = 0;
  private totalLatency = 0;

  // Reactive stores
  public isProcessing = writable(false);
  public lastResponse = writable<InferenceResponse | null>(null);
  public pipelineHealth = writable<PipelineHealth | null>(null);
  public stats = writable<PipelineStats>({
    requests: { total: 0, cached: 0, errors: 0 },
    performance: { averageLatency: 0, cacheHitRate: 0, errorRate: 0 },
    usage: { tokenizations: 0, embeddings: 0, generations: 0, similarities: 0, legalAnalyses: 0 }
  });

  static getInstance(): UnifiedInferenceClient {
    if (!UnifiedInferenceClient.instance) {
      UnifiedInferenceClient.instance = new UnifiedInferenceClient();
    }
    return UnifiedInferenceClient.instance;
  }

  async initialize(): Promise<void> {
    if (!browser) return;

    // Initialize client-side caching
    await inferenceCacheService.initialize();

    // Start health monitoring
    this.startHealthMonitoring();

    // Setup performance monitoring
    this.setupPerformanceTracking();

    console.log('✅ Unified Inference Client initialized');
  }

  // Main inference method with multi-level caching
  async processRequest(request: UnifiedInferenceRequest): Promise<InferenceResponse> {
    const startTime = Date.now();
    this.isProcessing.set(true);
    
    try {
      let response: InferenceResponse;
      const { cacheStrategy = 'hybrid', type, ...requestData } = request;

      // Determine caching strategy
      switch (cacheStrategy) {
        case 'client-only':
          response = await this.processWithClientCache(type, requestData);
          break;
        case 'server-only':
          response = await this.processWithServerCache(type, requestData);
          break;
        case 'hybrid':
          response = await this.processWithHybridCache(type, requestData);
          break;
        case 'no-cache':
          response = await this.processWithoutCache(type, requestData);
          break;
        default:
          response = await this.processWithHybridCache(type, requestData);
      }

      // Update statistics
      this.updateStats(response, Date.now() - startTime);
      this.lastResponse.set(response);

      return response;
    } catch (error) {
      this.errorCount++;
      this.updateStats(null, Date.now() - startTime, error);
      throw error;
    } finally {
      this.isProcessing.set(false);
    }
  }

  // Client-only caching (XState + Loki.js)
  private async processWithClientCache(type: string, data: any): Promise<InferenceResponse> {
    if (type === 'generate') {
      // Use existing XState inference cache for generation
      const inferenceRequest: Omit<InferenceRequest, 'id'> = {
        prompt: data.prompt,
        model: data.model,
        temperature: data.temperature,
        maxTokens: data.maxTokens,
        metadata: data.metadata
      };

      await inferenceCacheService.submitRequest(inferenceRequest);

      // Wait for completion and return result
      return new Promise((resolve, reject) => {
        const unsubscribe = inferenceCacheService['inferenceActor'].subscribe((state) => {
          if (state.value === 'success' && state.context.currentResult) {
            unsubscribe();
            resolve({
              requestId: state.context.currentResult.id,
              cached: state.context.currentResult.cached,
              processingTime: 0,
              data: state.context.currentResult,
              cacheLevel: 'client'
            });
          } else if (state.value === 'error') {
            unsubscribe();
            reject(new Error(state.context.error || 'Client cache processing failed'));
          }
        });
      });
    }

    // For other types, fall back to server cache
    return this.processWithServerCache(type, data);
  }

  // Server-only caching (Redis + Go gateway)
  private async processWithServerCache(type: string, data: any): Promise<InferenceResponse> {
    const response = await fetch(this.apiEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        type,
        data,
        options: { useCache: true }
      })
    });

    if (!response.ok) {
      throw new Error(`Server request failed: ${response.statusText}`);
    }

    const result = await response.json();
    return {
      ...result,
      cacheLevel: result.cached ? 'server' : 'none'
    };
  }

  // Hybrid caching (client first, then server)
  private async processWithHybridCache(type: string, data: any): Promise<InferenceResponse> {
    try {
      // Try client cache first for supported operations
      if (type === 'generate' && this.shouldUseClientCache(data)) {
        return await this.processWithClientCache(type, data);
      }
    } catch (error) {
      console.debug('Client cache miss, falling back to server');
    }

    // Fall back to server cache
    return this.processWithServerCache(type, data);
  }

  // No caching
  private async processWithoutCache(type: string, data: any): Promise<InferenceResponse> {
    const response = await fetch(this.apiEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        type,
        data,
        options: { useCache: false }
      })
    });

    if (!response.ok) {
      throw new Error(`Server request failed: ${response.statusText}`);
    }

    const result = await response.json();
    return {
      ...result,
      cacheLevel: 'none'
    };
  }

  // Specialized legal analysis methods
  async analyzeLegalDocument(
    documentId: string,
    content: string,
    analysisType: 'summary' | 'risks' | 'clauses' | 'compliance' = 'summary',
    caseId?: string
  ): Promise<InferenceResponse> {
    return this.processRequest({
      type: 'legal_analysis',
      prompt: '', // Will be built server-side
      metadata: { 
        documentId, 
        analysisType, 
        caseId,
        documentType: 'legal_document'
      },
      cacheStrategy: 'hybrid',
      priority: 'high'
    });
  }

  async generateEmbeddings(
    texts: string[],
    model: string = 'nomic-embed-text',
    normalize: boolean = true
  ): Promise<InferenceResponse> {
    return this.processRequest({
      type: 'embed',
      prompt: '', // Not used for embeddings
      metadata: { texts, model, normalize },
      cacheStrategy: 'server-only', // Embeddings benefit from server caching
      priority: 'medium'
    });
  }

  async searchSimilarDocuments(
    queryVector: number[],
    caseId?: string,
    documentType?: string,
    limit: number = 10
  ): Promise<InferenceResponse> {
    return this.processRequest({
      type: 'similarity',
      prompt: '', // Not used for similarity
      metadata: { 
        queryVector, 
        limit, 
        threshold: 0.7,
        filters: { caseId, documentType }
      },
      cacheStrategy: 'server-only', // Vector searches benefit from server caching
      priority: 'high'
    });
  }

  // Health monitoring
  private async startHealthMonitoring(): Promise<void> {
    const checkHealth = async () => {
      try {
        const response = await fetch(`${this.apiEndpoint}?health=true`);
        const health = await response.json();
        
        // Add client-side health info
        const clientHealthy = inferenceCacheService.isConnected?.() !== false;
        
        this.pipelineHealth.set({
          ...health,
          services: {
            ...health.services,
            clientCache: clientHealthy
          }
        });
      } catch (error) {
        this.pipelineHealth.set({
          status: 'unhealthy',
          services: { redis: false, goGateway: false, pythonGpu: false, clientCache: false },
          cache: { clientHitRate: 0, serverHitRate: 0, totalRequests: 0 }
        });
      }
    };

    // Check health every 30 seconds
    setInterval(checkHealth, 30000);
    await checkHealth(); // Initial check
  }

  private setupPerformanceTracking(): void {
    // Subscribe to cache stats
    if (typeof inferenceCacheService.getCacheStats === 'function') {
      setInterval(() => {
        const clientStats = inferenceCacheService.getCacheStats();
        this.updateClientStats(clientStats);
      }, 10000);
    }
  }

  private shouldUseClientCache(data: any): boolean {
    // Use client cache for short prompts and common queries
    return data.prompt && 
           data.prompt.length < 500 && 
           !data.stream && 
           (data.temperature || 0.7) < 0.9;
  }

  private updateStats(response: InferenceResponse | null, latency: number, error?: any): void {
    this.requestCount++;
    this.totalLatency += latency;

    const currentStats = get(this.stats);
    
    const updatedStats: PipelineStats = {
      requests: {
        total: this.requestCount,
        cached: response?.cached ? currentStats.requests.cached + 1 : currentStats.requests.cached,
        errors: error ? this.errorCount : currentStats.requests.errors
      },
      performance: {
        averageLatency: this.totalLatency / this.requestCount,
        cacheHitRate: (currentStats.requests.cached / this.requestCount) * 100,
        errorRate: (this.errorCount / this.requestCount) * 100
      },
      usage: {
        ...currentStats.usage,
        // Increment based on request type (would need to track per type)
      }
    };

    this.stats.set(updatedStats);
  }

  private updateClientStats(clientStats: any): void {
    // Update with client-side cache statistics
    const currentStats = get(this.stats);
    // Merge client stats with server stats
    this.stats.set({
      ...currentStats,
      // Add client-specific metrics
    });
  }

  // Convenience methods
  async generateText(
    prompt: string,
    options: {
      model?: string;
      temperature?: number;
      maxTokens?: number;
      stream?: boolean;
      cacheStrategy?: 'client-only' | 'server-only' | 'hybrid' | 'no-cache';
    } = {}
  ): Promise<InferenceResponse> {
    return this.processRequest({
      type: 'generate',
      prompt,
      model: options.model || 'gemma3-legal',
      temperature: options.temperature || 0.7,
      maxTokens: options.maxTokens || 500,
      metadata: { stream: options.stream || false },
      cacheStrategy: options.cacheStrategy || 'hybrid',
      priority: 'medium'
    });
  }

  // Prefetch common legal queries
  async prefetchLegalQueries(caseId: string, documentTypes: string[]): Promise<void> {
    const commonQueries = [
      `Summarize key points for case ${caseId}`,
      `Analyze risks in case ${caseId} documents`,
      `Extract important dates from case ${caseId}`,
      ...documentTypes.map(type => `Review ${type} documents for case ${caseId}`)
    ];

    // Prefetch using low priority
    for (const query of commonQueries) {
      this.processRequest({
        type: 'generate',
        prompt: query,
        model: 'gemma3-legal',
        temperature: 0.5,
        maxTokens: 200,
        cacheStrategy: 'server-only',
        priority: 'low'
      }).catch(() => {
        // Silently handle prefetch errors
      });
    }
  }

  // Cleanup
  async disconnect(): Promise<void> {
    // Cleanup resources if needed
    console.log('🔌 Unified Inference Client disconnected');
  }
}

// Export singleton instance
export const unifiedInferenceClient = UnifiedInferenceClient.getInstance();

// Initialize in browser
if (browser) {
  unifiedInferenceClient.initialize().catch(console.error);
}

// Export reactive stores for components
export const isProcessing = unifiedInferenceClient.isProcessing;
export const lastResponse = unifiedInferenceClient.lastResponse;
export const pipelineHealth = unifiedInferenceClient.pipelineHealth;
export const pipelineStats = unifiedInferenceClient.stats;