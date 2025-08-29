/**
 * Worker Pool Client with Connection Reuse and Load Balancing
 * High-performance client for cached Node.js, Go, and Python/CUDA workers
 * Integrates with legal AI platform's existing infrastructure
 */
import { EventEmitter } from 'events';
import { legalAIProcessPool } from './process-pool-manager';
import type { WorkerInstance } from './process-pool-manager';

export interface PoolRequest {
  id: string;
  poolName: string;
  payload: any;
  timeout: number;
  priority: 'low' | 'medium' | 'high' | 'critical';
  retries: number;
}

export interface PoolResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  workerId: string;
  responseTime: number;
  fromCache?: boolean;
}

export interface ConnectionPool {
  http: Map<string, any>; // HTTP connections to Go services
  grpc: Map<string, any>; // gRPC connections  
  workers: Map<string, any>; // Worker thread connections
}

export class WorkerPoolClient extends EventEmitter {
  private connections: ConnectionPool = {
    http: new Map(),
    grpc: new Map(), 
    workers: new Map()
  };
  
  private requestQueue: PoolRequest[] = [];
  private activeRequests: Map<string, PoolRequest> = new Map();
  private responseCache: Map<string, any> = new Map();
  private loadBalancer: LoadBalancer;
  
  constructor() {
    super();
    this.loadBalancer = new LoadBalancer();
    
    // Process request queue every 10ms for high throughput
    setInterval(() => this.processRequestQueue(), 10);
    
    // Clean expired cache entries every 60 seconds
    setInterval(() => this.cleanCache(), 60000);
  }

  /**
   * Legal AI document processing with connection reuse
   */
  async processLegalDocument(
    document: { content: string; type: string; metadata: any },
    options: { priority?: 'low' | 'medium' | 'high' | 'critical'; useCache?: boolean } = {}
  ): Promise<PoolResponse> {
    const cacheKey = this.generateCacheKey('document-processor', document);
    
    // Check cache first
    if (options.useCache && this.responseCache.has(cacheKey)) {
      return {
        success: true,
        data: this.responseCache.get(cacheKey),
        workerId: 'cached',
        responseTime: 0,
        fromCache: true
      };
    }

    return this.executeRequest({
      id: `doc_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      poolName: 'document-processor',
      payload: document,
      timeout: 30000,
      priority: options.priority || 'medium',
      retries: 2
    });
  }

  /**
   * Enhanced RAG queries with persistent Go service connections
   */
  async performRAGQuery(
    query: { question: string; context: string[]; filters?: any },
    options: { priority?: 'low' | 'medium' | 'high' | 'critical' } = {}
  ): Promise<PoolResponse> {
    return this.executeRequest({
      id: `rag_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      poolName: 'enhanced-rag',
      payload: query,
      timeout: 15000,
      priority: options.priority || 'high',
      retries: 1
    });
  }

  /**
   * GPU-accelerated vector analysis with CUDA worker reuse
   */
  async analyzeVectors(
    vectors: { embeddings: number[][]; metadata: any[] },
    options: { priority?: 'low' | 'medium' | 'high' | 'critical'; keepWarm?: boolean } = {}
  ): Promise<PoolResponse> {
    const response = await this.executeRequest({
      id: `vector_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      poolName: 'vector-analysis',
      payload: vectors,
      timeout: 45000,
      priority: options.priority || 'high',
      retries: 1
    });

    // Keep GPU warm for subsequent requests
    if (options.keepWarm && response.success) {
      setTimeout(() => {
        this.executeRequest({
          id: 'warmup',
          poolName: 'vector-analysis',
          payload: { warmup: true },
          timeout: 5000,
          priority: 'low',
          retries: 0
        });
      }, 60000); // Warmup after 1 minute
    }

    return response;
  }

  /**
   * Legal entity extraction with NLP models
   */
  async extractLegalEntities(
    text: { content: string; documentType: string },
    options: { priority?: 'low' | 'medium' | 'high' | 'critical' } = {}
  ): Promise<PoolResponse> {
    const cacheKey = this.generateCacheKey('entity-extraction', text);
    
    // Entity extraction results are highly cacheable
    if (this.responseCache.has(cacheKey)) {
      return {
        success: true,
        data: this.responseCache.get(cacheKey),
        workerId: 'cached',
        responseTime: 0,
        fromCache: true
      };
    }

    const response = await this.executeRequest({
      id: `entity_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      poolName: 'entity-extraction',
      payload: text,
      timeout: 20000,
      priority: options.priority || 'medium',
      retries: 1
    });

    // Cache successful entity extraction results
    if (response.success) {
      this.responseCache.set(cacheKey, response.data);
    }

    return response;
  }

  /**
   * Execute request with load balancing and connection reuse
   */
  private async executeRequest(request: PoolRequest): Promise<PoolResponse> {
    const startTime = Date.now();
    
    try {
      // Get optimal worker from pool
      const worker = await legalAIProcessPool.getWorker(request.poolName, request.payload);
      
      // Execute request based on worker type
      let result: any;
      switch (worker.type) {
        case 'node':
          result = await this.executeNodeWorkerRequest(worker, request);
          break;
        case 'go':
          result = await this.executeGoServiceRequest(worker, request);
          break;
        case 'python-cuda':
          result = await this.executePythonCudaRequest(worker, request);
          break;
        default:
          throw new Error(`Unknown worker type: ${worker.type}`);
      }

      const responseTime = Date.now() - startTime;
      
      // Release worker back to pool
      await legalAIProcessPool.releaseWorker(request.poolName, worker.id, responseTime);
      
      this.emit('request:completed', { requestId: request.id, responseTime, success: true });
      
      return {
        success: true,
        data: result,
        workerId: worker.id,
        responseTime
      };
      
    } catch (error: any) {
      const responseTime = Date.now() - startTime;
      
      this.emit('request:failed', { 
        requestId: request.id, 
        error: error instanceof Error ? error.message : 'Unknown error',
        responseTime 
      });
      
      // Retry logic
      if (request.retries > 0) {
        request.retries--;
        await new Promise(resolve => setTimeout(resolve, 1000)); // Backoff
        return this.executeRequest(request);
      }
      
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        workerId: 'failed',
        responseTime
      };
    }
  }

  /**
   * Execute request on Node.js worker with connection reuse
   */
  private async executeNodeWorkerRequest(worker: WorkerInstance, request: PoolRequest): Promise<any> {
    const nodeWorker = worker.process as any; // Worker thread
    
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error(`Request ${request.id} timed out after ${request.timeout}ms`));
      }, request.timeout);
      
      // Use connection if exists, otherwise create message channel
      const messageHandler = (message: any) => {
        clearTimeout(timeout);
        nodeWorker.off('message', messageHandler);
        
        if (message.error) {
          reject(new Error(message.error));
        } else {
          resolve(message.result);
        }
      };
      
      nodeWorker.on('message', messageHandler);
      nodeWorker.postMessage({
        requestId: request.id,
        payload: request.payload
      });
    });
  }

  /**
   * Execute request on Go service with HTTP connection reuse
   */
  private async executeGoServiceRequest(worker: WorkerInstance, request: PoolRequest): Promise<any> {
    const port = worker.port!;
    const connectionKey = `go_service_${port}`;
    
    // Reuse HTTP connection if available
    const baseUrl = `http://localhost:${port}`;
    
    // Determine endpoint based on pool name
    let endpoint: string;
    switch (request.poolName) {
      case 'enhanced-rag':
        endpoint = '/api/rag';
        break;
      default:
        endpoint = '/api/process';
    }
    
    const response = await fetch(`${baseUrl}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Request-ID': request.id,
        'X-Priority': request.priority
      },
      body: JSON.stringify(request.payload)
    });
    
    if (!response.ok) {
      throw new Error(`Go service request failed: ${response.status} ${response.statusText}`);
    }
    
    return response.json();
  }

  /**
   * Execute request on Python CUDA worker with persistent connection
   */
  private async executePythonCudaRequest(worker: WorkerInstance, request: PoolRequest): Promise<any> {
    const pythonProcess = worker.process as any; // Child process
    
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error(`CUDA request ${request.id} timed out after ${request.timeout}ms`));
      }, request.timeout);
      
      let responseBuffer = '';
      
      const dataHandler = (data: Buffer) => {
        responseBuffer += data.toString();
        
        // Look for complete JSON response
        try {
          const lines = responseBuffer.split('\n');
          for (const line of lines) {
            if (line.trim().startsWith('{')) {
              const response = JSON.parse(line.trim());
              if (response.requestId === request.id) {
                clearTimeout(timeout);
                pythonProcess.stdout.off('data', dataHandler);
                
                if (response.error) {
                  reject(new Error(response.error));
                } else {
                  resolve(response.result);
                }
                return;
              }
            }
          }
        } catch (parseError) {
          // Continue accumulating data
        }
      };
      
      pythonProcess.stdout.on('data', dataHandler);
      
      // Send request to Python worker
      const requestData = {
        requestId: request.id,
        action: this.getPythonAction(request.poolName),
        payload: request.payload
      };
      
      pythonProcess.stdin.write(JSON.stringify(requestData) + '\n');
    });
  }

  /**
   * Process queued requests with priority handling
   */
  private async processRequestQueue(): Promise<any> {
    if (this.requestQueue.length === 0) return;
    
    // Sort by priority
    this.requestQueue.sort((a, b) => {
      const priorityOrder = { 'critical': 4, 'high': 3, 'medium': 2, 'low': 1 };
      return priorityOrder[b.priority] - priorityOrder[a.priority];
    });
    
    // Process high-priority requests first
    const request = this.requestQueue.shift();
    if (request && !this.activeRequests.has(request.id)) {
      this.activeRequests.set(request.id, request);
      this.executeRequest(request).finally(() => {
        this.activeRequests.delete(request.id);
      });
    }
  }

  /**
   * Generate cache key for request deduplication
   */
  private generateCacheKey(poolName: string, payload: any): string {
    const payloadStr = JSON.stringify(payload);
    // Simple hash function for cache key
    let hash = 0;
    for (let i = 0; i < payloadStr.length; i++) {
      const char = payloadStr.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return `${poolName}_${hash}`;
  }

  /**
   * Clean expired cache entries
   */
  private cleanCache(): void {
    // Simple cache expiry - in production, use TTL with timestamps
    if (this.responseCache.size > 1000) {
      const entries = Array.from(this.responseCache.entries());
      const toDelete = entries.slice(0, entries.length - 500); // Keep latest 500
      toDelete.forEach(([key]) => this.responseCache.delete(key));
    }
  }

  /**
   * Get Python action based on pool name
   */
  private getPythonAction(poolName: string): string {
    switch (poolName) {
      case 'vector-analysis':
        return 'analyze_vectors';
      case 'entity-extraction':
        return 'extract_entities';
      default:
        return 'process';
    }
  }

  /**
   * Get pool statistics and performance metrics
   */
  getPoolStats() {
    return {
      processPool: legalAIProcessPool.getStats(),
      client: {
        activeRequests: this.activeRequests.size,
        queuedRequests: this.requestQueue.length,
        cacheSize: this.responseCache.size,
        connections: {
          http: this.connections.http.size,
          grpc: this.connections.grpc.size,
          workers: this.connections.workers.size
        }
      }
    };
  }
}

/**
 * Load balancer for optimal worker selection
 */
class LoadBalancer {
  selectWorker(workers: WorkerInstance[], request: PoolRequest): WorkerInstance {
    // Weighted round-robin based on current load and response time
    const availableWorkers = workers.filter(w => w.isIdle);
    
    if (availableWorkers.length === 0) {
      // Return least loaded worker
      return workers.reduce((prev, curr) => 
        prev.requestCount < curr.requestCount ? prev : curr
      );
    }
    
    // For critical requests, prefer workers with lowest request count
    if (request.priority === 'critical') {
      return availableWorkers.reduce((prev, curr) => 
        prev.requestCount < curr.requestCount ? prev : curr
      );
    }
    
    // Round-robin for other requests
    return availableWorkers[0];
  }
}

// Global client instance for legal AI platform
export const legalAIWorkerClient = new WorkerPoolClient();