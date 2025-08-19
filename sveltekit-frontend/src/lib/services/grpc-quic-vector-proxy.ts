/**
 * gRPC Protobuf QUIC Vector Proxy Integration
 * High-performance vector operations with Go microservices
 */

export interface VectorOperation {
  operation: 'search' | 'store' | 'update' | 'delete' | 'batch';
  vectorId?: string;
  embedding?: number[];
  query?: string;
  metadata?: Record<string, any>;
  threshold?: number;
  limit?: number;
}

export interface VectorResult {
  success: boolean;
  data?: any;
  error?: string;
  latency: number;
  protocol: 'quic' | 'grpc' | 'http';
  gpuAccelerated: boolean;
}

export interface ProtocolConfig {
  quic: {
    enabled: boolean;
    url: string;
    timeout: number;
    priority: number;
  };
  grpc: {
    enabled: boolean;
    url: string;
    timeout: number;
    priority: number;
  };
  http: {
    enabled: boolean;
    url: string;
    timeout: number;
    priority: number;
  };
}

/**
 * Multi-protocol vector proxy with automatic fallback
 */
export class GRPCQuicVectorProxy {
  private config: ProtocolConfig;
  private protobufSchema: any = null;
  private grpcClient: any = null;
  private quicClient: any = null;
  private performanceMetrics = new Map<string, number[]>();

  constructor(config: Partial<ProtocolConfig> = {}) {
    this.config = {
      quic: {
        enabled: true,
        url: 'http://localhost:8095', // QUIC proxy service
        timeout: 5000,
        priority: 1 // Highest priority
      },
      grpc: {
        enabled: true,
        url: 'http://localhost:8094', // Enhanced RAG gRPC
        timeout: 15000,
        priority: 2
      },
      http: {
        enabled: true,
        url: 'http://localhost:8093', // HTTP fallback
        timeout: 30000,
        priority: 3 // Lowest priority
      },
      ...config
    };
  }

  /**
   * Initialize protobuf schemas and clients
   */
  async initialize(): Promise<void> {
    console.log('🚀 Initializing gRPC/QUIC vector proxy...');

    try {
      // Initialize protobuf schema (mock implementation)
      this.protobufSchema = await this.loadProtobufSchema();
      
      // Test protocol availability and select best option
      await this.detectAvailableProtocols();
      
      console.log('✅ Vector proxy initialized with multi-protocol support');
    } catch (error) {
      console.error('❌ Vector proxy initialization failed:', error);
      throw error;
    }
  }

  /**
   * Load protobuf schema definition
   */
  private async loadProtobufSchema(): Promise<any> {
    // Mock protobuf schema for legal AI vector operations
    const schema = {
      VectorSearchRequest: {
        query: 'string',
        embedding: 'repeated float',
        limit: 'int32',
        threshold: 'float',
        metadata_filters: 'map<string, string>',
        use_gpu: 'bool'
      },
      VectorSearchResponse: {
        results: 'repeated VectorSearchResult',
        total_found: 'int32',
        processing_time_ms: 'int32',
        gpu_utilized: 'bool'
      },
      VectorSearchResult: {
        id: 'string',
        score: 'float',
        metadata: 'map<string, string>',
        content: 'string'
      }
    };

    console.log('📋 Protobuf schema loaded for vector operations');
    return schema;
  }

  /**
   * Detect available protocols and measure latency
   */
  private async detectAvailableProtocols(): Promise<void> {
    const protocols = ['quic', 'grpc', 'http'] as const;
    
    for (const protocol of protocols) {
      if (!this.config[protocol].enabled) continue;

      try {
        const startTime = performance.now();
        const response = await fetch(`${this.config[protocol].url}/health`, {
          signal: AbortSignal.timeout(this.config[protocol].timeout)
        });
        
        const latency = performance.now() - startTime;
        
        if (response.ok) {
          console.log(`✅ ${protocol.toUpperCase()}: Available (${latency.toFixed(2)}ms)`);
          this.recordLatency(protocol, latency);
        } else {
          console.log(`⚠️ ${protocol.toUpperCase()}: HTTP ${response.status}`);
          this.config[protocol].enabled = false;
        }
      } catch (error) {
        console.log(`🔴 ${protocol.toUpperCase()}: Unavailable`);
        this.config[protocol].enabled = false;
      }
    }
  }

  /**
   * Execute vector operation with protocol fallback
   */
  async executeVectorOperation(operation: VectorOperation): Promise<VectorResult> {
    await this.initialize();

    const startTime = performance.now();
    console.log(`🔄 Executing vector operation: ${operation.operation}`);

    // Try protocols in priority order
    const protocols = (['quic', 'grpc', 'http'] as const)
      .filter(p => this.config[p].enabled)
      .sort((a, b) => this.config[a].priority - this.config[b].priority);

    for (const protocol of protocols) {
      try {
        const result = await this.executeWithProtocol(operation, protocol);
        const latency = performance.now() - startTime;
        
        console.log(`✅ Vector operation complete via ${protocol.toUpperCase()} (${latency.toFixed(2)}ms)`);
        this.recordLatency(protocol, latency);
        
        return {
          ...result,
          latency,
          protocol
        };

      } catch (error) {
        console.warn(`⚠️ ${protocol.toUpperCase()} failed, trying next protocol:`, error.message);
        continue;
      }
    }

    // All protocols failed
    const latency = performance.now() - startTime;
    return {
      success: false,
      error: 'All protocols failed',
      latency,
      protocol: 'http',
      gpuAccelerated: false
    };
  }

  /**
   * Execute operation with specific protocol
   */
  private async executeWithProtocol(
    operation: VectorOperation, 
    protocol: 'quic' | 'grpc' | 'http'
  ): Promise<Omit<VectorResult, 'latency' | 'protocol'>> {
    const config = this.config[protocol];
    
    switch (protocol) {
      case 'quic':
        return this.executeQuicOperation(operation, config);
      case 'grpc':
        return this.executeGrpcOperation(operation, config);
      case 'http':
        return this.executeHttpOperation(operation, config);
      default:
        throw new Error(`Unsupported protocol: ${protocol}`);
    }
  }

  /**
   * Execute via QUIC protocol (highest performance)
   */
  private async executeQuicOperation(
    operation: VectorOperation,
    config: any
  ): Promise<Omit<VectorResult, 'latency' | 'protocol'>> {
    // QUIC implementation (using HTTP/3 simulation for now)
    const response = await fetch(`${config.url}/api/quic/vector`, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'X-Protocol': 'QUIC',
        'X-GPU-Acceleration': 'true'
      },
      body: JSON.stringify({
        operation: operation.operation,
        data: operation,
        protocol_version: 'QUIC/1.0',
        gpu_layers: 35 // RTX 3060 Ti
      }),
      signal: AbortSignal.timeout(config.timeout)
    });

    if (!response.ok) {
      throw new Error(`QUIC error: HTTP ${response.status}`);
    }

    const result = await response.json();
    return {
      success: result.success !== false,
      data: result.data || result,
      gpuAccelerated: result.gpu_accelerated || true
    };
  }

  /**
   * Execute via gRPC protocol
   */
  private async executeGrpcOperation(
    operation: VectorOperation,
    config: any
  ): Promise<Omit<VectorResult, 'latency' | 'protocol'>> {
    // gRPC implementation (HTTP/2 with protobuf simulation)
    const response = await fetch(`${config.url}/api/grpc/vector`, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/x-protobuf',
        'X-Protocol': 'gRPC',
        'X-GPU-Enabled': 'true'
      },
      body: JSON.stringify({
        operation: operation.operation,
        request: operation,
        protobuf_schema: 'VectorSearchRequest',
        gpu_config: {
          device: 'RTX3060Ti',
          memory_limit: '6GB',
          batch_size: 8
        }
      }),
      signal: AbortSignal.timeout(config.timeout)
    });

    if (!response.ok) {
      throw new Error(`gRPC error: HTTP ${response.status}`);
    }

    const result = await response.json();
    return {
      success: result.success !== false,
      data: result.data || result,
      gpuAccelerated: result.gpu_utilized || false
    };
  }

  /**
   * Execute via HTTP protocol (fallback)
   */
  private async executeHttpOperation(
    operation: VectorOperation,
    config: any
  ): Promise<Omit<VectorResult, 'latency' | 'protocol'>> {
    const response = await fetch(`${config.url}/api/vector`, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'X-Protocol': 'HTTP'
      },
      body: JSON.stringify(operation),
      signal: AbortSignal.timeout(config.timeout)
    });

    if (!response.ok) {
      throw new Error(`HTTP error: HTTP ${response.status}`);
    }

    const result = await response.json();
    return {
      success: result.success !== false,
      data: result.data || result,
      gpuAccelerated: false // HTTP typically doesn't use GPU
    };
  }

  /**
   * Vector similarity search with protocol optimization
   */
  async vectorSearch(
    query: string,
    embedding: number[],
    options: {
      limit?: number;
      threshold?: number;
      useGPU?: boolean;
      preferredProtocol?: 'quic' | 'grpc' | 'http';
    } = {}
  ): Promise<VectorResult> {
    const operation: VectorOperation = {
      operation: 'search',
      query,
      embedding,
      threshold: options.threshold || 0.7,
      limit: options.limit || 10,
      metadata: {
        use_gpu: options.useGPU !== false,
        preferred_protocol: options.preferredProtocol || 'quic'
      }
    };

    return this.executeVectorOperation(operation);
  }

  /**
   * Store vector with metadata
   */
  async storeVector(
    vectorId: string,
    embedding: number[],
    metadata: Record<string, any> = {}
  ): Promise<VectorResult> {
    const operation: VectorOperation = {
      operation: 'store',
      vectorId,
      embedding,
      metadata: {
        ...metadata,
        stored_at: new Date().toISOString(),
        model: 'nomic-embed-text'
      }
    };

    return this.executeVectorOperation(operation);
  }

  /**
   * Batch vector operations for high throughput
   */
  async batchVectorOperations(operations: VectorOperation[]): Promise<VectorResult[]> {
    console.log(`📦 Executing ${operations.length} vector operations in batch...`);

    const batchOperation: VectorOperation = {
      operation: 'batch',
      metadata: {
        operations,
        batch_size: operations.length,
        use_gpu: true
      }
    };

    const result = await this.executeVectorOperation(batchOperation);
    
    if (result.success && Array.isArray(result.data)) {
      return result.data as VectorResult[];
    }

    // Fallback: execute individually
    console.warn('⚠️ Batch operation failed, executing individually...');
    const results: VectorResult[] = [];
    
    for (const op of operations) {
      try {
        const opResult = await this.executeVectorOperation(op);
        results.push(opResult);
      } catch (error) {
        results.push({
          success: false,
          error: error.message,
          latency: 0,
          protocol: 'http',
          gpuAccelerated: false
        });
      }
    }

    return results;
  }

  /**
   * Record latency metrics for protocol optimization
   */
  private recordLatency(protocol: string, latency: number): void {
    if (!this.performanceMetrics.has(protocol)) {
      this.performanceMetrics.set(protocol, []);
    }
    
    const metrics = this.performanceMetrics.get(protocol)!;
    metrics.push(latency);
    
    // Keep only last 100 measurements
    if (metrics.length > 100) {
      metrics.shift();
    }
  }

  /**
   * Get performance statistics
   */
  getPerformanceStats() {
    const stats: Record<string, any> = {};
    
    for (const [protocol, latencies] of this.performanceMetrics) {
      if (latencies.length === 0) continue;
      
      const sorted = [...latencies].sort((a, b) => a - b);
      stats[protocol] = {
        count: latencies.length,
        avg: latencies.reduce((sum, val) => sum + val, 0) / latencies.length,
        min: sorted[0],
        max: sorted[sorted.length - 1],
        p50: sorted[Math.floor(sorted.length * 0.5)],
        p95: sorted[Math.floor(sorted.length * 0.95)],
        p99: sorted[Math.floor(sorted.length * 0.99)]
      };
    }
    
    return stats;
  }

  /**
   * Get optimal protocol based on performance metrics
   */
  getOptimalProtocol(): 'quic' | 'grpc' | 'http' {
    const stats = this.getPerformanceStats();
    
    // Sort protocols by average latency
    const sortedProtocols = Object.entries(stats)
      .sort(([,a], [,b]) => a.avg - b.avg)
      .map(([protocol]) => protocol) as ('quic' | 'grpc' | 'http')[];
    
    // Return fastest available protocol
    for (const protocol of sortedProtocols) {
      if (this.config[protocol].enabled) {
        return protocol;
      }
    }
    
    // Default fallback
    return 'http';
  }

  /**
   * Health check all protocols
   */
  async healthCheck(): Promise<Record<string, any>> {
    const health: Record<string, any> = {};
    
    const protocols = ['quic', 'grpc', 'http'] as const;
    
    for (const protocol of protocols) {
      if (!this.config[protocol].enabled) {
        health[protocol] = { status: 'disabled' };
        continue;
      }
      
      try {
        const startTime = performance.now();
        const response = await fetch(`${this.config[protocol].url}/health`, {
          signal: AbortSignal.timeout(5000)
        });
        
        const latency = performance.now() - startTime;
        
        health[protocol] = {
          status: response.ok ? 'healthy' : 'error',
          latency,
          httpStatus: response.status
        };
        
      } catch (error) {
        health[protocol] = {
          status: 'unreachable',
          error: error.message
        };
      }
    }
    
    return health;
  }

  /**
   * Vector search with automatic protocol selection
   */
  async search(
    queryEmbedding: number[],
    options: {
      query?: string;
      limit?: number;
      threshold?: number;
      useGPU?: boolean;
      forceProtocol?: 'quic' | 'grpc' | 'http';
    } = {}
  ): Promise<VectorResult> {
    const operation: VectorOperation = {
      operation: 'search',
      embedding: queryEmbedding,
      query: options.query,
      limit: options.limit || 10,
      threshold: options.threshold || 0.7,
      metadata: {
        use_gpu: options.useGPU !== false,
        force_protocol: options.forceProtocol
      }
    };

    return this.executeVectorOperation(operation);
  }

  /**
   * Store vector with optimized protocol
   */
  async store(
    vectorId: string,
    embedding: number[],
    metadata: Record<string, any> = {}
  ): Promise<VectorResult> {
    const operation: VectorOperation = {
      operation: 'store',
      vectorId,
      embedding,
      metadata: {
        ...metadata,
        model: 'nomic-embed-text',
        created_at: new Date().toISOString()
      }
    };

    return this.executeVectorOperation(operation);
  }

  /**
   * Integration with Go microservices for llama.cpp GPU parsing
   */
  async parseWithLlamaCppGPU(
    content: string,
    options: {
      model?: 'gemma3-legal';
      gpuLayers?: number;
      contextSize?: number;
      temperature?: number;
    } = {}
  ): Promise<{
    parsed: string;
    entities: string[];
    summary: string;
    confidence: number;
    gpuUtilization: number;
  }> {
    console.log('⚡ Calling llama.cpp GPU parsing via Go microservice...');

    try {
      const response = await fetch(`${this.config.grpc.url}/api/llama-cpp-parse`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'X-GPU-Acceleration': 'true'
        },
        body: JSON.stringify({
          content,
          model: options.model || 'gemma3-legal',
          gpu_config: {
            ngl: options.gpuLayers || 35,
            ctx_size: options.contextSize || 4096,
            temperature: options.temperature || 0.1
          },
          parsing_type: 'legal_document'
        })
      });

      if (!response.ok) {
        throw new Error(`llama.cpp GPU parsing failed: HTTP ${response.status}`);
      }

      const result = await response.json();
      console.log('🎯 llama.cpp GPU parsing complete');

      return {
        parsed: result.parsed_content || content,
        entities: result.extracted_entities || [],
        summary: result.summary || '',
        confidence: result.confidence || 0.8,
        gpuUtilization: result.gpu_utilization || 0
      };

    } catch (error) {
      console.error('❌ llama.cpp GPU parsing failed:', error);
      throw error;
    }
  }

  /**
   * Get service status and configuration
   */
  getStatus() {
    return {
      protocols: {
        quic: { 
          enabled: this.config.quic.enabled, 
          url: this.config.quic.url,
          priority: this.config.quic.priority
        },
        grpc: { 
          enabled: this.config.grpc.enabled, 
          url: this.config.grpc.url,
          priority: this.config.grpc.priority
        },
        http: { 
          enabled: this.config.http.enabled, 
          url: this.config.http.url,
          priority: this.config.http.priority
        }
      },
      optimalProtocol: this.getOptimalProtocol(),
      performanceMetrics: this.getPerformanceStats(),
      protobufSchema: !!this.protobufSchema
    };
  }
}

// Global proxy instance
export const vectorProxy = new GRPCQuicVectorProxy({
  quic: {
    enabled: true,
    url: 'http://localhost:8095',
    timeout: 5000,
    priority: 1
  },
  grpc: {
    enabled: true,
    url: 'http://localhost:8094',
    timeout: 15000,
    priority: 2
  },
  http: {
    enabled: true,
    url: 'http://localhost:8093',
    timeout: 30000,
    priority: 3
  }
});

// Auto-initialize
if (typeof window !== 'undefined') {
  vectorProxy.initialize().catch(console.warn);
}

// Legal AI specific vector operations
export class LegalVectorOperations {
  /**
   * Search legal precedents using vector similarity
   */
  static async searchPrecedents(
    caseDescription: string,
    embedding: number[]
  ): Promise<VectorResult> {
    return vectorProxy.search(embedding, {
      query: caseDescription,
      limit: 15,
      threshold: 0.75,
      useGPU: true
    });
  }

  /**
   * Store case evidence with vector indexing
   */
  static async storeEvidence(
    evidenceId: string,
    content: string,
    embedding: number[],
    metadata: Record<string, any>
  ): Promise<VectorResult> {
    return vectorProxy.store(evidenceId, embedding, {
      ...metadata,
      type: 'legal_evidence',
      content_preview: content.slice(0, 200),
      indexed_by: 'legal-ai-system'
    });
  }

  /**
   * Batch process case documents
   */
  static async batchProcessCaseDocuments(
    documents: Array<{
      id: string;
      content: string;
      embedding: number[];
      metadata: any;
    }>
  ): Promise<VectorResult[]> {
    const operations: VectorOperation[] = documents.map(doc => ({
      operation: 'store',
      vectorId: doc.id,
      embedding: doc.embedding,
      metadata: {
        ...doc.metadata,
        content_hash: this.hashContent(doc.content),
        processing_batch: new Date().toISOString()
      }
    }));

    return vectorProxy.batchVectorOperations(operations);
  }

  private static hashContent(content: string): string {
    // Simple hash for content deduplication
    let hash = 0;
    for (let i = 0; i < content.length; i++) {
      const char = content.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return hash.toString(16);
  }
}

export { LegalVectorOperations };