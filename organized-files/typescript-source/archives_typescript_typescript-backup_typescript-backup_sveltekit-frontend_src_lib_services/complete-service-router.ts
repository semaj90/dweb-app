/**
 * Complete Service Router & Integration Layer
 * Routes all 33 Go microservices with error handling and fallback
 */

import { dev } from '$app/environment';

export interface ServiceEndpoint {
  name: string;
  port: number;
  protocols: string[];
  category: string;
  health: string;
  status: 'running' | 'stopped' | 'unknown';
}

export interface ServiceResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  service?: string;
  protocol?: string;
  latency?: number;
}

export class CompleteServiceRouter {
  private services: Map<string, ServiceEndpoint> = new Map();
  private healthCache: Map<string, { status: boolean; timestamp: number }> = new Map();
  private readonly HEALTH_CACHE_TTL = 30000; // 30 seconds

  constructor() {
    this.initializeServices();
  }

  private initializeServices() {
    // Core Services (Priority 1)
    const coreServices: ServiceEndpoint[] = [
      { name: 'enhanced-rag', port: 8094, protocols: ['HTTP', 'gRPC', 'QUIC', 'WebSocket'], category: 'core', health: '/health', status: 'unknown' },
      { name: 'upload-service', port: 8093, protocols: ['HTTP'], category: 'core', health: '/health', status: 'unknown' },
      { name: 'simple-vector-service', port: 8095, protocols: ['HTTP', 'WebSocket'], category: 'core', health: '/api/health', status: 'unknown' },
      { name: 'grpc-server', port: 50051, protocols: ['gRPC'], category: 'core', health: '/health', status: 'unknown' },
      { name: 'rag-kratos', port: 50052, protocols: ['gRPC'], category: 'core', health: '/health', status: 'unknown' },
      { name: 'cluster-http', port: 8213, protocols: ['HTTP'], category: 'core', health: '/health', status: 'unknown' },
      { name: 'xstate-manager', port: 8212, protocols: ['HTTP'], category: 'core', health: '/health', status: 'unknown' },
      { name: 'gpu-indexer-service', port: 8220, protocols: ['HTTP'], category: 'core', health: '/health', status: 'unknown' }
    ];

    // Performance Services (Priority 2)
    const performanceServices: ServiceEndpoint[] = [
      { name: 'cuda-ai-service', port: 8096, protocols: ['HTTP'], category: 'performance', health: '/health', status: 'unknown' },
      { name: 'advanced-cuda-service', port: 8097, protocols: ['HTTP'], category: 'performance', health: '/health', status: 'unknown' },
      { name: 'gpu-orchestrator-service', port: 8225, protocols: ['HTTP'], category: 'performance', health: '/health', status: 'unknown' },
      { name: 'load-balancer', port: 8224, protocols: ['HTTP'], category: 'performance', health: '/health', status: 'unknown' },
      { name: 'recommendation-service', port: 8223, protocols: ['HTTP'], category: 'performance', health: '/health', status: 'unknown' },
      { name: 'context7-error-pipeline', port: 8219, protocols: ['HTTP'], category: 'performance', health: '/health', status: 'unknown' },
      { name: 'simd-health', port: 8217, protocols: ['HTTP'], category: 'performance', health: '/health', status: 'unknown' },
      { name: 'simd-parser', port: 8218, protocols: ['HTTP'], category: 'performance', health: '/health', status: 'unknown' }
    ];

    // Processing Services (Priority 3)
    const processingServices: ServiceEndpoint[] = [
      { name: 'gin-upload', port: 8207, protocols: ['HTTP'], category: 'processing', health: '/health', status: 'unknown' },
      { name: 'summarizer-service', port: 8209, protocols: ['HTTP'], category: 'processing', health: '/health', status: 'unknown' },
      { name: 'summarizer-http', port: 8210, protocols: ['HTTP'], category: 'processing', health: '/health', status: 'unknown' },
      { name: 'simple-upload', port: 8208, protocols: ['HTTP'], category: 'processing', health: '/health', status: 'unknown' },
      { name: 'simple-upload-fixed', port: 8211, protocols: ['HTTP'], category: 'processing', health: '/health', status: 'unknown' }
    ];

    // Protocol Services (Priority 4)
    const protocolServices: ServiceEndpoint[] = [
      { name: 'quic-ai-stream', port: 8216, protocols: ['QUIC'], category: 'protocol', health: '/health', status: 'unknown' },
      { name: 'quic-gateway', port: 8230, protocols: ['QUIC'], category: 'protocol', health: '/health', status: 'unknown' },
      { name: 'quic-vector-proxy', port: 8231, protocols: ['QUIC'], category: 'protocol', health: '/health', status: 'unknown' },
      { name: 'vector-service', port: 8232, protocols: ['HTTP'], category: 'protocol', health: '/health', status: 'unknown' },
      { name: 'vector-redis-service', port: 8233, protocols: ['HTTP'], category: 'protocol', health: '/health', status: 'unknown' }
    ];

    // Support Services (Priority 5)
    const supportServices: ServiceEndpoint[] = [
      { name: 'cuda-integration-service', port: 8098, protocols: ['HTTP'], category: 'support', health: '/health', status: 'unknown' },
      { name: 'cuda-service', port: 8099, protocols: ['HTTP'], category: 'support', health: '/health', status: 'unknown' },
      { name: 'enhanced-api-endpoints', port: 8201, protocols: ['HTTP'], category: 'support', health: '/health', status: 'unknown' },
      { name: 'simple-api-endpoints', port: 8226, protocols: ['HTTP'], category: 'support', health: '/health', status: 'unknown' },
      { name: 'main-service', port: 8227, protocols: ['HTTP'], category: 'support', health: '/health', status: 'unknown' }
    ];

    // Register all services
    [...coreServices, ...performanceServices, ...processingServices, ...protocolServices, ...supportServices]
      .forEach(service => {
        this.services.set(service.name, service);
      });
  }

  /**
   * Get service endpoint configuration
   */
  getService(name: string): ServiceEndpoint | undefined {
    return this.services.get(name);
  }

  /**
   * Get all services by category
   */
  getServicesByCategory(category: string): ServiceEndpoint[] {
    return Array.from(this.services.values()).filter(service => service.category === category);
  }

  /**
   * Get all services
   */
  getAllServices(): ServiceEndpoint[] {
    return Array.from(this.services.values());
  }

  /**
   * Check service health with caching
   */
  async checkServiceHealth(serviceName: string): Promise<boolean> {
    const service = this.services.get(serviceName);
    if (!service) return false;

    // Check cache first
    const cached = this.healthCache.get(serviceName);
    const now = Date.now();
    
    if (cached && (now - cached.timestamp) < this.HEALTH_CACHE_TTL) {
      return cached.status;
    }

    try {
      const url = `http://localhost:${service.port}${service.health}`;
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000); // 5s timeout

      const response = await fetch(url, {
        signal: controller.signal,
        method: 'GET',
        headers: { 'Accept': 'application/json' }
      });

      clearTimeout(timeoutId);
      const isHealthy = response.ok;
      
      // Update cache
      this.healthCache.set(serviceName, { status: isHealthy, timestamp: now });
      
      // Update service status
      service.status = isHealthy ? 'running' : 'stopped';
      
      return isHealthy;
      
    } catch (error: any) {
      console.warn(`Health check failed for ${serviceName}:`, error);
      this.healthCache.set(serviceName, { status: false, timestamp: now });
      service.status = 'stopped';
      return false;
    }
  }

  /**
   * Route request to service with fallback and error handling
   */
  async routeRequest<T = any>(
    serviceName: string, 
    endpoint: string, 
    options: RequestInit = {},
    fallbackServices?: string[]
  ): Promise<ServiceResponse<T>> {
    const startTime = Date.now();
    
    // Try primary service
    const result = await this.tryServiceRequest<T>(serviceName, endpoint, options);
    if (result.success) {
      result.latency = Date.now() - startTime;
      return result;
    }

    // Try fallback services
    if (fallbackServices && fallbackServices.length > 0) {
      for (const fallbackName of fallbackServices) {
        console.warn(`Trying fallback service: ${fallbackName} for ${serviceName}`);
        const fallbackResult = await this.tryServiceRequest<T>(fallbackName, endpoint, options);
        if (fallbackResult.success) {
          fallbackResult.latency = Date.now() - startTime;
          return fallbackResult;
        }
      }
    }

    return {
      success: false,
      error: `All services failed for ${serviceName}`,
      service: serviceName,
      latency: Date.now() - startTime
    };
  }

  private async tryServiceRequest<T>(
    serviceName: string, 
    endpoint: string, 
    options: RequestInit
  ): Promise<ServiceResponse<T>> {
    const service = this.services.get(serviceName);
    if (!service) {
      return { success: false, error: `Service ${serviceName} not found`, service: serviceName };
    }

    try {
      const url = `http://localhost:${service.port}${endpoint}`;
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s timeout

      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers
        }
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        return {
          success: false,
          error: `HTTP ${response.status}: ${response.statusText}`,
          service: serviceName
        };
      }

      let data: T;
      const contentType = response.headers.get('content-type');
      
      if (contentType?.includes('application/json')) {
        data = await response.json();
      } else {
        data = await response.text() as T;
      }

      return {
        success: true,
        data,
        service: serviceName,
        protocol: 'HTTP'
      };

    } catch (error: any) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        service: serviceName
      };
    }
  }

  /**
   * Enhanced RAG service with multiple protocol support
   */
  async queryEnhancedRAG(query: string, context?: any): Promise<ServiceResponse<any>> {
    const payload = { query, context, timestamp: new Date().toISOString() };
    
    return this.routeRequest(
      'enhanced-rag',
      '/api/rag/query',
      {
        method: 'POST',
        body: JSON.stringify(payload)
      },
      ['cuda-ai-service', 'advanced-cuda-service'] // Fallback to CUDA services
    );
  }

  /**
   * Upload service with multiple upload endpoint support
   */
  async uploadFile(file: File, metadata?: any): Promise<ServiceResponse<any>> {
    const formData = new FormData();
    formData.append('file', file);
    if (metadata) {
      formData.append('metadata', JSON.stringify(metadata));
    }

    return this.routeRequest(
      'upload-service',
      '/api/upload',
      {
        method: 'POST',
        body: formData
      },
      ['gin-upload', 'simple-upload-fixed'] // Fallback upload services
    );
  }

  /**
   * Vector similarity search with fallback
   */
  async vectorSearch(query: string, limit: number = 10): Promise<ServiceResponse<any>> {
    const payload = { query, limit, model: 'nomic-embed-text' };
    
    return this.routeRequest(
      'simple-vector-service',
      '/api/vector/search',
      {
        method: 'POST',
        body: JSON.stringify(payload)
      },
      ['vector-service', 'vector-redis-service'] // Fallback vector services
    );
  }

  /**
   * GPU processing with multiple CUDA services
   */
  async processWithGPU(data: any, operation: string): Promise<ServiceResponse<any>> {
    const payload = { data, operation, timestamp: new Date().toISOString() };
    
    return this.routeRequest(
      'cuda-ai-service',
      '/api/gpu/process',
      {
        method: 'POST',
        body: JSON.stringify(payload)
      },
      ['advanced-cuda-service', 'gpu-orchestrator-service'] // Multiple GPU fallbacks
    );
  }

  /**
   * Document summarization with fallback services
   */
  async summarizeDocument(content: string, options?: any): Promise<ServiceResponse<any>> {
    const payload = { content, options: options || {} };
    
    return this.routeRequest(
      'summarizer-service',
      '/api/summarize',
      {
        method: 'POST',
        body: JSON.stringify(payload)
      },
      ['summarizer-http'] // Fallback summarizer
    );
  }

  /**
   * Cluster management operations
   */
  async clusterOperation(operation: string, params?: any): Promise<ServiceResponse<any>> {
    const payload = { operation, params: params || {}, timestamp: new Date().toISOString() };
    
    return this.routeRequest(
      'cluster-http',
      '/api/cluster',
      {
        method: 'POST',
        body: JSON.stringify(payload)
      }
    );
  }

  /**
   * Health check all services
   */
  async healthCheckAll(): Promise<{ [category: string]: { [serviceName: string]: boolean } }> {
    const healthResults: { [category: string]: { [serviceName: string]: boolean } } = {
      core: {},
      performance: {},
      processing: {},
      protocol: {},
      support: {}
    };

    const healthPromises: Promise<void>[] = [];

    for (const [serviceName, service] of this.services.entries()) {
      healthPromises.push(
        this.checkServiceHealth(serviceName).then(isHealthy => {
          healthResults[service.category][serviceName] = isHealthy;
        })
      );
    }

    await Promise.all(healthPromises);
    return healthResults;
  }

  /**
   * Get service statistics
   */
  getServiceStats() {
    const services = this.getAllServices();
    const stats = {
      total: services.length,
      byCategory: {} as { [key: string]: number },
      byStatus: { running: 0, stopped: 0, unknown: 0 },
      byProtocol: {} as { [key: string]: number }
    };

    services.forEach(service => {
      // Category stats
      stats.byCategory[service.category] = (stats.byCategory[service.category] || 0) + 1;
      
      // Status stats
      stats.byStatus[service.status]++;
      
      // Protocol stats
      service.protocols.forEach(protocol => {
        stats.byProtocol[protocol] = (stats.byProtocol[protocol] || 0) + 1;
      });
    });

    return stats;
  }
}

// Global service router instance
export const serviceRouter = new CompleteServiceRouter();

// Export specific service functions for easy use
export const {
  queryEnhancedRAG,
  uploadFile,
  vectorSearch,
  processWithGPU,
  summarizeDocument,
  clusterOperation,
  healthCheckAll
} = serviceRouter;