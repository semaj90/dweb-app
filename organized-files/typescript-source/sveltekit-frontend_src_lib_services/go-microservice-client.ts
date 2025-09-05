import * as http from "http";
/**
 * Go Microservice Client - SvelteKit Integration Layer
 * Provides type-safe client for communicating with Go microservices via QUIC/gRPC/HTTP protocols
 */

export interface GoServiceConfig {
  name: string;
  http: { host: string; port: number };
  grpc?: { host: string; port: number };
  quic?: { host: string; port: number };
  healthEndpoint?: string;
}

export interface ServiceResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  protocol?: string;
  responseTime?: number;
}

// Service configurations based on available Go microservices
export const GO_SERVICES: Record<string, GoServiceConfig> = {
  enhancedRag: {
    name: 'Enhanced RAG Service',
    http: { host: 'localhost', port: 8094 },
    grpc: { host: 'localhost', port: 8095 },
    healthEndpoint: '/health'
  },
  uploadService: {
    name: 'Upload Service',
    http: { host: 'localhost', port: 8093 },
    healthEndpoint: '/health'
  },
  vectorService: {
    name: 'Vector Service', 
    http: { host: 'localhost', port: 8096 },
    healthEndpoint: '/health'
  },
  clusterService: {
    name: 'Cluster Service',
    http: { host: 'localhost', port: 8213 },
    healthEndpoint: '/health'
  }
};

export class GoMicroserviceClient {
  private serviceConfig: GoServiceConfig;
  private timeout: number;

  constructor(serviceName: keyof typeof GO_SERVICES, timeout = 10000) {
    const config = GO_SERVICES[serviceName];
    if (!config) {
      throw new Error(`Unknown service: ${serviceName}`);
    }
    this.serviceConfig = config;
    this.timeout = timeout;
  }

  /**
   * Make HTTP request with automatic fallback
   */
  async request<T = any>(
    endpoint: string, 
    options: RequestInit = {}
  ): Promise<ServiceResponse<T>> {
    const startTime = Date.now();
    const url = `http://${this.serviceConfig.http.host}:${this.serviceConfig.http.port}${endpoint}`;

    try {
      const response = await fetch(url, {
        ...options,
        signal: AbortSignal.timeout(this.timeout),
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
      });

      const responseTime = Date.now() - startTime;

      if (!response.ok) {
        return {
          success: false,
          error: `HTTP ${response.status}: ${response.statusText}`,
          protocol: 'HTTP',
          responseTime,
        };
      }

      let data: T;
      const contentType = response.headers.get('content-type');
      
      if (contentType?.includes('application/json')) {
        data = await response.json();
      } else {
        data = await response.text() as any;
      }

      return {
        success: true,
        data,
        protocol: 'HTTP',
        responseTime,
      };
    } catch (error: any) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        protocol: 'HTTP',
        responseTime: Date.now() - startTime,
      };
    }
  }

  /**
   * GET request
   */
  async get<T = any>(endpoint: string): Promise<ServiceResponse<T>> {
    return this.request<T>(endpoint, { method: 'GET' });
  }

  /**
   * POST request
   */
  async post<T = any>(endpoint: string, data?: any): Promise<ServiceResponse<T>> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  /**
   * Check service health
   */
  async health(): Promise<ServiceResponse<{ status: string; uptime?: number }>> {
    const healthEndpoint = this.serviceConfig.healthEndpoint || '/health';
    return this.get(healthEndpoint);
  }

  /**
   * Get service configuration
   */
  getConfig(): GoServiceConfig {
    return this.serviceConfig;
  }
}

/**
 * Enhanced RAG Service Client
 */
export class EnhancedRAGClient extends GoMicroserviceClient {
  constructor(timeout?: number) {
    super('enhancedRag', timeout);
  }

  async ragQuery(query: string, options: {
    context?: string[];
    maxResults?: number;
    threshold?: number;
    model?: string;
  } = {}) {
    return this.post('/api/rag', {
      query,
      max_results: options.maxResults || 5,
      threshold: options.threshold || 0.7,
      context: options.context || [],
      model: options.model || 'gemma3-legal'
    });
  }

  async semanticSearch(query: string, options: {
    collection?: string;
    limit?: number;
  } = {}) {
    return this.post('/api/search', {
      query,
      collection: options.collection || 'legal_documents',
      limit: options.limit || 10
    });
  }
}

/**
 * Upload Service Client
 */
export class UploadServiceClient extends GoMicroserviceClient {
  constructor(timeout?: number) {
    super('uploadService', timeout);
  }

  async uploadFile(file: File, metadata?: Record<string, any>) {
    const formData = new FormData();
    formData.append('file', file);
    if (metadata) {
      formData.append('metadata', JSON.stringify(metadata));
    }

    return this.request('/upload', {
      method: 'POST',
      body: formData,
      headers: {
        // Don't set Content-Type, let browser set it for FormData
      }
    });
  }

  async getUploadStatus(uploadId: string) {
    return this.get(`/upload/${uploadId}/status`);
  }
}

/**
 * Service Manager - manages multiple Go services
 */
export class GoServiceManager {
  private clients: Map<string, GoMicroserviceClient> = new Map();

  constructor() {
    // Initialize clients for all available services
    Object.keys(GO_SERVICES).forEach(serviceName => {
      this.clients.set(serviceName, new GoMicroserviceClient(serviceName as any));
    });
  }

  getClient(serviceName: keyof typeof GO_SERVICES): GoMicroserviceClient | undefined {
    return this.clients.get(serviceName);
  }

  getEnhancedRAG(): EnhancedRAGClient {
    return new EnhancedRAGClient();
  }

  getUploadService(): UploadServiceClient {
    return new UploadServiceClient();
  }

  /**
   * Check health of all services
   */
  async checkAllServices(): Promise<Record<string, ServiceResponse<any>>> {
    const results: Record<string, ServiceResponse<any>> = {};
    
    const healthChecks = Array.from(this.clients.entries()).map(async ([name, client]) => {
      try {
        const health = await client.health();
        results[name] = health;
      } catch (error: any) {
        results[name] = {
          success: false,
          error: error instanceof Error ? error.message : 'Unknown error',
        };
      }
    });

    await Promise.allSettled(healthChecks);
    return results;
  }
}

// Export singleton instance
export const goServiceManager = new GoServiceManager();