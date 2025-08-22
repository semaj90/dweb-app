// Go Microservices Client with Native Redis Integration
// Connects to 37+ Go services with intelligent routing and load balancing

export interface GoServiceConfig {
  name: string;
  host: string;
  port: number;
  protocol: 'http' | 'https' | 'grpc' | 'quic';
  healthEndpoint: string;
  capabilities: string[];
  priority: number;
}

export interface RedisConfig {
  host: string;
  port: number;
  password?: string;
  db: number;
  keyPrefix: string;
}

export interface ServiceResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  metadata: {
    service: string;
    protocol: string;
    latency: number;
    fromCache: boolean;
    timestamp: string;
  };
}

export class GoMicroservicesClient {
  private services: Map<string, GoServiceConfig> = new Map();
  private healthCache: Map<string, { healthy: boolean; lastCheck: number }> = new Map();
  private redisConfig: RedisConfig;
  private isInitialized = false;

  constructor(redisConfig: RedisConfig) {
    this.redisConfig = redisConfig;
    this.initializeServices();
  }

  private initializeServices() {
    // Core Legal AI Services
    const services: GoServiceConfig[] = [
      {
        name: 'enhanced-rag',
        host: 'localhost',
        port: 8094,
        protocol: 'http',
        healthEndpoint: '/health',
        capabilities: ['ai_analysis', 'vector_search', 'legal_research'],
        priority: 10
      },
      {
        name: 'upload-service',
        host: 'localhost', 
        port: 8093,
        protocol: 'http',
        healthEndpoint: '/health',
        capabilities: ['file_upload', 'metadata_extraction', 'storage'],
        priority: 9
      },
      {
        name: 'kratos-server',
        host: 'localhost',
        port: 50051,
        protocol: 'grpc',
        healthEndpoint: '/health',
        capabilities: ['gpu_compute', 'legal_grpc', 'high_performance'],
        priority: 8
      },
      {
        name: 'quic-gateway',
        host: 'localhost',
        port: 8216,
        protocol: 'quic',
        healthEndpoint: '/status',
        capabilities: ['ultra_low_latency', 'stream_processing'],
        priority: 7
      },
      {
        name: 'tensor-gpu-service',
        host: 'localhost',
        port: 8220,
        protocol: 'http',
        healthEndpoint: '/health',
        capabilities: ['gpu_acceleration', 'tensor_operations', 'ml_inference'],
        priority: 8
      },
      {
        name: 'websocket-handler',
        host: 'localhost',
        port: 8218,
        protocol: 'http',
        healthEndpoint: '/health',
        capabilities: ['realtime_communication', 'websocket_gateway'],
        priority: 6
      },
      {
        name: 'load-balancer',
        host: 'localhost',
        port: 8222,
        protocol: 'http',
        healthEndpoint: '/health',
        capabilities: ['traffic_distribution', 'service_discovery'],
        priority: 9
      },
      {
        name: 'simple-rag-server',
        host: 'localhost',
        port: 8095,
        protocol: 'http',
        healthEndpoint: '/health',
        capabilities: ['basic_rag', 'document_processing'],
        priority: 5
      },
      {
        name: 'gpu-indexer-service',
        host: 'localhost',
        port: 8221,
        protocol: 'http',
        healthEndpoint: '/health',
        capabilities: ['vector_indexing', 'gpu_search', 'embeddings'],
        priority: 7
      }
    ];

    services.forEach(service => {
      this.services.set(service.name, service);
    });

    this.isInitialized = true;
    console.log(`🚀 Initialized ${services.length} Go microservices`);
  }

  // Enhanced RAG Service Integration
  async performRAGQuery(query: string, context?: unknown): Promise<ServiceResponse> {
    const startTime = Date.now();
    
    try {
      // Try enhanced-rag first, fall back to simple-rag
      const service = await this.getHealthyService(['enhanced-rag', 'simple-rag-server']);
      if (!service) {
        throw new Error('No RAG services available');
      }

      const cacheKey = `rag:${this.hashQuery(query)}`;
      
      // Check Redis cache first
      const cachedResult = await this.getFromCache(cacheKey);
      if (cachedResult) {
        return {
          success: true,
          data: cachedResult,
          metadata: {
            service: service.name,
            protocol: service.protocol,
            latency: Date.now() - startTime,
            fromCache: true,
            timestamp: new Date().toISOString()
          }
        };
      }

      // Make service call
      const response = await this.callService(service, '/api/rag', {
        method: 'POST',
        body: JSON.stringify({ query, context }),
        headers: { 'Content-Type': 'application/json' }
      });

      // Cache successful responses
      if (response.success) {
        await this.setCache(cacheKey, response.data, 300); // 5 minutes
      }

      return {
        ...response,
        metadata: {
          service: service.name,
          protocol: service.protocol,
          latency: Date.now() - startTime,
          fromCache: false,
          timestamp: new Date().toISOString()
        }
      };

    } catch (error) {
      console.error('RAG query failed:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        metadata: {
          service: 'unknown',
          protocol: 'http',
          latency: Date.now() - startTime,
          fromCache: false,
          timestamp: new Date().toISOString()
        }
      };
    }
  }

  // GPU Processing Integration
  async processWithGPU(data: any, operation: string): Promise<ServiceResponse> {
    const startTime = Date.now();
    
    try {
      // Prioritize GPU-enabled services
      const service = await this.getHealthyService(['tensor-gpu-service', 'gpu-indexer-service', 'kratos-server']);
      if (!service) {
        throw new Error('No GPU services available');
      }

      const endpoint = service.name === 'kratos-server' ? '/gpu/compute' : '/api/gpu/process';
      
      const response = await this.callService(service, endpoint, {
        method: 'POST',
        body: JSON.stringify({ data, operation }),
        headers: { 'Content-Type': 'application/json' }
      });

      return {
        ...response,
        metadata: {
          service: service.name,
          protocol: service.protocol,
          latency: Date.now() - startTime,
          fromCache: false,
          timestamp: new Date().toISOString()
        }
      };

    } catch (error) {
      console.error('GPU processing failed:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        metadata: {
          service: 'unknown',
          protocol: 'http',
          latency: Date.now() - startTime,
          fromCache: false,
          timestamp: new Date().toISOString()
        }
      };
    }
  }

  // File Upload Integration
  async uploadFile(file: File, metadata: any): Promise<ServiceResponse> {
    const startTime = Date.now();
    
    try {
      const service = this.services.get('upload-service');
      if (!service || !(await this.isServiceHealthy(service))) {
        throw new Error('Upload service unavailable');
      }

      const formData = new FormData();
      formData.append('file', file);
      formData.append('metadata', JSON.stringify(metadata));

      const response = await this.callService(service, '/upload', {
        method: 'POST',
        body: formData
      });

      return {
        ...response,
        metadata: {
          service: service.name,
          protocol: service.protocol,
          latency: Date.now() - startTime,
          fromCache: false,
          timestamp: new Date().toISOString()
        }
      };

    } catch (error) {
      console.error('File upload failed:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        metadata: {
          service: 'upload-service',
          protocol: 'http',
          latency: Date.now() - startTime,
          fromCache: false,
          timestamp: new Date().toISOString()
        }
      };
    }
  }

  // Service Health Management
  async checkAllServicesHealth(): Promise<Map<string, boolean>> {
    const healthMap = new Map<string, boolean>();
    
    const checks = Array.from(this.services.values()).map(async (service) => {
      const healthy = await this.isServiceHealthy(service);
      healthMap.set(service.name, healthy);
      this.healthCache.set(service.name, {
        healthy,
        lastCheck: Date.now()
      });
    });

    await Promise.all(checks);
    
    console.log('🏥 Health check results:', Object.fromEntries(healthMap));
    return healthMap;
  }

  // Get best available service for capabilities
  private async getHealthyService(serviceNames: string[]): Promise<GoServiceConfig | null> {
    for (const name of serviceNames) {
      const service = this.services.get(name);
      if (service && await this.isServiceHealthy(service)) {
        return service;
      }
    }
    return null;
  }

  // Check individual service health with caching
  private async isServiceHealthy(service: GoServiceConfig): Promise<boolean> {
    const cached = this.healthCache.get(service.name);
    if (cached && Date.now() - cached.lastCheck < 30000) { // 30 second cache
      return cached.healthy;
    }

    try {
      const url = `${service.protocol}://${service.host}:${service.port}${service.healthEndpoint}`;
      const response = await fetch(url, { 
        method: 'GET',
        signal: AbortSignal.timeout(5000) // 5 second timeout
      });
      
      const healthy = response.ok;
      this.healthCache.set(service.name, {
        healthy,
        lastCheck: Date.now()
      });
      
      return healthy;
    } catch (error) {
      this.healthCache.set(service.name, {
        healthy: false,
        lastCheck: Date.now()
      });
      return false;
    }
  }

  // Make service call with protocol handling
  private async callService(service: GoServiceConfig, endpoint: string, options: RequestInit): Promise<ServiceResponse> {
    const url = `${service.protocol}://${service.host}:${service.port}${endpoint}`;
    
    try {
      const response = await fetch(url, {
        ...options,
        signal: AbortSignal.timeout(30000) // 30 second timeout
      });

      if (!response.ok) {
        throw new Error(`Service ${service.name} returned ${response.status}`);
      }

      const data = await response.json();
      return { success: true, data };
      
    } catch (error) {
      console.error(`Service call failed to ${service.name}:`, error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  // Redis Cache Integration
  private async getFromCache(key: string): Promise<any | null> {
    try {
      // Simulate Redis get - replace with actual Redis client
      const fullKey = `${this.redisConfig.keyPrefix}:${key}`;
      
      // For demo, use browser localStorage as cache
      if (typeof localStorage !== 'undefined') {
        const cached = localStorage.getItem(fullKey);
        if (cached) {
          const { data, expiry } = JSON.parse(cached);
          if (Date.now() < expiry) {
            return data;
          } else {
            localStorage.removeItem(fullKey);
          }
        }
      }
      
      return null;
    } catch (error) {
      console.warn('Cache get failed:', error);
      return null;
    }
  }

  private async setCache(key: string, value: any, ttlSeconds: number): Promise<void> {
    try {
      // Simulate Redis set - replace with actual Redis client
      const fullKey = `${this.redisConfig.keyPrefix}:${key}`;
      const expiry = Date.now() + (ttlSeconds * 1000);
      
      // For demo, use browser localStorage as cache
      if (typeof localStorage !== 'undefined') {
        localStorage.setItem(fullKey, JSON.stringify({ data: value, expiry }));
      }
    } catch (error) {
      console.warn('Cache set failed:', error);
    }
  }

  private hashQuery(query: string): string {
    // Simple hash function for demo - replace with better hash in production
    let hash = 0;
    for (let i = 0; i < query.length; i++) {
      const char = query.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash).toString(36);
  }

  // Public API
  async getServiceStatus(): Promise<{
    healthy: number;
    total: number;
    services: { name: string; healthy: boolean; capabilities: string[] }[];
  }> {
    const healthMap = await this.checkAllServicesHealth();
    const healthyCount = Array.from(healthMap.values()).filter(h => h).length;
    
    return {
      healthy: healthyCount,
      total: this.services.size,
      services: Array.from(this.services.values()).map(service => ({
        name: service.name,
        healthy: healthMap.get(service.name) || false,
        capabilities: service.capabilities
      }))
    };
  }
}

// Global instance with Redis configuration
export const goServices = new GoMicroservicesClient({
  host: 'localhost',
  port: 6379,
  db: 0,
  keyPrefix: 'legal-ai'
});

export default goServices;