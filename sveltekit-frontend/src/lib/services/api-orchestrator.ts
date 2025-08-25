/**
 * Comprehensive API Orchestration System - SvelteKit 2 Production
 * Integrates all 37 Go microservices with multi-protocol support
 * Windows-native deployment with intelligent routing and failover
 */

import type { 
  ServiceEndpoints, 
  ServiceTier,
  ProtocolEndpoint,
  DatabaseEndpoint,
  MessagingEndpoint,
  FrontendEndpoint,
  HealthCheckResult,
  APIRequestContext,
  MultiProtocolRequestOptions
} from '$lib/types/api.js';

export class APIOrchestrator {
  private static instance: APIOrchestrator;
  private serviceEndpoints: ServiceEndpoints;
  private healthCache: Map<string, { result: HealthCheckResult; timestamp: number }> = new Map();
  private requestMetrics: Map<string, { count: number; totalTime: number; errors: number }> = new Map();

  private constructor() {
    this.serviceEndpoints = this.initializeServiceEndpoints();
  }

  public static getInstance(): APIOrchestrator {
    if (!APIOrchestrator.instance) {
      APIOrchestrator.instance = new APIOrchestrator();
    }
    return APIOrchestrator.instance;
  }

  /**
   * Initialize all service endpoints with production configuration
   */
  private initializeServiceEndpoints(): ServiceEndpoints {
    return {
      // Core AI Services (Tier 1) - Always Running
      enhancedRAG: {
        http: 'http://localhost:8094',
        grpc: 'localhost:50051',
        quic: 'localhost:8216',
        websocket: 'ws://localhost:8094/ws',
        health: '/health',
        tier: ServiceTier.ULTRA_FAST,
        status: 'active'
      },
      uploadService: {
        http: 'http://localhost:8093',
        health: '/health',
        tier: ServiceTier.STANDARD,
        status: 'active'
      },
      documentProcessor: {
        http: 'http://localhost:8081',
        health: '/api/health',
        tier: ServiceTier.STANDARD,
        status: 'active'
      },
      grpcServer: {
        http: 'http://localhost:50051',
        grpc: 'localhost:50051',
        health: '/health',
        tier: ServiceTier.HIGH_PERF,
        status: 'active'
      },

      // AI Enhancement Services (Tier 2) - Advanced Features
      advancedCUDA: {
        http: 'http://localhost:8095',
        health: '/health',
        tier: ServiceTier.ULTRA_FAST,
        status: 'experimental'
      },
      dimensionalCache: {
        http: 'http://localhost:8097',
        health: '/health',
        tier: ServiceTier.HIGH_PERF,
        status: 'experimental'
      },
      xstateManager: {
        http: 'http://localhost:8212',
        health: '/health',
        tier: ServiceTier.STANDARD,
        status: 'active'
      },
      moduleManager: {
        http: 'http://localhost:8099',
        health: '/health',
        tier: ServiceTier.STANDARD,
        status: 'experimental'
      },
      recommendationEngine: {
        http: 'http://localhost:8100',
        health: '/health',
        tier: ServiceTier.HIGH_PERF,
        status: 'experimental'
      },

      // Specialized AI Services
      enhancedSemanticArchitecture: {
        http: 'http://localhost:8201',
        health: '/health',
        tier: ServiceTier.HIGH_PERF,
        status: 'active'
      },
      enhancedLegalAI: {
        http: 'http://localhost:8202',
        health: '/health',
        tier: ServiceTier.HIGH_PERF,
        status: 'active'
      },
      enhancedMulticore: {
        http: 'http://localhost:8206',
        health: '/health',
        tier: ServiceTier.HIGH_PERF,
        status: 'active'
      },
      liveAgentEnhanced: {
        http: 'http://localhost:8200',
        websocket: 'ws://localhost:8200/ws',
        health: '/health',
        tier: ServiceTier.REALTIME,
        status: 'active'
      },

      // File & Document Services
      ginUpload: {
        http: 'http://localhost:8207',
        health: '/health',
        tier: ServiceTier.STANDARD,
        status: 'active'
      },
      summarizerService: {
        http: 'http://localhost:8209',
        health: '/health',
        tier: ServiceTier.STANDARD,
        status: 'active'
      },
      aiSummary: {
        http: 'http://localhost:8211',
        health: '/health',
        tier: ServiceTier.HIGH_PERF,
        status: 'active'
      },

      // Multi-Core Ollama Cluster
      ollama: {
        primary: 'http://localhost:11434',
        secondary: 'http://localhost:11435',
        embeddings: 'http://localhost:11436',
        health: '/api/tags',
        tier: ServiceTier.HIGH_PERF,
        status: 'active'
      },

      // Database Services
      postgresql: {
        host: 'localhost',
        port: 5432,
        database: 'legal_ai_db',
        status: 'active'
      },
      redis: {
        host: 'localhost',
        port: 6379,
        status: 'active'
      },
      qdrant: {
        http: 'http://localhost:6333',
        health: '/health',
        tier: ServiceTier.HIGH_PERF,
        status: 'active'
      },
      neo4j: {
        http: 'http://localhost:7474',
        health: '/db/data/',
        tier: ServiceTier.STANDARD,
        status: 'active'
      },

      // Messaging & Communication
      nats: {
        server: 'nats://localhost:4225',
        websocket: 'ws://localhost:4226',
        monitor: 'http://localhost:8225',
        health: '/healthz',
        status: 'active'
      },

      // Infrastructure & Monitoring Services
      clusterManager: {
        http: 'http://localhost:8213',
        health: '/health',
        tier: ServiceTier.STANDARD,
        status: 'active'
      },
      loadBalancer: {
        http: 'http://localhost:8224',
        health: '/health',
        tier: ServiceTier.STANDARD,
        status: 'active'
      },
      gpuIndexerService: {
        http: 'http://localhost:8220',
        health: '/health',
        tier: ServiceTier.HIGH_PERF,
        status: 'active'
      },
      contextErrorPipeline: {
        http: 'http://localhost:8219',
        health: '/health',
        tier: ServiceTier.STANDARD,
        status: 'active'
      },
      simdHealth: {
        http: 'http://localhost:8217',
        health: '/health',
        tier: ServiceTier.STANDARD,
        status: 'active'
      },

      // Development & Testing
      simpleServer: {
        http: 'http://localhost:8225',
        health: '/health',
        tier: ServiceTier.STANDARD,
        status: 'active'
      },
      testServer: {
        http: 'http://localhost:8226',
        health: '/health',
        tier: ServiceTier.STANDARD,
        status: 'active'
      },

      // Frontend
      sveltekit: {
        http: 'http://localhost:5173',
        dev: 'http://localhost:5174',
        status: 'active'
      }
    };
  }

  /**
   * Route request to optimal service with protocol selection
   */
  async routeRequest<T extends keyof ServiceEndpoints>(
    service: T,
    endpoint: string,
    options: MultiProtocolRequestOptions = {}
  ): Promise<Response> {
    const serviceConfig = this.serviceEndpoints[service];
    
    if (!serviceConfig || serviceConfig.status === 'maintenance') {
      throw new Error(`Service ${service} not available`);
    }

    const startTime = Date.now();
    const protocol = this.selectOptimalProtocol(serviceConfig, options.protocol);
    const baseUrl = this.getServiceUrl(serviceConfig, protocol);
    
    if (!baseUrl) {
      throw new Error(`No available endpoint for service ${service}`);
    }

    const requestOptions: RequestInit = {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'SvelteKit-Legal-AI-Orchestrator/2.0',
        ...options.headers
      }
    };

    // Add timeout based on service tier
    const timeout = this.getTimeoutForTier(serviceConfig);
    requestOptions.signal = AbortSignal.timeout(timeout);

    let response: Response;
    let error: Error | null = null;

    try {
      response = await fetch(`${baseUrl}${endpoint}`, requestOptions);
      
      // Record metrics
      this.recordMetrics(service, Date.now() - startTime, !response.ok);
      
      return response;
    } catch (fetchError) {
      error = fetchError instanceof Error ? fetchError : new Error('Request failed');
      
      // Attempt fallback if enabled
      if (options.fallback && 'http' in serviceConfig && protocol !== 'http') {
        try {
          const fallbackUrl = serviceConfig.http;
          if (fallbackUrl) {
            console.warn(`Service ${service} failover: ${protocol} → HTTP`);
            response = await fetch(`${fallbackUrl}${endpoint}`, {
              ...requestOptions,
              signal: AbortSignal.timeout(timeout * 2) // Extended timeout for fallback
            });
            
            this.recordMetrics(service, Date.now() - startTime, !response.ok);
            return response;
          }
        } catch (fallbackError) {
          console.error(`Fallback failed for ${service}:`, fallbackError);
        }
      }
      
      this.recordMetrics(service, Date.now() - startTime, true);
      throw error;
    }
  }

  /**
   * Perform comprehensive health check across all services
   */
  async performHealthCheck(): Promise<Record<string, HealthCheckResult>> {
    const healthResults: Record<string, HealthCheckResult> = {};
    const healthPromises: Promise<void>[] = [];

    // Check active services only
    const activeServices = Object.entries(this.serviceEndpoints).filter(
      ([_, config]) => config.status === 'active' || config.status === 'experimental'
    );

    for (const [serviceName, config] of activeServices) {
      healthPromises.push(
        this.checkServiceHealth(serviceName as keyof ServiceEndpoints, config)
          .then(result => {
            healthResults[serviceName] = result;
          })
          .catch(error => {
            healthResults[serviceName] = {
              status: 'error',
              error: String(error),
              lastCheck: new Date().toISOString()
            };
          })
      );
    }

    await Promise.allSettled(healthPromises);
    return healthResults;
  }

  /**
   * Check health of individual service
   */
  private async checkServiceHealth(
    serviceName: keyof ServiceEndpoints,
    config: ProtocolEndpoint | DatabaseEndpoint | MessagingEndpoint | FrontendEndpoint
  ): Promise<HealthCheckResult> {
    const cacheKey = serviceName;
    const cached = this.healthCache.get(cacheKey);
    
    // Return cached result if less than 30 seconds old
    if (cached && Date.now() - cached.timestamp < 30000) {
      return cached.result;
    }

    const startTime = Date.now();
    let healthUrl: string;
    let result: HealthCheckResult;

    try {
      // Determine health check URL
      if ('http' in config && config.http) {
        const healthPath = 'health' in config ? config.health || '/health' : '/health';
        healthUrl = `${config.http}${healthPath}`;
      } else if ('primary' in config && config.primary) {
        const healthPath = 'health' in config ? config.health || '/health' : '/health';
        healthUrl = `${config.primary}${healthPath}`;
      } else if ('server' in config && config.monitor) {
        healthUrl = `${config.monitor}/healthz`;
      } else {
        throw new Error('No health check endpoint available');
      }

      const response = await fetch(healthUrl, {
        method: 'GET',
        signal: AbortSignal.timeout(5000)
      });

      const responseTime = Date.now() - startTime;

      result = {
        status: response.ok ? 'healthy' : 'unhealthy',
        responseTime,
        endpoint: healthUrl,
        lastCheck: new Date().toISOString(),
        metadata: response.ok ? await this.extractHealthMetadata(response) : undefined
      };
    } catch (error) {
      result = {
        status: 'error',
        error: String(error),
        lastCheck: new Date().toISOString()
      };
    }

    // Cache the result
    this.healthCache.set(cacheKey, {
      result,
      timestamp: Date.now()
    });

    return result;
  }

  /**
   * Extract metadata from health check response
   */
  private async extractHealthMetadata(response: Response): Promise<any> {
    try {
      const contentType = response.headers.get('content-type');
      if (contentType?.includes('application/json')) {
        const data = await response.clone().json();
        return {
          version: data.version,
          uptime: data.uptime,
          connections: data.connections,
          memoryUsage: data.memory_usage,
          cpuUsage: data.cpu_usage
        };
      }
    } catch (error) {
      // Ignore JSON parsing errors
    }
    return {};
  }

  /**
   * Select optimal protocol based on service configuration and preferences
   */
  private selectOptimalProtocol(
    config: ProtocolEndpoint | DatabaseEndpoint | MessagingEndpoint | FrontendEndpoint,
    preferredProtocol?: 'auto' | 'http' | 'grpc' | 'quic' | 'websocket'
  ): string {
    if (preferredProtocol && preferredProtocol !== 'auto') {
      return preferredProtocol;
    }

    // Protocol selection based on service tier and availability
    if ('tier' in config) {
      switch (config.tier) {
        case ServiceTier.ULTRA_FAST:
          if ('quic' in config && config.quic) return 'quic';
          if ('grpc' in config && config.grpc) return 'grpc';
          break;
        case ServiceTier.HIGH_PERF:
          if ('grpc' in config && config.grpc) return 'grpc';
          break;
        case ServiceTier.REALTIME:
          if ('websocket' in config && config.websocket) return 'websocket';
          break;
      }
    }

    // Default to HTTP
    return 'http';
  }

  /**
   * Get service URL for specified protocol
   */
  private getServiceUrl(
    config: ProtocolEndpoint | DatabaseEndpoint | MessagingEndpoint | FrontendEndpoint,
    protocol: string
  ): string | null {
    switch (protocol) {
      case 'http':
        if ('http' in config) return config.http || null;
        if ('primary' in config) return config.primary || null;
        if ('monitor' in config) return config.monitor || null;
        break;
      case 'grpc':
        if ('grpc' in config) return `http://${config.grpc}`;
        break;
      case 'quic':
        if ('quic' in config) return `http://${config.quic}`;
        break;
      case 'websocket':
        if ('websocket' in config) return config.websocket || null;
        break;
    }
    return null;
  }

  /**
   * Get timeout based on service tier
   */
  private getTimeoutForTier(config: ProtocolEndpoint | DatabaseEndpoint | MessagingEndpoint | FrontendEndpoint): number {
    if ('tier' in config) {
      switch (config.tier) {
        case ServiceTier.ULTRA_FAST: return 5000;  // 5s
        case ServiceTier.HIGH_PERF: return 15000;  // 15s
        case ServiceTier.REALTIME: return 1000;   // 1s
        case ServiceTier.STANDARD: return 30000;  // 30s
      }
    }
    return 30000; // Default 30s
  }

  /**
   * Record performance metrics
   */
  private recordMetrics(service: string, responseTime: number, hasError: boolean): void {
    const existing = this.requestMetrics.get(service) || { count: 0, totalTime: 0, errors: 0 };
    
    this.requestMetrics.set(service, {
      count: existing.count + 1,
      totalTime: existing.totalTime + responseTime,
      errors: existing.errors + (hasError ? 1 : 0)
    });
  }

  /**
   * Get performance metrics
   */
  getMetrics(): Record<string, any> {
    const metrics: Record<string, any> = {};
    
    for (const [service, stats] of this.requestMetrics.entries()) {
      metrics[service] = {
        requestCount: stats.count,
        averageResponseTime: stats.count > 0 ? Math.round(stats.totalTime / stats.count) : 0,
        errorRate: stats.count > 0 ? Math.round((stats.errors / stats.count) * 100) : 0,
        successRate: stats.count > 0 ? Math.round(((stats.count - stats.errors) / stats.count) * 100) : 0
      };
    }
    
    return metrics;
  }

  /**
   * Get service configuration
   */
  getServiceConfig<T extends keyof ServiceEndpoints>(service: T): ServiceEndpoints[T] {
    return this.serviceEndpoints[service];
  }

  /**
   * Get all services with their configurations
   */
  getAllServices(): Array<{
    name: keyof ServiceEndpoints;
    config: ServiceEndpoints[keyof ServiceEndpoints];
    protocols: string[];
  }> {
    return Object.entries(this.serviceEndpoints).map(([name, config]) => ({
      name: name as keyof ServiceEndpoints,
      config,
      protocols: this.getAvailableProtocols(config)
    }));
  }

  /**
   * Get available protocols for a service
   */
  private getAvailableProtocols(config: ProtocolEndpoint | DatabaseEndpoint | MessagingEndpoint | FrontendEndpoint): string[] {
    const protocols: string[] = [];
    
    if ('http' in config && config.http) protocols.push('HTTP');
    if ('grpc' in config && config.grpc) protocols.push('gRPC');
    if ('quic' in config && config.quic) protocols.push('QUIC');
    if ('websocket' in config && config.websocket) protocols.push('WebSocket');
    if ('primary' in config && config.primary) protocols.push('HTTP');
    if ('server' in config && config.server) protocols.push('NATS');
    
    return protocols;
  }
}

// Export singleton instance
export const apiOrchestrator = APIOrchestrator.getInstance();