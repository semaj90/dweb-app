/**
 * Production Service Client - Multi-Protocol Native Windows
 * Handles HTTP, gRPC, QUIC, and WebSocket protocols with automatic failover
 * Uses existing compiled Go binaries for maximum performance
 */

import { 
  SERVICES_CONFIG, 
  API_ROUTES, 
  PROTOCOL_PRIORITY, 
  PROTOCOL_CONFIG,
  type ServiceConfig,
  type ProtocolRoute
} from '../config/multi-protocol-routes.js';

export type ProtocolType = 'http' | 'grpc' | 'quic' | 'ws';
export type ServiceResponse<T = any> = {
  success: boolean;
  data?: T;
  error?: string;
  protocol: ProtocolType;
  latency: number;
  service: string;
};

export interface ServiceHealth {
  service: string;
  status: 'healthy' | 'unhealthy' | 'unknown';
  protocols: Record<ProtocolType, boolean>;
  lastCheck: Date;
  latency: number;
  errorCount: number;
}

class ProductionServiceClient {
  private healthStatus: Map<string, ServiceHealth> = new Map();
  private circuitBreakers: Map<string, { failures: number; lastFailure: Date; state: 'closed' | 'open' | 'half-open' }> = new Map();
  private protocolClients: Map<ProtocolType, any> = new Map();

  constructor() {
    this.initializeProtocolClients();
    this.startHealthChecking();
  }

  private initializeProtocolClients() {
    // HTTP Client (using fetch)
    this.protocolClients.set('http', {
      request: async (url: string, options: any) => {
        const response = await fetch(url, {
          ...options,
          headers: {
            ...PROTOCOL_CONFIG.http.headers,
            ...options.headers
          }
        });
        return response;
      }
    });

    // WebSocket Client
    this.protocolClients.set('ws', {
      connect: (url: string) => {
        return new WebSocket(url);
      }
    });

    // QUIC Client (HTTP/3 fallback)
    this.protocolClients.set('quic', {
      request: async (url: string, options: any) => {
        // For now, use HTTP/3 over fetch if available, fallback to HTTP/2
        const response = await fetch(url.replace('quic://', 'https://'), {
          ...options,
          headers: {
            ...PROTOCOL_CONFIG.quic,
            ...options.headers
          }
        });
        return response;
      }
    });

    // gRPC-Web Client (for browser compatibility)
    this.protocolClients.set('grpc', {
      request: async (url: string, options: any) => {
        // Convert gRPC call to HTTP POST for browser compatibility
        const httpUrl = url.replace('grpc://', 'http://') + '/invoke';
        const response = await fetch(httpUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/grpc-web+proto',
            ...options.headers
          },
          body: options.body
        });
        return response;
      }
    });
  }

  /**
   * Main service call method with automatic protocol selection and failover
   */
  async callService<T = any>(
    endpoint: string, 
    data?: unknown, 
    options: {
      preferredProtocol?: ProtocolType;
      timeout?: number;
      retries?: number;
      priority?: 'performance' | 'reliability' | 'realtime';
    } = {}
  ): Promise<ServiceResponse<T>> {
    const route = API_ROUTES.find(r => r.endpoint === endpoint);
    if (!route) {
      throw new Error(`No route configuration found for endpoint: ${endpoint}`);
    }

    const priority = options.priority || 'performance';
    const protocols = this.getProtocolOrder(route, priority, options.preferredProtocol);
    
    let lastError: Error | null = null;
    
    for (const protocol of protocols) {
      try {
        // Check circuit breaker
        if (this.isCircuitOpen(route.service, protocol)) {
          continue;
        }

        const startTime = performance.now();
        const result = await this.executeProtocolCall(
          protocol, 
          route, 
          data, 
          options
        );
        const latency = performance.now() - startTime;

        // Reset circuit breaker on success
        this.resetCircuitBreaker(route.service, protocol);
        
        // Update health status
        this.updateHealthStatus(route.service, protocol, true, latency);

        return {
          success: true,
          data: result,
          protocol,
          latency,
          service: route.service
        };
      } catch (error) {
        lastError = error as Error;
        console.warn(`Protocol ${protocol} failed for ${endpoint}:`, error);
        
        // Update circuit breaker
        this.updateCircuitBreaker(route.service, protocol);
        
        // Update health status
        this.updateHealthStatus(route.service, protocol, false, 0);
      }
    }

    return {
      success: false,
      error: lastError?.message || 'All protocols failed',
      protocol: protocols[0],
      latency: 0,
      service: route.service
    };
  }

  private async executeProtocolCall(
    protocol: ProtocolType,
    route: ProtocolRoute,
    data: any,
    options: any
  ): Promise<any> {
    const url = route.protocols[protocol];
    if (!url) {
      throw new Error(`Protocol ${protocol} not available for ${route.endpoint}`);
    }

    const timeout = options.timeout || route.timeout;
    const client = this.protocolClients.get(protocol);

    switch (protocol) {
      case 'http':
      case 'quic':
        const response = await Promise.race([
          client.request(url, {
            method: data ? 'POST' : 'GET',
            body: data ? JSON.stringify(data) : undefined,
            headers: {
              'Content-Type': 'application/json'
            }
          }),
          new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Timeout')), timeout)
          )
        ]) as Response;

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const contentType = response.headers.get('content-type');
        if (contentType?.includes('application/json')) {
          return await response.json();
        }
        return await response.text();

      case 'grpc':
        // gRPC-Web implementation
        return await client.request(url, {
          body: data,
          timeout
        });

      case 'ws':
        // WebSocket implementation
        return new Promise((resolve, reject) => {
          const ws = client.connect(url);
          const timer = setTimeout(() => reject(new Error('WebSocket timeout')), timeout);
          
          ws.onopen = () => {
            if (data) {
              ws.send(JSON.stringify(data));
            }
          };
          
          ws.onmessage = (event) => {
            clearTimeout(timer);
            ws.close();
            resolve(JSON.parse(event.data));
          };
          
          ws.onerror = (error) => {
            clearTimeout(timer);
            reject(error);
          };
        });

      default:
        throw new Error(`Unsupported protocol: ${protocol}`);
    }
  }

  private getProtocolOrder(
    route: ProtocolRoute, 
    priority: 'performance' | 'reliability' | 'realtime',
    preferred?: ProtocolType
  ): ProtocolType[] {
    const availableProtocols = Object.keys(route.protocols) as ProtocolType[];
    const priorityOrder = PROTOCOL_PRIORITY[priority];
    
    // If preferred protocol is available, try it first
    if (preferred && availableProtocols.includes(preferred)) {
      return [preferred, ...availableProtocols.filter(p => p !== preferred)];
    }
    
    // Sort by priority order
    return priorityOrder.filter(p => availableProtocols.includes(p));
  }

  // Circuit Breaker Implementation
  private isCircuitOpen(service: string, protocol: ProtocolType): boolean {
    const key = `${service}-${protocol}`;
    const breaker = this.circuitBreakers.get(key);
    
    if (!breaker) return false;
    
    if (breaker.state === 'open') {
      // Check if we should try half-open
      const timeSinceLastFailure = Date.now() - breaker.lastFailure.getTime();
      if (timeSinceLastFailure > 60000) { // 1 minute
        breaker.state = 'half-open';
        return false;
      }
      return true;
    }
    
    return false;
  }

  private updateCircuitBreaker(service: string, protocol: ProtocolType): void {
    const key = `${service}-${protocol}`;
    const breaker = this.circuitBreakers.get(key) || { 
      failures: 0, 
      lastFailure: new Date(), 
      state: 'closed' as const 
    };
    
    breaker.failures++;
    breaker.lastFailure = new Date();
    
    if (breaker.failures >= 5) {
      breaker.state = 'open';
    }
    
    this.circuitBreakers.set(key, breaker);
  }

  private resetCircuitBreaker(service: string, protocol: ProtocolType): void {
    const key = `${service}-${protocol}`;
    const breaker = this.circuitBreakers.get(key);
    
    if (breaker) {
      breaker.failures = 0;
      breaker.state = 'closed';
      this.circuitBreakers.set(key, breaker);
    }
  }

  // Health Monitoring
  private updateHealthStatus(
    service: string, 
    protocol: ProtocolType, 
    success: boolean, 
    latency: number
  ): void {
    const health = this.healthStatus.get(service) || {
      service,
      status: 'unknown' as const,
      protocols: {} as Record<ProtocolType, boolean>,
      lastCheck: new Date(),
      latency: 0,
      errorCount: 0
    };

    health.protocols[protocol] = success;
    health.lastCheck = new Date();
    health.latency = success ? latency : health.latency;
    
    if (!success) {
      health.errorCount++;
    } else {
      health.errorCount = Math.max(0, health.errorCount - 1);
    }

    // Determine overall health
    const protocolStatuses = Object.values(health.protocols);
    const healthyCount = protocolStatuses.filter(Boolean).length;
    
    if (healthyCount === 0) {
      health.status = 'unhealthy';
    } else if (healthyCount === protocolStatuses.length) {
      health.status = 'healthy';
    } else {
      health.status = 'unhealthy'; // Conservative approach
    }

    this.healthStatus.set(service, health);
  }

  private async startHealthChecking(): Promise<void> {
    setInterval(async () => {
      for (const [serviceName, config] of Object.entries(SERVICES_CONFIG)) {
        const serviceConfig = config as ServiceConfig;
        if (serviceConfig.healthEndpoint) {
          try {
            // Create AbortController for timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000);
            
            const response = await fetch(`http://localhost:${serviceConfig.port}${serviceConfig.healthEndpoint}`, {
              signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            this.updateHealthStatus(serviceName, 'http', response.ok, 0);
          } catch (error) {
            this.updateHealthStatus(serviceName, 'http', false, 0);
          }
        }
      }
    }, 30000); // Check every 30 seconds
  }

  // Public API Methods
  async getServiceHealth(service?: string): Promise<ServiceHealth[]> {
    if (service) {
      const health = this.healthStatus.get(service);
      return health ? [health] : [];
    }
    return Array.from(this.healthStatus.values());
  }

  async getClusterStatus(): Promise<{
    totalServices: number;
    healthyServices: number;
    unhealthyServices: number;
    protocols: Record<ProtocolType, number>;
  }> {
    const allHealth = Array.from(this.healthStatus.values());
    const protocolCounts = { http: 0, grpc: 0, quic: 0, ws: 0 };
    
    for (const health of allHealth) {
      for (const [protocol, isHealthy] of Object.entries(health.protocols)) {
        if (isHealthy) {
          protocolCounts[protocol as ProtocolType]++;
        }
      }
    }

    return {
      totalServices: allHealth.length,
      healthyServices: allHealth.filter(h => h.status === 'healthy').length,
      unhealthyServices: allHealth.filter(h => h.status === 'unhealthy').length,
      protocols: protocolCounts
    };
  }

  // Convenience methods for common operations
  async uploadDocument(file: File, metadata?: unknown): Promise<ServiceResponse> {
    const formData = new FormData();
    formData.append('file', file);
    if (metadata) {
      formData.append('metadata', JSON.stringify(metadata));
    }

    return this.callService('/api/v1/upload', formData, {
      preferredProtocol: 'http',
      timeout: 60000
    });
  }

  async queryRAG(query: string, context?: unknown): Promise<ServiceResponse> {
    return this.callService('/api/v1/rag', {
      query,
      context
    }, {
      priority: 'performance',
      timeout: 30000
    });
  }

  async getLegalAnalysis(document: string, options?: unknown): Promise<ServiceResponse> {
    return this.callService('/api/v1/legal', {
      document,
      options
    }, {
      preferredProtocol: 'grpc',
      timeout: 45000
    });
  }

  async getRecommendations(userId: string, context?: unknown): Promise<ServiceResponse> {
    return this.callService('/api/v1/recommendations', {
      userId,
      context
    }, {
      timeout: 15000
    });
  }

  async subscribeToStateChanges(callback: (event: any) => void): Promise<WebSocket> {
    const url = 'ws://localhost:8212/api/v1/state/events';
    const ws = new WebSocket(url);
    
    ws.onmessage = (event) => {
      callback(JSON.parse(event.data));
    };
    
    return ws;
  }
}

// Singleton instance
export const productionServiceClient = new ProductionServiceClient();

export default productionServiceClient;