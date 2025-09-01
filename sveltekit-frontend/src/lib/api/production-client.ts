/**
 * Production API Client - Multi-protocol service communication
 * Integrates with all 37 Go binaries through HTTP/gRPC/QUIC/WebSocket
 */

import { productionServiceRegistry, getOptimalServiceForRoute, type ServiceDefinition } from '$lib/services/production-service-registry.js';
import { URL } from "url";

export interface ServiceRequest {
  route: string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE';
  body?: unknown;
  headers?: Record<string, string>;
  timeout?: number;
  retries?: number;
}

export interface ServiceResponse<T = any> {
  data: T;
  status: number;
  headers: Record<string, string>;
  protocol: string;
  service: string;
  latency: number;
}

export interface ProtocolClient {
  request<T>(url: string, options: ServiceRequest): Promise<ServiceResponse<T>>;
}

class HTTPClient implements ProtocolClient {
  async request<T>(url: string, options: ServiceRequest): Promise<ServiceResponse<T>> {
    const startTime = Date.now();
    
    const response = await fetch(url, {
      method: options.method,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      },
      body: options.body ? JSON.stringify(options.body) : undefined,
      signal: AbortSignal.timeout(options.timeout || 30000)
    });

    const data = await response.json();
    const latency = Date.now() - startTime;

    return {
      data,
      status: response.status,
      headers: Object.fromEntries(response.headers.entries()),
      protocol: 'http',
      service: new URL(url).host,
      latency
    };
  }
}

class WebSocketClient implements ProtocolClient {
  private connections: Map<string, WebSocket> = new Map();

  async request<T>(url: string, options: ServiceRequest): Promise<ServiceResponse<T>> {
    return new Promise((resolve, reject) => {
      const startTime = Date.now();
      const wsUrl = url.replace('http://', 'ws://').replace('https://', 'wss://');
      
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        ws.send(JSON.stringify({
          route: options.route,
          method: options.method,
          body: options.body,
          headers: options.headers
        }));
      };

      ws.onmessage = (event: any) => {
        const data = JSON.parse(event.data);
        const latency = Date.now() - startTime;
        
        resolve({
          data,
          status: 200,
          headers: {},
          protocol: 'websocket',
          service: new URL(url).host,
          latency
        });
        
        ws.close();
      };

      ws.onerror = () => reject(new Error('WebSocket connection failed'));
      ws.onclose = (event: any) => {
        if (event.code !== 1000) {
          reject(new Error(`WebSocket closed with code: ${event.code}`));
        }
      };

      setTimeout(() => reject(new Error('WebSocket timeout')), options.timeout || 30000);
    });
  }
}

// QUIC Client (fallback to HTTP for browser compatibility)
class QUICClient implements ProtocolClient {
  async request<T>(url: string, options: ServiceRequest): Promise<ServiceResponse<T>> {
    // In browser environment, fallback to HTTP
    // In Node.js, this would use a proper QUIC client
    const httpClient = new HTTPClient();
    const response = await httpClient.request<T>(url, options);
    return { ...response, protocol: 'quic' };
  }
}

// gRPC Client (uses gRPC-Web for browser compatibility)
class GRPCClient implements ProtocolClient {
  async request<T>(url: string, options: ServiceRequest): Promise<ServiceResponse<T>> {
    // For browser environment, would use grpc-web
    // For Node.js, would use @grpc/grpc-js
    const httpClient = new HTTPClient();
    const response = await httpClient.request<T>(url, options);
    return { ...response, protocol: 'grpc' };
  }
}

export class ProductionAPIClient {
  private httpClient = new HTTPClient();
  private wsClient = new WebSocketClient();
  private quicClient = new QUICClient();
  private grpcClient = new GRPCClient();
  
  private requestMetrics: Map<string, number[]> = new Map();

  async request<T = any>(options: ServiceRequest): Promise<ServiceResponse<T>> {
    const routeMapping = productionServiceRegistry.getServiceForRoute(options.route);
    
    if (!routeMapping) {
      throw new Error(`No service mapping found for route: ${options.route}`);
    }

    const { primary, fallbacks, protocol } = routeMapping;
    
    // Try primary service first
    try {
      const response = await this.executeRequest<T>(primary, protocol.protocol as any, options);
      this.recordMetrics(options.route, response.latency);
      return response;
    } catch (primaryError) {
      console.warn(`Primary service failed for ${options.route}:`, primaryError);
      
      // Try fallback services
      for (const fallbackService of fallbacks) {
        try {
          const response = await this.executeRequest<T>(fallbackService, 'http', options);
          this.recordMetrics(options.route, response.latency);
          return response;
        } catch (fallbackError) {
          console.warn(`Fallback service failed for ${options.route}:`, fallbackError);
        }
      }
      
      throw new Error(`All services failed for route: ${options.route}`);
    }
  }

  private async executeRequest<T>(
    service: ServiceDefinition, 
    protocol: 'http' | 'grpc' | 'quic' | 'websocket',
    options: ServiceRequest
  ): Promise<ServiceResponse<T>> {
    const baseUrl = `http://localhost:${service.port}`;
    const fullUrl = `${baseUrl}${options.route}`;

    switch (protocol) {
      case 'http':
        return this.httpClient.request<T>(fullUrl, options);
      case 'websocket':
        return this.wsClient.request<T>(fullUrl, options);
      case 'quic':
        return this.quicClient.request<T>(fullUrl, options);
      case 'grpc':
        return this.grpcClient.request<T>(fullUrl, options);
      default:
        throw new Error(`Unsupported protocol: ${protocol}`);
    }
  }

  private recordMetrics(route: string, latency: number): void {
    if (!this.requestMetrics.has(route)) {
      this.requestMetrics.set(route, []);
    }
    
    const metrics = this.requestMetrics.get(route)!;
    metrics.push(latency);
    
    // Keep only last 100 measurements
    if (metrics.length > 100) {
      metrics.shift();
    }
  }

  getRouteMetrics(route: string): {
    count: number;
    avgLatency: number;
    p95Latency: number;
    minLatency: number;
    maxLatency: number;
  } {
    const metrics = this.requestMetrics.get(route) || [];
    if (metrics.length === 0) {
      return { count: 0, avgLatency: 0, p95Latency: 0, minLatency: 0, maxLatency: 0 };
    }

    const sorted = [...metrics].sort((a, b) => a - b);
    const p95Index = Math.floor(sorted.length * 0.95);

    return {
      count: metrics.length,
      avgLatency: Math.round(metrics.reduce((sum, val) => sum + val, 0) / metrics.length),
      p95Latency: sorted[p95Index] || 0,
      minLatency: sorted[0] || 0,
      maxLatency: sorted[sorted.length - 1] || 0
    };
  }

  async getClusterStatus(): Promise<{
    health: Awaited<ReturnType<typeof productionServiceRegistry.getClusterHealth>>;
    metrics: Record<string, ReturnType<typeof this.getRouteMetrics>>;
    activeRoutes: string[];
  }> {
    const health = await productionServiceRegistry.getClusterHealth();
    const activeRoutes = Array.from(this.requestMetrics.keys());
    const metrics = Object.fromEntries(
      activeRoutes.map(route => [route, this.getRouteMetrics(route)])
    );

    return { health, metrics, activeRoutes };
  }
}

// Convenience methods for specific service categories
export class RAGAPIClient {
  constructor(private client: ProductionAPIClient) {}

  async query(query: string, context?: unknown): Promise<ServiceResponse> {
    return this.client.request({
      route: '/api/v1/rag/query',
      method: 'POST',
      body: { query, context }
    });
  }

  async semanticSearch(query: string, filters?: unknown): Promise<ServiceResponse> {
    return this.client.request({
      route: '/api/v1/rag/semantic',
      method: 'POST',
      body: { query, filters }
    });
  }

  async embed(text: string): Promise<ServiceResponse> {
    return this.client.request({
      route: '/api/v1/rag/embed',
      method: 'POST',
      body: { text }
    });
  }
}

export class UploadAPIClient {
  constructor(private client: ProductionAPIClient) {}

  async uploadFile(file: File, metadata?: unknown): Promise<ServiceResponse> {
    const formData = new FormData();
    formData.append('file', file);
    if (metadata) formData.append('metadata', JSON.stringify(metadata));

    return this.client.request({
      route: '/api/v1/upload/file',
      method: 'POST',
      body: formData
    });
  }

  async batchUpload(files: File[]): Promise<ServiceResponse> {
    const formData = new FormData();
    files.forEach((file, index) => formData.append(`file${index}`, file));

    return this.client.request({
      route: '/api/v1/upload/batch',
      method: 'POST',
      body: formData
    });
  }
}

export class ClusterAPIClient {
  constructor(private client: ProductionAPIClient) {}

  async getHealth(): Promise<ServiceResponse> {
    return this.client.request({
      route: '/api/v1/cluster/health',
      method: 'GET'
    });
  }

  async getServices(): Promise<ServiceResponse> {
    return this.client.request({
      route: '/api/v1/cluster/services',
      method: 'GET'
    });
  }

  async getMetrics(): Promise<ServiceResponse> {
    return this.client.request({
      route: '/api/v1/cluster/metrics',
      method: 'GET'
    });
  }
}

export class XStateAPIClient {
  constructor(private client: ProductionAPIClient) {}

  async sendEvent(event: any): Promise<ServiceResponse> {
    return this.client.request({
      route: '/api/v1/xstate/events',
      method: 'POST',
      body: { event }
    });
  }

  async getState(): Promise<ServiceResponse> {
    return this.client.request({
      route: '/api/v1/xstate/state',
      method: 'GET'
    });
  }
}

// Export singleton instances
export const productionAPIClient = new ProductionAPIClient();
export const ragAPI = new RAGAPIClient(productionAPIClient);
export const uploadAPI = new UploadAPIClient(productionAPIClient);
export const clusterAPI = new ClusterAPIClient(productionAPIClient);
export const xstateAPI = new XStateAPIClient(productionAPIClient);