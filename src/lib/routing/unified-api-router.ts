/**
 * Unified API Router for Legal AI Platform
 * Consolidates all API endpoints with binary encoding support
 */

import { binaryEncoder, type EncodingFormat } from '$lib/middleware/binary-encoding';
import type { RequestEvent } from '@sveltejs/kit';

export interface RouteConfig {
  path: string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
  service: string;
  port: number;
  encoding?: EncodingFormat[];
  auth?: boolean;
  rateLimit?: number;
  cache?: boolean;
  timeout?: number;
}

export interface ProxyResponse {
  status: number;
  data: any;
  encoding: EncodingFormat;
  metrics?: any;
}

export class UnifiedAPIRouter {
  private routes: Map<string, RouteConfig> = new Map();
  private serviceHealth: Map<string, boolean> = new Map();

  constructor() {
    this.initializeRoutes();
    this.startHealthMonitoring();
  }

  /**
   * Initialize all API routes
   */
  private initializeRoutes(): void {
    const routes: RouteConfig[] = [
      // Enhanced RAG Service
      {
        path: '/api/ai/chat',
        method: 'POST',
        service: 'enhanced-rag',
        port: 8094,
        encoding: ['cbor', 'msgpack', 'json'],
        auth: true,
        rateLimit: 60,
        timeout: 30000
      },
      {
        path: '/api/ai/analyze',
        method: 'POST',
        service: 'enhanced-rag',
        port: 8094,
        encoding: ['cbor', 'msgpack', 'json'],
        auth: true,
        timeout: 60000
      },
      {
        path: '/api/ai/embeddings',
        method: 'POST',
        service: 'enhanced-rag',
        port: 8094,
        encoding: ['cbor', 'msgpack', 'json'],
        auth: true,
        cache: true
      },
      {
        path: '/api/vector/search',
        method: 'POST',
        service: 'enhanced-rag',
        port: 8094,
        encoding: ['cbor', 'msgpack', 'json'],
        auth: true,
        cache: true
      },

      // Upload Service
      {
        path: '/api/upload',
        method: 'POST',
        service: 'upload',
        port: 8093,
        encoding: ['cbor', 'json'],
        auth: true,
        timeout: 120000
      },
      {
        path: '/api/documents',
        method: 'GET',
        service: 'upload',
        port: 8093,
        encoding: ['cbor', 'msgpack', 'json'],
        auth: true,
        cache: true
      },
      {
        path: '/api/documents/:id',
        method: 'GET',
        service: 'upload',
        port: 8093,
        encoding: ['cbor', 'msgpack', 'json'],
        auth: true,
        cache: true
      },

      // GPU & CUDA Services
      {
        path: '/api/gpu/status',
        method: 'GET',
        service: 'gpu-monitor',
        port: 8095,
        encoding: ['json'],
        auth: false,
        cache: false
      },
      {
        path: '/api/gpu/test-array',
        method: 'POST',
        service: 'cuda-worker',
        port: 8096,
        encoding: ['cbor', 'msgpack', 'json'],
        auth: true,
        timeout: 30000
      },
      {
        path: '/api/som/train',
        method: 'POST',
        service: 'cuda-worker',
        port: 8096,
        encoding: ['cbor', 'msgpack'],
        auth: true,
        timeout: 120000
      },

      // Load Balancer & Orchestration
      {
        path: '/api/orchestration/restart-all',
        method: 'POST',
        service: 'load-balancer',
        port: 8099,
        encoding: ['json'],
        auth: true
      },
      {
        path: '/api/orchestration/stop-all',
        method: 'POST',
        service: 'load-balancer',
        port: 8099,
        encoding: ['json'],
        auth: true
      },
      {
        path: '/api/health/all',
        method: 'GET',
        service: 'load-balancer',
        port: 8099,
        encoding: ['cbor', 'msgpack', 'json'],
        auth: false,
        cache: false
      },

      // Database Services
      {
        path: '/api/database/health',
        method: 'GET',
        service: 'database-monitor',
        port: 8097,
        encoding: ['json'],
        auth: false,
        cache: false
      },
      {
        path: '/api/database/vector-stats',
        method: 'GET',
        service: 'database-monitor',
        port: 8097,
        encoding: ['cbor', 'msgpack', 'json'],
        auth: true,
        cache: true
      },
      {
        path: '/api/cache/stats',
        method: 'GET',
        service: 'redis-monitor',
        port: 8098,
        encoding: ['json'],
        auth: false,
        cache: true
      },

      // Binary Encoding Control
      {
        path: '/api/encoding/format',
        method: 'PUT',
        service: 'encoding-controller',
        port: 8091,
        encoding: ['json'],
        auth: true
      },
      {
        path: '/api/encoding/metrics',
        method: 'GET',
        service: 'encoding-controller',
        port: 8091,
        encoding: ['cbor', 'msgpack', 'json'],
        auth: true,
        cache: true
      },

      // Logging & Monitoring
      {
        path: '/api/logs/stream',
        method: 'GET',
        service: 'log-aggregator',
        port: 8092,
        encoding: ['json'],
        auth: true,
        cache: false
      },
      {
        path: '/api/metrics/dashboard',
        method: 'GET',
        service: 'metrics-collector',
        port: 8090,
        encoding: ['cbor', 'msgpack', 'json'],
        auth: true,
        cache: true
      },
      {
        path: '/api/tracing/toggle',
        method: 'POST',
        service: 'tracing-controller',
        port: 8089,
        encoding: ['json'],
        auth: true
      }
    ];

    routes.forEach(route => {
      const key = `${route.method}:${route.path}`;
      this.routes.set(key, route);
    });
  }

  /**
   * Route request to appropriate service
   */
  async routeRequest(event: RequestEvent): Promise<Response> {
    const { request, url } = event;
    const method = request.method as RouteConfig['method'];
    const path = url.pathname;
    
    // Find matching route
    const route = this.findRoute(method, path);
    if (!route) {
      return new Response(JSON.stringify({ error: 'Route not found' }), {
        status: 404,
        headers: { 'content-type': 'application/json' }
      });
    }

    // Check service health
    if (!this.serviceHealth.get(route.service)) {
      return new Response(JSON.stringify({ error: 'Service unavailable' }), {
        status: 503,
        headers: { 'content-type': 'application/json' }
      });
    }

    // Handle authentication
    if (route.auth && !this.isAuthenticated(request)) {
      return new Response(JSON.stringify({ error: 'Unauthorized' }), {
        status: 401,
        headers: { 'content-type': 'application/json' }
      });
    }

    // Determine encoding format
    const acceptHeader = request.headers.get('accept') || '';
    const preferredEncoding = this.determineEncoding(acceptHeader, route.encoding);

    try {
      // Proxy request to service
      const proxyResponse = await this.proxyToService(request, route, path);
      
      // Encode response if needed
      if (preferredEncoding !== 'json') {
        const { encoded, metrics } = await binaryEncoder.encode(proxyResponse.data, preferredEncoding);
        
        const contentType = preferredEncoding === 'cbor' ? 'application/cbor' : 'application/msgpack';
        
        return new Response(encoded as ArrayBuffer, {
          status: proxyResponse.status,
          headers: {
            'content-type': contentType,
            'x-encoding-metrics': JSON.stringify(metrics)
          }
        });
      }

      return new Response(JSON.stringify(proxyResponse.data), {
        status: proxyResponse.status,
        headers: { 'content-type': 'application/json' }
      });

    } catch (error) {
      console.error(`Routing error for ${method} ${path}:`, error);
      
      return new Response(JSON.stringify({ 
        error: 'Internal server error',
        service: route.service,
        message: error instanceof Error ? error.message : 'Unknown error'
      }), {
        status: 500,
        headers: { 'content-type': 'application/json' }
      });
    }
  }

  /**
   * Find matching route
   */
  private findRoute(method: RouteConfig['method'], path: string): RouteConfig | null {
    // Exact match first
    const exactKey = `${method}:${path}`;
    if (this.routes.has(exactKey)) {
      return this.routes.get(exactKey)!;
    }

    // Pattern matching for parameterized routes
    for (const [key, route] of this.routes) {
      const [routeMethod, routePath] = key.split(':');
      if (routeMethod === method && this.pathMatches(routePath, path)) {
        return route;
      }
    }

    return null;
  }

  /**
   * Check if path matches route pattern
   */
  private pathMatches(pattern: string, path: string): boolean {
    const patternParts = pattern.split('/');
    const pathParts = path.split('/');
    
    if (patternParts.length !== pathParts.length) {
      return false;
    }

    return patternParts.every((part, index) => {
      return part.startsWith(':') || part === pathParts[index];
    });
  }

  /**
   * Determine best encoding format
   */
  private determineEncoding(acceptHeader: string, supportedFormats?: EncodingFormat[]): EncodingFormat {
    if (!supportedFormats) return 'json';

    if (acceptHeader.includes('application/cbor') && supportedFormats.includes('cbor')) {
      return 'cbor';
    }
    if (acceptHeader.includes('application/msgpack') && supportedFormats.includes('msgpack')) {
      return 'msgpack';
    }
    return 'json';
  }

  /**
   * Check authentication
   */
  private isAuthenticated(request: Request): boolean {
    const authHeader = request.headers.get('authorization');
    // Implement your authentication logic here
    return authHeader !== null; // Simplified for demo
  }

  /**
   * Proxy request to service
   */
  private async proxyToService(request: Request, route: RouteConfig, originalPath: string): Promise<ProxyResponse> {
    const serviceUrl = `http://localhost:${route.port}${originalPath}`;
    
    // Clone request with appropriate headers
    const headers = new Headers(request.headers);
    headers.set('x-forwarded-for', 'api-router');
    headers.set('x-service', route.service);

    const proxyRequest = new Request(serviceUrl, {
      method: request.method,
      headers,
      body: request.body
    });

    // Set timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), route.timeout || 30000);

    try {
      const response = await fetch(proxyRequest, {
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      const data = await response.json();
      
      return {
        status: response.status,
        data,
        encoding: 'json'
      };
      
    } catch (error) {
      clearTimeout(timeoutId);
      throw error;
    }
  }

  /**
   * Start health monitoring for services
   */
  private startHealthMonitoring(): void {
    // Set initial health status
    const services = ['enhanced-rag', 'upload', 'load-balancer'];
    services.forEach(service => {
      this.serviceHealth.set(service, true);
    });

    // Periodic health checks
    setInterval(async () => {
      await this.checkServiceHealth();
    }, 30000); // Check every 30 seconds
  }

  /**
   * Check health of all services
   */
  private async checkServiceHealth(): Promise<void> {
    const healthChecks = [
      { service: 'enhanced-rag', port: 8094, endpoint: '/health' },
      { service: 'upload', port: 8093, endpoint: '/health' },
      { service: 'load-balancer', port: 8099, endpoint: '/health' }
    ];

    await Promise.all(
      healthChecks.map(async ({ service, port, endpoint }) => {
        try {
          const response = await fetch(`http://localhost:${port}${endpoint}`, {
            signal: AbortSignal.timeout(5000)
          });
          
          this.serviceHealth.set(service, response.ok);
        } catch (error) {
          this.serviceHealth.set(service, false);
          console.warn(`Health check failed for ${service}:`, error);
        }
      })
    );
  }

  /**
   * Get service health status
   */
  getServiceHealth(): Map<string, boolean> {
    return new Map(this.serviceHealth);
  }

  /**
   * Get all routes
   */
  getRoutes(): RouteConfig[] {
    return Array.from(this.routes.values());
  }

  /**
   * Add route
   */
  addRoute(path: string, method: RouteConfig['method'], config: Omit<RouteConfig, 'path' | 'method'>): void {
    const key = `${method}:${path}`;
    this.routes.set(key, { path, method, ...config });
  }

  /**
   * Remove route
   */
  removeRoute(path: string, method: RouteConfig['method']): void {
    const key = `${method}:${path}`;
    this.routes.delete(key);
  }
}

// Global router instance
export const unifiedRouter = new UnifiedAPIRouter();