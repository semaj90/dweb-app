// Enhanced REST Architecture Types and Interfaces
// Provides type definitions for clustering and document processing

export interface KMeansConfig {
  k: number;
  maxIterations: number;
  tolerance: number;
  distanceMetric: 'euclidean' | 'cosine' | 'manhattan';
  initialization: 'random' | 'kmeans++';
}

export interface KMeansClusterer {
  fit(data: number[][]): Promise<void>;
  predict(data: number[][]): Promise<number[]>;
  getCentroids(): number[][];
  silhouetteScore(): Promise<number>;
}

export interface ClusterResult {
  clusterId: number;
  centroid: number[];
  members: any[];
  size: number;
  inertia: number;
}

export interface DocumentCluster {
  id: string;
  documents: any[];
  centroid: number[];
  keywords: string[];
  summary: string;
}

export interface SOMConfig {
  width: number;
  height: number;
  dimensions: number;
  learningRate: number;
  radius: number;
  iterations: number;
}

export interface SelfOrganizingMap {
  train(data: number[][]): Promise<void>;
  map(data: number[]): Promise<[number, number]>;
  getWeights(): number[][][];
  quantizationError(): number;
}

export interface RESTEndpoint {
  path: string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
  handler: (req: any, res: any) => Promise<void>;
  middleware?: any[];
  rateLimit?: {
    windowMs: number;
    max: number;
  };
  cache?: {
    ttl: number;
    key: (req: any) => string;
  };
}

export interface APIResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  metadata?: {
    timestamp: string;
    version: string;
    requestId: string;
  };
}

export interface PaginatedResponse<T = any> extends APIResponse<T[]> {
  pagination: {
    page: number;
    pageSize: number;
    totalPages: number;
    totalItems: number;
    hasNext: boolean;
    hasPrevious: boolean;
  };
}

export interface SearchRequest {
  query: string;
  filters?: Record<string, any>;
  sort?: {
    field: string;
    order: 'asc' | 'desc';
  };
  pagination?: {
    page: number;
    pageSize: number;
  };
  includeMetadata?: boolean;
}

export interface SearchResponse<T = any> extends PaginatedResponse<T> {
  facets?: Record<string, Array<{
    value: string;
    count: number;
  }>>;
  suggestions?: string[];
  totalTime?: number;
}

export interface BatchRequest<T = any> {
  operations: Array<{
    id: string;
    method: string;
    params: T;
  }>;
  parallel?: boolean;
  stopOnError?: boolean;
}

export interface BatchResponse<T = any> {
  results: Array<{
    id: string;
    success: boolean;
    data?: T;
    error?: string;
  }>;
  summary: {
    total: number;
    successful: number;
    failed: number;
    duration: number;
  };
}

export interface WebSocketMessage {
  type: 'subscribe' | 'unsubscribe' | 'message' | 'error' | 'ping' | 'pong';
  channel?: string;
  data?: any;
  timestamp: number;
}

export interface StreamResponse {
  stream: ReadableStream;
  contentType: string;
  contentLength?: number;
}

export interface CacheConfig {
  type: 'memory' | 'redis' | 'disk';
  ttl: number;
  maxSize?: number;
  invalidation?: {
    events: string[];
    patterns: string[];
  };
}

export interface RateLimitConfig {
  windowMs: number;
  max: number;
  message?: string;
  statusCode?: number;
  skipSuccessfulRequests?: boolean;
  skipFailedRequests?: boolean;
  keyGenerator?: (req: any) => string;
}

export interface SecurityConfig {
  cors: {
    origins: string[];
    methods: string[];
    credentials: boolean;
  };
  helmet: boolean;
  csrf: boolean;
  rateLimit: RateLimitConfig;
  authentication: {
    type: 'jwt' | 'oauth' | 'apikey' | 'basic';
    config: any;
  };
}

export interface MonitoringConfig {
  metrics: boolean;
  logging: {
    level: 'debug' | 'info' | 'warn' | 'error';
    format: 'json' | 'text';
  };
  tracing: boolean;
  healthCheck: {
    path: string;
    interval: number;
  };
}

export class RESTArchitecture {
  private endpoints: Map<string, RESTEndpoint>;
  private middleware: any[];
  private config: {
    security?: SecurityConfig;
    cache?: CacheConfig;
    monitoring?: MonitoringConfig;
  };

  constructor(config?: any) {
    this.endpoints = new Map();
    this.middleware = [];
    this.config = config || {};
  }

  registerEndpoint(endpoint: RESTEndpoint): void {
    const key = `${endpoint.method}:${endpoint.path}`;
    this.endpoints.set(key, endpoint);
  }

  use(middleware: any): void {
    this.middleware.push(middleware);
  }

  async handleRequest(method: string, path: string, req: any, res: any): Promise<void> {
    const key = `${method}:${path}`;
    const endpoint = this.endpoints.get(key);
    
    if (!endpoint) {
      res.status(404).json({
        success: false,
        error: 'Endpoint not found'
      });
      return;
    }

    try {
      // Apply middleware
      for (const mw of this.middleware) {
        await mw(req, res);
      }

      // Apply endpoint-specific middleware
      if (endpoint.middleware) {
        for (const mw of endpoint.middleware) {
          await mw(req, res);
        }
      }

      // Execute handler
      await endpoint.handler(req, res);
    } catch (error) {
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Internal server error'
      });
    }
  }

  getEndpoints(): RESTEndpoint[] {
    return Array.from(this.endpoints.values());
  }
}

// Export default instance
const restArchitecture = new RESTArchitecture();
export default restArchitecture;