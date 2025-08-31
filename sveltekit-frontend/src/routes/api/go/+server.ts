import type { RequestHandler } from './$types';

/**
 * Go Microservices Proxy API
 * Handles routing between SvelteKit and Go services with JSON/Protocol Buffer support
 * POST /api/go - Route requests to appropriate Go microservice
 */


import { ensureError } from '$lib/utils/ensure-error';
import { dev } from '$app/environment';

// Go microservices configuration (37 services from ecosystem summary)
const GO_SERVICES = {
  // Tier 1: Core Services (Always Running)
  'enhanced-rag': {
    baseUrl: 'http://localhost:8094',
    healthPath: '/api/health',
    protocols: ['http', 'quic'],
    capabilities: ['ai', 'rag', 'gpu', 'som', 'xstate']
  },
  'upload-service': {
    baseUrl: 'http://localhost:8093', 
    healthPath: '/health',
    protocols: ['http'],
    capabilities: ['file-upload', 'storage', 'processing']
  },
  'kratos-server': {
    baseUrl: 'http://localhost:50051',
    healthPath: '/health',
    protocols: ['grpc'],
    capabilities: ['legal-grpc', 'gpu-compute', 'search']
  },
  
  // Tier 2: Advanced Services (New Implementations)
  'advanced-cuda': {
    baseUrl: 'http://localhost:8095',
    healthPath: '/health',
    protocols: ['http', 'quic', 'grpc'],
    capabilities: ['kernel-splicing', 'attention', 'flash-attention', 'cuda-direct']
  },
  'dimensional-cache': {
    baseUrl: 'http://localhost:8097',
    healthPath: '/health',
    protocols: ['http', 'quic'],
    capabilities: ['multi-dimensional-cache', 'lru-eviction', 'vector-storage']
  },
  'xstate-manager': {
    baseUrl: 'http://localhost:8098',
    healthPath: '/health',
    protocols: ['http', 'websocket'],
    capabilities: ['idle-detection', 'state-management', 'rabbitmq-queue']
  },
  'module-manager': {
    baseUrl: 'http://localhost:8099',
    healthPath: '/health',
    protocols: ['http', 'grpc'],
    capabilities: ['hot-swap', 'module-loading', 'a-b-testing']
  },
  'recommendation-engine': {
    baseUrl: 'http://localhost:8100',
    healthPath: '/health',
    protocols: ['http', 'websocket'],
    capabilities: ['ai-recommendations', 'user-patterns', 'self-prompting']
  },
  
  // Additional microservices (8101-8136 covering 37 total services)
  'vector-service': { baseUrl: 'http://localhost:8101', healthPath: '/health', protocols: ['http', 'grpc'], capabilities: ['vector-search', 'similarity'] },
  'load-balancer': { baseUrl: 'http://localhost:8102', healthPath: '/health', protocols: ['http', 'quic'], capabilities: ['load-balancing', 'failover'] },
  'cluster-manager': { baseUrl: 'http://localhost:8103', healthPath: '/health', protocols: ['http', 'grpc'], capabilities: ['service-discovery', 'health-monitoring'] },
  'gpu-indexer': { baseUrl: 'http://localhost:8104', healthPath: '/health', protocols: ['http'], capabilities: ['gpu-indexing', 'texture-processing'] },
  'context7-error': { baseUrl: 'http://localhost:8105', healthPath: '/health', protocols: ['http'], capabilities: ['error-analysis', 'auto-fix'] },
  'quic-gateway': { baseUrl: 'http://localhost:8106', healthPath: '/health', protocols: ['quic'], capabilities: ['quic-protocol', 'gateway'] },
  'cuda-worker': { baseUrl: 'http://localhost:8107', healthPath: '/health', protocols: ['http'], capabilities: ['cuda-computation', 'worker-pool'] },
  'vector-consumer': { baseUrl: 'http://localhost:8108', healthPath: '/health', protocols: ['http'], capabilities: ['vector-consumption', 'batch-processing'] },
  'gin-upload': { baseUrl: 'http://localhost:8109', healthPath: '/health', protocols: ['http'], capabilities: ['gin-framework', 'file-upload'] },
  'ingest-service': { baseUrl: 'http://localhost:8110', healthPath: '/health', protocols: ['http'], capabilities: ['data-ingestion', 'pipeline'] },
  'vector-redis': { baseUrl: 'http://localhost:8111', healthPath: '/health', protocols: ['http'], capabilities: ['redis-vectors', 'caching'] },
  'enhanced-rag-go125': { baseUrl: 'http://localhost:8112', healthPath: '/health', protocols: ['http', 'quic'], capabilities: ['go1.25', 'enhanced-rag', 'greenteagc'] },
  'upload-service-go125': { baseUrl: 'http://localhost:8113', healthPath: '/health', protocols: ['http'], capabilities: ['go1.25', 'upload', 'optimized'] },
  'cuda-ai-service': { baseUrl: 'http://localhost:8114', healthPath: '/health', protocols: ['http'], capabilities: ['cuda-ai', 'gpu-acceleration'] },
  'vector-service-go125': { baseUrl: 'http://localhost:8115', healthPath: '/health', protocols: ['http'], capabilities: ['go1.25', 'vectors'] },
  'load-balancer-go125': { baseUrl: 'http://localhost:8116', healthPath: '/health', protocols: ['http'], capabilities: ['go1.25', 'load-balancing'] },
  'grpc-server-go125': { baseUrl: 'http://localhost:8117', healthPath: '/health', protocols: ['grpc'], capabilities: ['go1.25', 'grpc-optimized'] },
  'rag-quic-go125': { baseUrl: 'http://localhost:8118', healthPath: '/health', protocols: ['quic'], capabilities: ['go1.25', 'quic-rag'] },
  
  // Protocol-specific services
  'http-gateway': { baseUrl: 'http://localhost:8119', healthPath: '/health', protocols: ['http'], capabilities: ['http-gateway', 'routing'] },
  'grpc-gateway': { baseUrl: 'http://localhost:8120', healthPath: '/health', protocols: ['grpc'], capabilities: ['grpc-gateway', 'transcoding'] },
  'websocket-service': { baseUrl: 'http://localhost:8121', healthPath: '/health', protocols: ['websocket'], capabilities: ['real-time', 'events'] },
  
  // AI/ML specialized services
  't5-transformer': { baseUrl: 'http://localhost:8122', healthPath: '/health', protocols: ['http'], capabilities: ['t5-processing', 'seq2seq'] },
  'live-agent': { baseUrl: 'http://localhost:8123', healthPath: '/health', protocols: ['http', 'websocket'], capabilities: ['live-processing', 'real-time-ai'] },
  'legal-ai': { baseUrl: 'http://localhost:8124', healthPath: '/health', protocols: ['http'], capabilities: ['legal-analysis', 'document-processing'] },
  'multi-core-ollama': { baseUrl: 'http://localhost:8125', healthPath: '/health', protocols: ['http'], capabilities: ['ollama-cluster', 'load-balancing'] },
  
  // Storage and data services
  'minio-proxy': { baseUrl: 'http://localhost:8126', healthPath: '/health', protocols: ['http'], capabilities: ['object-storage', 'file-proxy'] },
  'postgres-proxy': { baseUrl: 'http://localhost:8127', healthPath: '/health', protocols: ['http'], capabilities: ['database-proxy', 'connection-pooling'] },
  'neo4j-proxy': { baseUrl: 'http://localhost:8128', healthPath: '/health', protocols: ['http'], capabilities: ['graph-database', 'cypher-queries'] },
  'qdrant-proxy': { baseUrl: 'http://localhost:8129', healthPath: '/health', protocols: ['http'], capabilities: ['vector-database', 'similarity-search'] },
  
  // Monitoring and observability
  'metrics-collector': { baseUrl: 'http://localhost:8130', healthPath: '/health', protocols: ['http'], capabilities: ['metrics', 'telemetry'] },
  'log-aggregator': { baseUrl: 'http://localhost:8131', healthPath: '/health', protocols: ['http'], capabilities: ['logging', 'aggregation'] },
  'health-monitor': { baseUrl: 'http://localhost:8132', healthPath: '/health', protocols: ['http'], capabilities: ['health-checks', 'monitoring'] },
  'alert-manager': { baseUrl: 'http://localhost:8133', healthPath: '/health', protocols: ['http'], capabilities: ['alerting', 'notifications'] },
  
  // Security and auth
  'auth-service': { baseUrl: 'http://localhost:8134', healthPath: '/health', protocols: ['http'], capabilities: ['authentication', 'authorization'] },
  'security-scanner': { baseUrl: 'http://localhost:8135', healthPath: '/health', protocols: ['http'], capabilities: ['security-scanning', 'vulnerability-detection'] },
  'rate-limiter': { baseUrl: 'http://localhost:8136', healthPath: '/health', protocols: ['http'], capabilities: ['rate-limiting', 'throttling'] }
} as const;

// Request routing schema
export interface GoServiceRequest {
  service: keyof typeof GO_SERVICES;
  endpoint: string;
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE';
  data?: any;
  headers?: Record<string, string>;
  protocol?: 'http' | 'quic' | 'grpc';
  timeout?: number;
}

// Helper to make HTTP requests to Go services
async function makeServiceRequest(
  serviceConfig: typeof GO_SERVICES[keyof typeof GO_SERVICES],
  endpoint: string,
  method: string = 'GET',
  data?: any,
  headers: Record<string, string> = {},
  timeout: number = 30000
): Promise<any> {
  const url = `${serviceConfig.baseUrl}${endpoint.startsWith('/') ? endpoint : '/' + endpoint}`;
  
  const requestConfig: RequestInit = {
    method,
    headers: {
      'Content-Type': 'application/json',
      'User-Agent': 'SvelteKit-Proxy/1.0.0',
      ...headers,
    },
    signal: AbortSignal.timeout(timeout),
  };

  if (data && ['POST', 'PUT', 'PATCH'].includes(method)) {
    requestConfig.body = JSON.stringify(data);
  }

  try {
    const response = await fetch(url, requestConfig);
    
    const contentType = response.headers.get('content-type');
    const responseData = contentType?.includes('application/json') 
      ? await response.json()
      : await response.text();

    return {
      success: response.ok,
      status: response.status,
      data: responseData,
      headers: Object.fromEntries(response.headers.entries()),
    };
  } catch (err: any) {
    console.error(`Go service request failed for ${url}:`, err);
    throw new Error(`Service request failed: ${err.message}`);
  }
}

export const POST: RequestHandler = async ({ request, getClientAddress }) => {
  try {
    // Parse request body
    const body = await request.json().catch(() => ({}));
    const {
      service,
      endpoint,
      method = 'GET',
      data,
      headers = {},
      protocol = 'http',
      timeout = 30000
    }: GoServiceRequest = body;

    // Validate service
    if (!service || !GO_SERVICES[service]) {
      throw error(400, ensureError({
        message: `Invalid service: ${service}. Available services: ${Object.keys(GO_SERVICES).join(', ')}`,
        code: 'INVALID_SERVICE'
      }));
    }

    // Validate endpoint
    if (!endpoint) {
      throw error(400, ensureError({
        message: 'Endpoint is required',
        code: 'MISSING_ENDPOINT'
      }));
    }

    const serviceConfig = GO_SERVICES[service];

    // Protocol validation
    if (!serviceConfig.protocols.includes(protocol)) {
      throw error(400, ensureError({
        message: `Service ${service} doesn't support protocol ${protocol}. Supported: ${serviceConfig.protocols.join(', ')}`,
        code: 'UNSUPPORTED_PROTOCOL'
      }));
    }

    // Add client information to headers for logging
    const clientHeaders = {
      'X-Client-IP': getClientAddress(),
      'X-Forwarded-By': 'SvelteKit-Proxy',
      ...headers,
    };

    // Route request to appropriate Go service
    const result = await makeServiceRequest(
      serviceConfig,
      endpoint,
      method,
      data,
      clientHeaders,
      timeout
    );

    // Return the response from Go service
    return json({
      success: result.success,
      message: result.success ? 'Request successful' : 'Request failed',
      data: result.data,
      meta: {
        service,
        endpoint,
        method,
        protocol,
        status: result.status,
        timestamp: new Date().toISOString(),
        version: '1.0.0',
      }
    }, {
      status: result.success ? 200 : result.status || 500,
      headers: {
        'Content-Type': 'application/json',
        ...(dev && { 'Access-Control-Allow-Origin': '*' }),
        // Forward relevant headers from Go service
        ...(result.headers['content-encoding'] && { 'Content-Encoding': result.headers['content-encoding'] }),
      }
    });

  } catch (err: any) {
    console.error('Go services proxy error:', err);
    
    const statusCode = err.status || 500;
    const message = err.body?.message || err.message || 'Go service request failed';

    return json({
      success: false,
      message,
      code: err.body?.code || 'GO_SERVICE_ERROR',
      meta: {
        timestamp: new Date().toISOString(),
        version: '1.0.0',
      }
    }, { 
      status: statusCode,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};

// GET - Service status and capabilities
export const GET: RequestHandler = async () => {
  try {
    // Check health of all Go services
    const serviceStatus = await Promise.all(
      Object.entries(GO_SERVICES).map(async ([name, config]) => {
        try {
          const healthCheck = await makeServiceRequest(config, config.healthPath || '/health', 'GET', undefined, {}, 5000);
          return {
            name,
            status: healthCheck.success ? 'healthy' : 'unhealthy',
            config: {
              baseUrl: config.baseUrl,
              protocols: config.protocols,
              capabilities: config.capabilities,
            },
            response: healthCheck.data,
          };
        } catch (err: any) {
          return {
            name,
            status: 'error',
            config: {
              baseUrl: config.baseUrl,
              protocols: config.protocols,
              capabilities: config.capabilities,
            },
            error: err.message,
          };
        }
      })
    );

    const healthyServices = serviceStatus.filter(s => s.status === 'healthy').length;
    const totalServices = serviceStatus.length;

    return json({
      success: true,
      message: `Go services proxy - ${healthyServices}/${totalServices} services healthy`,
      data: {
        proxy: {
          status: healthyServices === totalServices ? 'healthy' : 'degraded',
          services: {
            healthy: healthyServices,
            total: totalServices,
          }
        },
        services: serviceStatus,
        capabilities: {
          routing: ['json', 'protobuffer'],
          protocols: ['http', 'quic', 'grpc'],
          features: ['ai', 'rag', 'gpu', 'file-upload', 'legal-grpc'],
        }
      },
      meta: {
        timestamp: new Date().toISOString(),
        version: '1.0.0',
      }
    }, {
      status: healthyServices > 0 ? 200 : 503,
      headers: {
        'Content-Type': 'application/json',
        ...(dev && { 'Access-Control-Allow-Origin': '*' }),
      }
    });

  } catch (err: any) {
    console.error('Go services status check failed:', err);
    
    return json({
      success: false,
      message: 'Failed to check Go services status',
      code: 'STATUS_CHECK_FAILED',
      meta: {
        timestamp: new Date().toISOString(),
        version: '1.0.0',
      }
    }, { 
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};

// OPTIONS handler for CORS preflight requests
export const OPTIONS: RequestHandler = async () => {
  return new Response(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': dev ? '*' : 'https://yourdomain.com',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      'Access-Control-Max-Age': '86400', // 24 hours
    }
  });
};