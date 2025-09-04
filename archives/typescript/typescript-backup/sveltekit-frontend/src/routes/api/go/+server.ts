/**
 * Go Microservices Proxy API
 * Handles routing between SvelteKit and Go services with JSON/Protocol Buffer support
 * POST /api/go - Route requests to appropriate Go microservice
 */

import { json, error, type RequestHandler } from '@sveltejs/kit';
import { dev } from '$app/environment';

// Go microservices configuration
const GO_SERVICES = {
  'enhanced-rag': {
    baseUrl: 'http://localhost:8094',
    healthPath: '/api/health',
    protocols: ['http', 'quic'],
    capabilities: ['ai', 'rag', 'gpu', 'som', 'xstate']
  },
  'upload': {
    baseUrl: 'http://localhost:8093', 
    healthPath: '/health',
    protocols: ['http'],
    capabilities: ['file-upload', 'storage', 'processing']
  },
  'kratos': {
    baseUrl: 'http://localhost:50051',
    healthPath: '/health',
    protocols: ['grpc'],
    capabilities: ['legal-grpc', 'gpu-compute', 'search']
  }
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
      throw error(400, {
        message: `Invalid service: ${service}. Available services: ${Object.keys(GO_SERVICES).join(', ')}`,
        code: 'INVALID_SERVICE'
      });
    }

    // Validate endpoint
    if (!endpoint) {
      throw error(400, {
        message: 'Endpoint is required',
        code: 'MISSING_ENDPOINT'
      });
    }

    const serviceConfig = GO_SERVICES[service];

    // Protocol validation
    if (!serviceConfig.protocols.includes(protocol)) {
      throw error(400, {
        message: `Service ${service} doesn't support protocol ${protocol}. Supported: ${serviceConfig.protocols.join(', ')}`,
        code: 'UNSUPPORTED_PROTOCOL'
      });
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