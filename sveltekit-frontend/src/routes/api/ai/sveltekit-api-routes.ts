/**
 * SvelteKit API Routes Configuration
 * Defines all API endpoints and their handlers for the Legal AI platform
 */

export interface ApiRoute {
  path: string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
  handler: string;
  description: string;
  params?: string[];
  body?: Record<string, any>;
  response?: Record<string, any>;
}

export interface ApiRouteGroup {
  name: string;
  description: string;
  baseUrl: string;
  routes: ApiRoute[];
}

/**
 * All API routes for the Legal AI platform
 */
export const API_ROUTES: Record<string, ApiRouteGroup> = {
  // Authentication routes
  auth: {
    name: 'Authentication',
    description: 'User authentication and session management',
    baseUrl: '/api/auth',
    routes: [
      {
        path: '/login',
        method: 'POST',
        handler: 'src/routes/api/auth/login/+server.ts',
        description: 'User login',
        body: { email: 'string', password: 'string' },
        response: { success: 'boolean', user: 'object', token: 'string' }
      },
      {
        path: '/register',
        method: 'POST',
        handler: 'src/routes/api/auth/register/+server.ts',
        description: 'User registration',
        body: { 
          email: 'string', 
          password: 'string', 
          name: 'string', 
          role: 'string' 
        }
      },
      {
        path: '/logout',
        method: 'POST',
        handler: 'src/routes/api/auth/logout/+server.ts',
        description: 'User logout'
      },
      {
        path: '/me',
        method: 'GET',
        handler: 'src/routes/api/auth/me/+server.ts',
        description: 'Get current user info'
      }
    ]
  },

  // Case management routes
  cases: {
    name: 'Case Management',
    description: 'Legal case CRUD operations and search',
    baseUrl: '/api/cases',
    routes: [
      {
        path: '/',
        method: 'GET',
        handler: 'src/routes/api/cases/+server.ts',
        description: 'List cases with pagination and filtering',
        params: ['page', 'limit', 'search', 'status', 'type']
      },
      {
        path: '/',
        method: 'POST',
        handler: 'src/routes/api/cases/+server.ts',
        description: 'Create new case',
        body: {
          title: 'string',
          description: 'string',
          type: 'string',
          status: 'string',
          priority: 'string'
        }
      },
      {
        path: '/[id]',
        method: 'GET',
        handler: 'src/routes/api/cases/[id]/+server.ts',
        description: 'Get case by ID',
        params: ['id']
      },
      {
        path: '/[id]',
        method: 'PUT',
        handler: 'src/routes/api/cases/[id]/+server.ts',
        description: 'Update case',
        params: ['id']
      },
      {
        path: '/[id]',
        method: 'DELETE',
        handler: 'src/routes/api/cases/[id]/+server.ts',
        description: 'Delete case',
        params: ['id']
      },
      {
        path: '/search',
        method: 'POST',
        handler: 'src/routes/api/cases/search/+server.ts',
        description: 'Advanced case search with Fuse.js',
        body: { query: 'string', filters: 'object' }
      }
    ]
  },

  // Document management routes
  documents: {
    name: 'Document Management',
    description: 'Document upload, processing, and retrieval',
    baseUrl: '/api/documents',
    routes: [
      {
        path: '/',
        method: 'GET',
        handler: 'src/routes/api/documents/+server.ts',
        description: 'List documents',
        params: ['caseId', 'page', 'limit', 'type', 'status']
      },
      {
        path: '/upload',
        method: 'POST',
        handler: 'src/routes/api/documents/upload/+server.ts',
        description: 'Upload document',
        body: { file: 'File', metadata: 'object' }
      },
      {
        path: '/process',
        method: 'POST',
        handler: 'src/routes/api/documents/process/+server.ts',
        description: 'Process uploaded document',
        body: { file: 'File', options: 'object' }
      },
      {
        path: '/[id]',
        method: 'GET',
        handler: 'src/routes/api/documents/[id]/+server.ts',
        description: 'Get document by ID',
        params: ['id']
      },
      {
        path: '/[id]/process',
        method: 'POST',
        handler: 'src/routes/api/documents/[id]/process/+server.ts',
        description: 'Process existing document',
        params: ['id']
      },
      {
        path: '/[id]/status',
        method: 'GET',
        handler: 'src/routes/api/documents/[id]/status/+server.ts',
        description: 'Get processing status',
        params: ['id']
      },
      {
        path: '/search',
        method: 'POST',
        handler: 'src/routes/api/documents/search/+server.ts',
        description: 'Search documents',
        body: { query: 'string', options: 'object' }
      },
      {
        path: '/analytics',
        method: 'GET',
        handler: 'src/routes/api/documents/analytics/+server.ts',
        description: 'Document processing analytics'
      }
    ]
  },

  // Legal AI routes
  legal: {
    name: 'Legal AI Processing',
    description: 'Legal document analysis and processing',
    baseUrl: '/api/legal',
    routes: [
      {
        path: '/ingest',
        method: 'POST',
        handler: 'src/routes/api/legal/ingest/+server.ts',
        description: 'Ingest legal PDFs with WHO/WHAT/WHY/HOW extraction',
        body: {
          pdfFiles: 'File[]',
          jurisdiction: 'string',
          caseId: 'string',
          enhanceRAG: 'boolean'
        }
      },
      {
        path: '/analyze',
        method: 'POST',
        handler: 'src/routes/api/legal/analyze/+server.ts',
        description: 'Analyze legal document',
        body: { documentId: 'string', analysisType: 'string' }
      },
      {
        path: '/entities',
        method: 'POST',
        handler: 'src/routes/api/legal/entities/+server.ts',
        description: 'Extract legal entities',
        body: { text: 'string', jurisdiction: 'string' }
      }
    ]
  },

  // Evidence processing routes
  evidence: {
    name: 'Evidence Processing',
    description: 'Evidence document processing with XState workflow',
    baseUrl: '/api/evidence',
    routes: [
      {
        path: '/process',
        method: 'POST',
        handler: 'src/routes/evidence/+server.ts',
        description: 'Process evidence through XState machine',
        body: {
          evidenceId: 'string',
          caseId: 'string',
          content: 'string',
          metadata: 'object'
        }
      },
      {
        path: '/[id]/status',
        method: 'GET',
        handler: 'src/routes/api/evidence/[id]/status/+server.ts',
        description: 'Get evidence processing status',
        params: ['id']
      },
      {
        path: '/[id]/retry',
        method: 'POST',
        handler: 'src/routes/api/evidence/[id]/retry/+server.ts',
        description: 'Retry failed evidence processing',
        params: ['id']
      }
    ]
  },

  // AI and RAG routes
  ai: {
    name: 'AI Services',
    description: 'AI processing and RAG queries',
    baseUrl: '/api/ai',
    routes: [
      {
        path: '/chat',
        method: 'POST',
        handler: 'src/routes/api/ai/chat/+server.ts',
        description: 'AI chat interface',
        body: { message: 'string', context: 'object' }
      },
      {
        path: '/rag',
        method: 'POST',
        handler: 'src/routes/api/v1/rag/+server.ts',
        description: 'Enhanced RAG query',
        body: { query: 'string', caseId: 'string', options: 'object' }
      },
      {
        path: '/summarize',
        method: 'POST',
        handler: 'src/routes/api/ai/summarize/+server.ts',
        description: 'Document summarization',
        body: { documentId: 'string', options: 'object' }
      }
    ]
  },

  // Vector operations routes
  vector: {
    name: 'Vector Operations',
    description: 'Vector search and embedding operations',
    baseUrl: '/api/v1/vector',
    routes: [
      {
        path: '/search',
        method: 'POST',
        handler: 'src/routes/api/v1/vector/search/+server.ts',
        description: 'Vector similarity search',
        body: { query: 'string', options: 'object' }
      },
      {
        path: '/index',
        method: 'POST',
        handler: 'src/routes/api/v1/vector/index/+server.ts',
        description: 'Index document for vector search',
        body: { documentId: 'string', content: 'string', metadata: 'object' }
      }
    ]
  },

  // Graph operations routes
  graph: {
    name: 'Graph Operations',
    description: 'Neo4j knowledge graph operations',
    baseUrl: '/api/v1/graph',
    routes: [
      {
        path: '/query',
        method: 'POST',
        handler: 'src/routes/api/v1/graph/query/+server.ts',
        description: 'Execute Cypher query',
        body: { cypher: 'string', parameters: 'object' }
      },
      {
        path: '/entities',
        method: 'POST',
        handler: 'src/routes/api/v1/graph/entities/+server.ts',
        description: 'Create entity relationships',
        body: { entities: 'array', relationships: 'array' }
      }
    ]
  },

  // System and cluster routes
  system: {
    name: 'System Management',
    description: 'System health and cluster management',
    baseUrl: '/api/v1',
    routes: [
      {
        path: '/cluster/health',
        method: 'GET',
        handler: 'src/routes/api/v1/cluster/health/+server.ts',
        description: 'Cluster health status'
      },
      {
        path: '/cluster/metrics',
        method: 'GET',
        handler: 'src/routes/api/v1/cluster/metrics/+server.ts',
        description: 'Cluster performance metrics'
      },
      {
        path: '/cluster/services',
        method: 'GET',
        handler: 'src/routes/api/v1/cluster/services/+server.ts',
        description: 'List cluster services'
      },
      {
        path: '/system/status',
        method: 'GET',
        handler: 'src/routes/api/system/status/+server.ts',
        description: 'System status check'
      }
    ]
  },

  // GPU orchestration routes
  gpu: {
    name: 'GPU Orchestration',
    description: 'GPU task dispatch and monitoring',
    baseUrl: '/api/v1/gpu',
    routes: [
      {
        path: '/orchestrate',
        method: 'POST',
        handler: 'src/routes/api/v1/gpu/orchestrate/+server.ts',
        description: 'Dispatch GPU task',
        body: { action: 'string', data: 'object', config: 'object' }
      },
      {
        path: '/metrics',
        method: 'GET',
        handler: 'src/routes/api/gpu/metrics/+server.ts',
        description: 'GPU utilization metrics'
      },
      {
        path: '/status',
        method: 'GET',
        handler: 'src/routes/api/v1/gpu/status/+server.ts',
        description: 'GPU cluster status'
      }
    ]
  },

  // Context7 autosolve routes
  autosolve: {
    name: 'Context7 Autosolve',
    description: 'Automated error resolution with Context7',
    baseUrl: '/api/context7-autosolve',
    routes: [
      {
        path: '/',
        method: 'GET',
        handler: 'src/routes/api/context7-autosolve/+server.ts',
        description: 'Get autosolve status',
        params: ['action']
      },
      {
        path: '/',
        method: 'POST',
        handler: 'src/routes/api/context7-autosolve/+server.ts',
        description: 'Trigger autosolve cycle',
        body: { force: 'boolean', threshold: 'number' }
      }
    ]
  }
};

/**
 * Get all API routes as a flat array
 */
export function getAllRoutes(): Array<ApiRoute & { group: string; fullPath: string }> {
  const routes: Array<ApiRoute & { group: string; fullPath: string }> = [];
  
  Object.entries(API_ROUTES).forEach(([groupKey, group]) => {
    group.routes.forEach(route => {
      routes.push({
        ...route,
        group: groupKey,
        fullPath: group.baseUrl + route.path
      });
    });
  });
  
  return routes;
}

/**
 * Get routes by method
 */
export function getRoutesByMethod(method: string): Array<ApiRoute & { group: string; fullPath: string }> {
  return getAllRoutes().filter(route => route.method === method);
}

/**
 * Test endpoint availability and return status
 */
export async function testEndpoint(path: string, method: string = 'GET'): Promise<{
  available: boolean;
  responseTime?: number;
  status?: number;
  error?: string;
}> {
  const startTime = Date.now();
  try {
    const response = await fetch(`http://localhost:5173${path}`, {
      method,
      signal: AbortSignal.timeout(5000)
    });
    
    return {
      available: true,
      responseTime: Date.now() - startTime,
      status: response.status
    };
  } catch (error) {
    return {
      available: false,
      responseTime: Date.now() - startTime,
      error: (error as Error).message
    };
  }
}

/**
 * Get routes by group
 */
export function getRoutesByGroup(groupKey: string): ApiRoute[] {
  return API_ROUTES[groupKey]?.routes || [];
}

/**
 * Find route by path and method
 */
export function findRoute(path: string, method: string): (ApiRoute & { group: string; fullPath: string }) | null {
  return getAllRoutes().find(route => 
    route.fullPath === path && route.method === method
  ) || null;
}

/**
 * Generate OpenAPI/Swagger documentation
 */
export function generateApiDocs() {
  return {
    openapi: '3.0.0',
    info: {
      title: 'Legal AI Platform API',
      version: '1.0.0',
      description: 'Complete API documentation for the Legal AI platform'
    },
    servers: [
      {
        url: 'http://localhost:5173',
        description: 'Development server'
      }
    ],
    paths: getAllRoutes().reduce((paths, route) => {
      const pathKey = route.fullPath.replace(/\[(\w+)\]/g, '{$1}');
      
      if (!paths[pathKey]) {
        paths[pathKey] = {};
      }
      
      paths[pathKey][route.method.toLowerCase()] = {
        summary: route.description,
        tags: [route.group],
        parameters: route.params?.map(param => ({
          name: param,
          in: param === 'id' ? 'path' : 'query',
          required: param === 'id',
          schema: { type: 'string' }
        })),
        requestBody: route.body ? {
          content: {
            'application/json': {
              schema: {
                type: 'object',
                properties: Object.fromEntries(
                  Object.entries(route.body).map(([key, type]) => [
                    key,
                    { type: type === 'File' || type === 'File[]' ? 'string' : type }
                  ])
                )
              }
            }
          }
        } : undefined,
        responses: {
          200: {
            description: 'Successful response',
            content: {
              'application/json': {
                schema: route.response || { type: 'object' }
              }
            }
          }
        }
      };
      
      return paths;
    }, {} as any)
  };
}

export default API_ROUTES;