import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

interface ServiceHealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy' | 'unknown';
  message?: string;
  details?: any;
  responseTime?: number;
  lastChecked: string;
}

interface AggregatedHealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  services: {
    database: ServiceHealthStatus;
    redis: ServiceHealthStatus;
    neo4j: ServiceHealthStatus;
    ollama: ServiceHealthStatus;
    ocr: ServiceHealthStatus;
    vectorSearch: ServiceHealthStatus;
    minio: ServiceHealthStatus;
    cluster: ServiceHealthStatus;
    svelteKit: ServiceHealthStatus;
  };
  metadata: {
    nodeVersion: string;
    platform: string;
    arch: string;
    pid: number;
    uptime: number;
  };
  performance: {
    memoryUsage: NodeJS.MemoryUsage;
    cpuUsage: NodeJS.CpuUsage;
    loadAverage: number[] | string;
  };
  summary: {
    totalServices: number;
    healthyServices: number;
    degradedServices: number;
    unhealthyServices: number;
    unknownServices: number;
    overallHealthScore: number;
  };
}

async function checkServiceHealth(url: string, timeout: number = 5000): Promise<{ status: ServiceHealthStatus['status'], responseTime: number, details?: any }> {
  const startTime = Date.now();
  
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    
    const response = await fetch(url, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      signal: controller.signal
    });
    
    clearTimeout(timeoutId);
    const responseTime = Date.now() - startTime;
    
    if (response.ok) {
      const data = await response.json().catch(() => ({}));
      return { status: 'healthy', responseTime, details: data };
    } else {
      return { status: 'degraded', responseTime, details: { statusCode: response.status } };
    }
  } catch (error: any) {
    const responseTime = Date.now() - startTime;
    
    if (error.name === 'AbortError') {
      return { status: 'unhealthy', responseTime, details: { error: 'Request timeout' } };
    }
    
    return { 
      status: 'unhealthy', 
      responseTime, 
      details: { error: error.message || 'Connection failed' } 
    };
  }
}

async function checkDatabaseHealth(): Promise<ServiceHealthStatus> {
  const startTime = Date.now();
  try {
    // Make a request to our database health endpoint
    const result = await checkServiceHealth('/api/database/health');
    return {
      status: result.status,
      message: result.status === 'healthy' ? 'PostgreSQL connection successful' : 'Database connection issues',
      details: result.details,
      responseTime: result.responseTime,
      lastChecked: new Date().toISOString()
    };
  } catch (error: any) {
    return {
      status: 'unhealthy',
      message: `Database health check failed: ${error.message}`,
      responseTime: Date.now() - startTime,
      lastChecked: new Date().toISOString()
    };
  }
}

async function checkRedisHealth(): Promise<ServiceHealthStatus> {
  const startTime = Date.now();
  try {
    const result = await checkServiceHealth('/api/health/redis');
    return {
      status: result.status,
      message: result.status === 'healthy' ? 'Redis connection active' : 'Redis connection issues',
      details: result.details,
      responseTime: result.responseTime,
      lastChecked: new Date().toISOString()
    };
  } catch (error: any) {
    return {
      status: 'unknown',
      message: 'Redis health endpoint not accessible',
      responseTime: Date.now() - startTime,
      lastChecked: new Date().toISOString()
    };
  }
}

async function checkNeo4jHealth(): Promise<ServiceHealthStatus> {
  const startTime = Date.now();
  try {
    const result = await checkServiceHealth('/api/health/neo4j');
    return {
      status: result.status,
      message: result.status === 'healthy' ? 'Neo4j graph database active' : 'Neo4j connection issues',
      details: result.details,
      responseTime: result.responseTime,
      lastChecked: new Date().toISOString()
    };
  } catch (error: any) {
    return {
      status: 'unknown',
      message: 'Neo4j health endpoint not accessible',
      responseTime: Date.now() - startTime,
      lastChecked: new Date().toISOString()
    };
  }
}

async function checkOllamaHealth(): Promise<ServiceHealthStatus> {
  const startTime = Date.now();
  try {
    const result = await checkServiceHealth('http://localhost:11434/api/tags');
    return {
      status: result.status,
      message: result.status === 'healthy' ? 'Ollama AI service running' : 'Ollama service issues',
      details: result.details,
      responseTime: result.responseTime,
      lastChecked: new Date().toISOString()
    };
  } catch (error: any) {
    return {
      status: 'unhealthy',
      message: 'Ollama service unavailable',
      details: { error: error.message },
      responseTime: Date.now() - startTime,
      lastChecked: new Date().toISOString()
    };
  }
}

async function checkOCRHealth(): Promise<ServiceHealthStatus> {
  const startTime = Date.now();
  try {
    // Check OCR service using the same endpoint structure as API client
    const ocrBaseUrl = (globalThis as any).__OCR_BASE__ || '/api/ocr';
    const result = await checkServiceHealth(`${ocrBaseUrl}/status`);
    
    return {
      status: result.status,
      message: result.status === 'healthy' ? 'OCR processing service operational' : 'OCR service issues',
      details: {
        ...result.details,
        endpoint: `${ocrBaseUrl}/status`,
        capabilities: result.details?.features || []
      },
      responseTime: result.responseTime,
      lastChecked: new Date().toISOString()
    };
  } catch (error: any) {
    return {
      status: 'unhealthy',
      message: 'OCR service unavailable',
      details: { 
        error: error.message,
        endpoint: (globalThis as any).__OCR_BASE__ || '/api/ocr'
      },
      responseTime: Date.now() - startTime,
      lastChecked: new Date().toISOString()
    };
  }
}

async function checkVectorSearchHealth(): Promise<ServiceHealthStatus> {
  const startTime = Date.now();
  try {
    const result = await checkServiceHealth('/api/v1/vector/health');
    return {
      status: result.status,
      message: result.status === 'healthy' ? 'Vector search service operational' : 'Vector search issues',
      details: result.details,
      responseTime: result.responseTime,
      lastChecked: new Date().toISOString()
    };
  } catch (error: any) {
    return {
      status: 'unknown',
      message: 'Vector search health endpoint not accessible',
      responseTime: Date.now() - startTime,
      lastChecked: new Date().toISOString()
    };
  }
}

async function checkMinIOHealth(): Promise<ServiceHealthStatus> {
  const startTime = Date.now();
  try {
    const result = await checkServiceHealth('/api/v1/minio/health');
    return {
      status: result.status,
      message: result.status === 'healthy' ? 'MinIO storage service operational' : 'MinIO storage issues',
      details: result.details,
      responseTime: result.responseTime,
      lastChecked: new Date().toISOString()
    };
  } catch (error: any) {
    return {
      status: 'degraded',
      message: 'MinIO service unavailable - file storage in degraded mode',
      responseTime: Date.now() - startTime,
      lastChecked: new Date().toISOString()
    };
  }
}

async function checkClusterHealth(): Promise<ServiceHealthStatus> {
  const startTime = Date.now();
  try {
    const result = await checkServiceHealth('/api/v1/cluster/health');
    return {
      status: result.status,
      message: result.status === 'healthy' ? 'Cluster orchestration active' : 'Cluster management issues',
      details: result.details,
      responseTime: result.responseTime,
      lastChecked: new Date().toISOString()
    };
  } catch (error: any) {
    return {
      status: 'unknown',
      message: 'Cluster health endpoint not accessible',
      responseTime: Date.now() - startTime,
      lastChecked: new Date().toISOString()
    };
  }
}

function calculateOverallHealth(services: AggregatedHealthResponse['services']): AggregatedHealthResponse['status'] {
  const serviceStatuses = Object.values(services).map(s => s.status);
  const healthyCount = serviceStatuses.filter(s => s === 'healthy').length;
  const totalCount = serviceStatuses.length;
  
  if (healthyCount === totalCount) return 'healthy';
  if (healthyCount > totalCount / 2) return 'degraded';
  return 'unhealthy';
}

function calculateHealthScore(services: AggregatedHealthResponse['services']): number {
  const serviceStatuses = Object.values(services).map(s => s.status);
  const scores = serviceStatuses.map(status => {
    switch (status) {
      case 'healthy': return 100;
      case 'degraded': return 50;
      case 'unknown': return 25;
      case 'unhealthy': return 0;
      default: return 0;
    }
  });
  
  return Math.round(scores.reduce((sum, score) => sum + score, 0) / scores.length);
}

export const GET: RequestHandler = async () => {
  const timestamp = new Date().toISOString();
  
  try {
    console.log('= Running comprehensive system health check...');
    
    // Check all services concurrently for better performance
    const [
      database,
      redis,
      neo4j,
      ollama,
      ocr,
      vectorSearch,
      minio,
      cluster
    ] = await Promise.all([
      checkDatabaseHealth(),
      checkRedisHealth(),
      checkNeo4jHealth(),
      checkOllamaHealth(),
      checkOCRHealth(),
      checkVectorSearchHealth(),
      checkMinIOHealth(),
      checkClusterHealth()
    ]);
    
    // SvelteKit service is always healthy if we can respond
    const svelteKit: ServiceHealthStatus = {
      status: 'healthy',
      message: 'SvelteKit frontend service operational',
      details: {
        uptime: process.uptime(),
        memory: process.memoryUsage(),
        pid: process.pid
      },
      responseTime: 0,
      lastChecked: timestamp
    };
    
    const services = {
      database,
      redis,
      neo4j,
      ollama,
      ocr,
      vectorSearch,
      minio,
      cluster,
      svelteKit
    };
    
    const serviceStatuses = Object.values(services).map(s => s.status);
    const healthyServices = serviceStatuses.filter(s => s === 'healthy').length;
    const degradedServices = serviceStatuses.filter(s => s === 'degraded').length;
    const unhealthyServices = serviceStatuses.filter(s => s === 'unhealthy').length;
    const unknownServices = serviceStatuses.filter(s => s === 'unknown').length;
    
    const response: AggregatedHealthResponse = {
      status: calculateOverallHealth(services),
      timestamp,
      services,
      metadata: {
        nodeVersion: process.version,
        platform: process.platform,
        arch: process.arch,
        pid: process.pid,
        uptime: process.uptime()
      },
      performance: {
        memoryUsage: process.memoryUsage(),
        cpuUsage: process.cpuUsage(),
        loadAverage: process.platform !== 'win32' ? require('os').loadavg() : 'N/A (Windows)'
      },
      summary: {
        totalServices: serviceStatuses.length,
        healthyServices,
        degradedServices,
        unhealthyServices,
        unknownServices,
        overallHealthScore: calculateHealthScore(services)
      }
    };
    
    console.log(` Health check complete: ${healthyServices}/${serviceStatuses.length} services healthy (${response.summary.overallHealthScore}% overall health)`);
    
    const httpStatus = response.status === 'healthy' ? 200 : 
                      response.status === 'degraded' ? 206 : 503;
    
    return json(response, {
      status: httpStatus,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0',
        'X-Health-Check': 'comprehensive',
        'X-Health-Score': response.summary.overallHealthScore.toString(),
        'X-Healthy-Services': `${healthyServices}/${serviceStatuses.length}`
      }
    });
    
  } catch (error: any) {
    console.error('L Comprehensive health check failed:', error);
    
    return json({
      status: 'unhealthy',
      timestamp,
      error: 'Health check system failure',
      message: error instanceof Error ? error.message : 'Unknown error',
      services: {},
      summary: {
        totalServices: 0,
        healthyServices: 0,
        degradedServices: 0,
        unhealthyServices: 0,
        unknownServices: 0,
        overallHealthScore: 0
      }
    }, {
      status: 503,
      headers: {
        'Content-Type': 'application/json',
        'X-Health-Check': 'failed'
      }
    });
  }
};

// Optional: Support POST for forced health checks or specific service checks
export const POST: RequestHandler = async ({ request }) => {
  try {
    const { service, force } = await request.json();
    
    if (service && !force) {
      // Check specific service
      let serviceHealth: ServiceHealthStatus;
      
      switch (service.toLowerCase()) {
        case 'database':
          serviceHealth = await checkDatabaseHealth();
          break;
        case 'redis':
          serviceHealth = await checkRedisHealth();
          break;
        case 'neo4j':
          serviceHealth = await checkNeo4jHealth();
          break;
        case 'ollama':
          serviceHealth = await checkOllamaHealth();
          break;
        case 'ocr':
          serviceHealth = await checkOCRHealth();
          break;
        default:
          return json({
            error: 'Unknown service',
            availableServices: ['database', 'redis', 'neo4j', 'ollama', 'ocr']
          }, { status: 400 });
      }
      
      return json({
        service,
        health: serviceHealth,
        timestamp: new Date().toISOString()
      });
    }
    
    // If force is true or no specific service, do full health check
    return GET();
    
  } catch (error: any) {
    return json({
      error: 'Invalid request',
      message: error.message
    }, { status: 400 });
  }
};