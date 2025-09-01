import { json } from '@sveltejs/kit';
import { apiRegistry } from '$lib/server/api/service-registry';
import type { RequestHandler } from './$types.js';
import { URL } from "url";

export const GET: RequestHandler = async ({ url }) => {
  const showDetails = url.searchParams.has('details');
  const checkRoutes = url.searchParams.has('routes');
  
  try {
    // Get all service statuses
    const serviceStatuses = await apiRegistry.checkAllServices();
    
    // Basic system info
    const systemInfo = {
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      platform: process.platform,
      nodeVersion: process.version,
      memory: process.memoryUsage(),
      pid: process.pid
    };

    // Service summary
    const services = Array.from(serviceStatuses.values());
    const healthyServices = services.filter(s => s.status === 'healthy');
    const requiredServices = services.filter(s => s.required);
    const healthyRequiredServices = requiredServices.filter(s => s.status === 'healthy');
    
    const summary = {
      overall: {
        status: healthyRequiredServices.length === requiredServices.length ? 'operational' : 'degraded',
        healthScore: Math.round((healthyServices.length / services.length) * 100),
        servicesHealthy: healthyServices.length,
        servicesTotal: services.length,
        requiredHealthy: healthyRequiredServices.length,
        requiredTotal: requiredServices.length
      },
      models: {
        chat: 'gemma3:legal:latest',
        embeddings: 'nomic-embed-text',
        dimensions: 768
      },
      database: 'legal_ai_db',
      features: {
        vectorSearch: serviceStatuses.get('qdrant')?.status === 'healthy' || serviceStatuses.get('postgresql')?.status === 'healthy',
        aiProcessing: serviceStatuses.get('ollama')?.status === 'healthy',
        enhancedRag: serviceStatuses.get('enhanced_rag')?.status === 'healthy',
        gpuAcceleration: serviceStatuses.get('gpu_orchestrator')?.status === 'healthy',
        objectStorage: serviceStatuses.get('minio')?.status === 'healthy',
        caching: serviceStatuses.get('redis')?.status === 'healthy'
      }
    };

    // Build response based on query parameters
    let response = {
      system: systemInfo,
      summary,
      services: Object.fromEntries(serviceStatuses)
    };

    if (showDetails) {
      response.details = {
        apiRoutes: apiRegistry.generateServiceReport(),
        environment: {
          DATABASE_URL: import.meta.env.DATABASE_URL ? 'configured' : 'missing',
          OLLAMA_MODEL: import.meta.env.OLLAMA_MODEL || 'not set',
          EMBEDDING_MODEL: import.meta.env.EMBEDDING_MODEL || 'not set',
          REDIS_URL: import.meta.env.REDIS_URL ? 'configured' : 'missing',
          QDRANT_URL: import.meta.env.QDRANT_URL ? 'configured' : 'missing'
        }
      };
    }

    if (checkRoutes) {
      response.routes = await apiRegistry.validateApiRoutes();
    }

    // Set appropriate HTTP status
    const httpStatus = summary.overall.status === 'operational' ? 200 : 
                      summary.overall.status === 'degraded' ? 206 : 503;

    return json(response, { 
      status: httpStatus,
      headers: {
        'X-System-Status': summary.overall.status,
        'X-Health-Score': summary.overall.healthScore.toString(),
        'X-Services': `${summary.overall.servicesHealthy}/${summary.overall.servicesTotal}`,
        'X-Required-Services': `${summary.overall.requiredHealthy}/${summary.overall.requiredTotal}`,
        'Cache-Control': 'no-cache, must-revalidate'
      }
    });
    
  } catch (error) {
    console.error('System status check failed:', error);
    
    return json({
      system: { timestamp: new Date().toISOString() },
      summary: { overall: { status: 'error', error: error.message } },
      services: {}
    }, { status: 500 });
  }
};
