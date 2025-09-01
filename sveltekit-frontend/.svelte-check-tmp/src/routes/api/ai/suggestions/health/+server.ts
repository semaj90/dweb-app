import { json } from '@sveltejs/kit';
import type { RequestEvent } from '@sveltejs/kit';
import type { RequestHandler } from './$types';


// Import health check functions from our services
import { ollamaSuggestionsService } from '$lib/services/ollama-suggestions-service.js';
import { enhancedRAGSuggestionsService } from '$lib/services/enhanced-rag-suggestions-service.js';
import { aiSuggestionsClient } from '$lib/services/ai-suggestions-grpc-client.js';

/**
 * Health check endpoint for AI Suggestions services
 */
export async function GET({ url }: RequestEvent): Promise<any> {
  const startTime = Date.now();
  
  try {
    // Check all AI suggestion services in parallel
    const [
      ollamaHealth,
      ragHealth,
      grpcHealth
    ] = await Promise.allSettled([
      checkOllamaService(),
      checkEnhancedRAGService(),
      checkGRPCService()
    ]);

    const healthStatus = {
      status: 'operational',
      timestamp: new Date().toISOString(),
      responseTime: Date.now() - startTime,
      services: {
        ollama: getHealthResult(ollamaHealth),
        enhancedRAG: getHealthResult(ragHealth),
        protobufGRPC: getHealthResult(grpcHealth)
      },
      overall: {
        healthy: 0,
        degraded: 0,
        down: 0
      }
    };

    // Calculate overall health metrics
    Object.values(healthStatus.services).forEach(service => {
      if (service.status === 'healthy') {
        healthStatus.overall.healthy++;
      } else if (service.status === 'degraded') {
        healthStatus.overall.degraded++;
      } else {
        healthStatus.overall.down++;
      }
    });

    // Determine overall status
    if (healthStatus.overall.down > 0) {
      healthStatus.status = 'partial_outage';
    } else if (healthStatus.overall.degraded > 0) {
      healthStatus.status = 'degraded';
    }

    const httpStatus = healthStatus.status === 'operational' ? 200 : 
                      healthStatus.status === 'degraded' ? 207 : 503;

    return json(healthStatus, { status: httpStatus });

  } catch (error: any) {
    return json({
      status: 'error',
      timestamp: new Date().toISOString(),
      responseTime: Date.now() - startTime,
      error: error instanceof Error ? error.message : 'Unknown error',
      services: {
        ollama: { status: 'unknown', error: 'Health check failed' },
        enhancedRAG: { status: 'unknown', error: 'Health check failed' },
        protobufGRPC: { status: 'unknown', error: 'Health check failed' }
      }
    }, { status: 500 });
  }
}

async function checkOllamaService(): Promise<any> {
  try {
    const isHealthy = await ollamaSuggestionsService.healthCheck();
    const models = await ollamaSuggestionsService.getAvailableModels();
    const config = ollamaSuggestionsService.getConfig();

    return {
      status: isHealthy ? 'healthy' : 'down',
      config,
      availableModels: models.length,
      models: models.slice(0, 5) // Limit for response size
    };
  } catch (error: any) {
    return {
      status: 'down',
      error: error instanceof Error ? error.message : 'Unknown error'
    };
  }
}

async function checkEnhancedRAGService(): Promise<any> {
  try {
    const health = await enhancedRAGSuggestionsService.healthCheck();
    const config = enhancedRAGSuggestionsService.getServiceInfo();

    return {
      status: health.available ? 'healthy' : 'down',
      version: health.version,
      capabilities: health.capabilities,
      responseTime: health.responseTime,
      config
    };
  } catch (error: any) {
    return {
      status: 'down',
      error: error instanceof Error ? error.message : 'Unknown error'
    };
  }
}

async function checkGRPCService(): Promise<any> {
  try {
    const isHealthy = await aiSuggestionsClient.healthCheck();
    const status = aiSuggestionsClient.getConnectionStatus();

    return {
      status: isHealthy ? 'healthy' : 'down',
      connected: status.connected,
      serviceUrl: status.serviceUrl
    };
  } catch (error: any) {
    return {
      status: 'down',
      error: error instanceof Error ? error.message : 'Unknown error'
    };
  }
}

function getHealthResult(promiseResult: PromiseSettledResult<any>) {
  if (promiseResult.status === 'fulfilled') {
    return promiseResult.value;
  } else {
    return {
      status: 'down',
      error: promiseResult.reason?.message || 'Service check failed'
    };
  }
}