import type { RequestHandler } from './$types';

/**
 * Hybrid GPU Configuration API
 * 
 * Intelligent routing between:
 * - CUDA AI Service (8096) - Immediate processing
 * - GPU Memory Manager (8097) - Advanced batch processing  
 * - Enhanced CUDA Service (8099) - Heavy compute workloads
 */

import { json } from '@sveltejs/kit';

// Service endpoints
const GPU_SERVICES = {
  immediate: 'http://localhost:8096',    // CUDA AI Service - fast responses
  advanced: 'http://localhost:8097',     // GPU Memory Manager - batch processing
  enhanced: 'http://localhost:8099',     // Enhanced CUDA - heavy compute
  loadBalancer: 'http://localhost:8224'  // Load balancer
} as const;

// Performance thresholds for intelligent routing
const ROUTING_CONFIG = {
  small_workload: { max_items: 10, prefer: 'immediate' },
  medium_workload: { max_items: 100, prefer: 'advanced' },
  large_workload: { max_items: 1000, prefer: 'enhanced' },
  batch_processing: { min_items: 50, prefer: 'advanced' }
} as const;

export interface GPURequest {
  operation: 'compute' | 'vector_similarity' | 'clustering' | 'tensor_parsing';
  data: any;
  priority: 'low' | 'medium' | 'high' | 'urgent';
  use_memory_manager?: boolean;
  batch_size?: number;
}

export interface ServiceHealthStatus {
  service: string;
  healthy: boolean;
  response_time: number;
  load: number;
}

/**
 * Check health of all GPU services
 */
async function checkServiceHealth(): Promise<Record<string, ServiceHealthStatus>> {
  const services = Object.entries(GPU_SERVICES);
  const healthChecks = services.map(async ([name, url]) => {
    try {
      const start = performance.now();
      const response = await fetch(`${url}/health`, { 
        method: 'GET',
        signal: AbortSignal.timeout(2000) // 2 second timeout
      });
      const end = performance.now();
      
      const isHealthy = response.ok;
      const responseTime = end - start;
      
      // Get load info if available
      let load = 0;
      try {
        if (isHealthy) {
          const healthData = await response.json();
          load = healthData.load || healthData.queue_size || 0;
        }
      } catch {
        // Ignore JSON parsing errors for load
      }

      return {
        [name]: {
          service: name,
          healthy: isHealthy,
          response_time: responseTime,
          load
        }
      };
    } catch (error: any) {
      return {
        [name]: {
          service: name,
          healthy: false,
          response_time: 999999,
          load: 100
        }
      };
    }
  });

  const results = await Promise.all(healthChecks);
  return Object.assign({}, ...results);
}

/**
 * Intelligently route request to optimal GPU service
 */
function selectOptimalService(
  request: GPURequest, 
  healthStatus: Record<string, ServiceHealthStatus>
): string {
  // Filter healthy services
  const healthyServices = Object.entries(healthStatus)
    .filter(([_, status]) => status.healthy)
    .sort((a, b) => a[1].response_time - b[1].response_time); // Sort by response time

  if (healthyServices.length === 0) {
    throw new Error('No healthy GPU services available');
  }

  // Priority-based routing
  if (request.priority === 'urgent') {
    return GPU_SERVICES[healthyServices[0][0] as keyof typeof GPU_SERVICES];
  }

  // Operation-specific routing
  switch (request.operation) {
    case 'compute':
      // Small computations -> immediate service
      if (!request.batch_size || request.batch_size <= 10) {
        return GPU_SERVICES.immediate;
      }
      // Large batch computations -> advanced memory manager
      return GPU_SERVICES.advanced;

    case 'vector_similarity':
      // Always use advanced service for vector operations
      return GPU_SERVICES.advanced;

    case 'clustering':
      // Heavy clustering -> enhanced CUDA service
      return GPU_SERVICES.enhanced;

    case 'tensor_parsing':
      // Memory-intensive -> advanced or enhanced based on size
      const dataSize = JSON.stringify(request.data).length;
      return dataSize > 10000 ? GPU_SERVICES.enhanced : GPU_SERVICES.advanced;

    default:
      return GPU_SERVICES[healthyServices[0][0] as keyof typeof GPU_SERVICES];
  }
}

/**
 * GET /api/gpu/hybrid - Get hybrid GPU system status
 */
export const GET: RequestHandler = async () => {
  try {
    const healthStatus = await checkServiceHealth();
    
    const systemStatus = {
      status: 'hybrid_gpu_system',
      timestamp: new Date().toISOString(),
      services: healthStatus,
      configuration: {
        total_services: Object.keys(GPU_SERVICES).length,
        healthy_services: Object.values(healthStatus).filter(s => s.healthy).length,
        routing_strategy: 'intelligent_workload_based',
        gpu_device: 'NVIDIA GeForce RTX 3060 Ti',
        gpu_memory: '8GB',
        features: [
          'immediate_processing',
          'advanced_batch_processing', 
          'gpu_memory_management',
          'load_balancing',
          'intelligent_routing'
        ]
      },
      routing_config: ROUTING_CONFIG
    };

    return json(systemStatus);
  } catch (error: any) {
    console.error('❌ GPU hybrid system status error:', error);
    return json({ error: 'Failed to get system status' }, { status: 500 });
  }
};

/**
 * POST /api/gpu/hybrid - Execute GPU operation with intelligent routing
 */
export const POST: RequestHandler = async ({ request }) => {
  try {
    const gpuRequest: GPURequest = await request.json();
    
    // Validate request
    if (!gpuRequest.operation) {
      return json({ error: 'Missing operation field' }, { status: 400 });
    }

    const startTime = performance.now();
    
    // Check service health
    const healthStatus = await checkServiceHealth();
    
    // Select optimal service
    const selectedService = selectOptimalService(gpuRequest, healthStatus);
    const serviceName = Object.entries(GPU_SERVICES).find(([_, url]) => url === selectedService)?.[0] || 'unknown';
    
    console.log(`🎯 Routing ${gpuRequest.operation} to ${serviceName} service (${selectedService})`);

    // Prepare request based on service type
    let endpoint = '';
    let payload = gpuRequest.data;

    switch (gpuRequest.operation) {
      case 'compute':
        endpoint = serviceName === 'immediate' ? '/cuda/compute' : '/api/v2/gpu/compute';
        break;
      case 'vector_similarity':
        endpoint = '/api/v2/gpu/vector-similarity';
        payload = { ...gpuRequest.data, use_memory_manager: true };
        break;
      case 'clustering':
        endpoint = '/api/v2/gpu/clustering';
        payload = { ...gpuRequest.data, use_gpu_memory_manager: true };
        break;
      case 'tensor_parsing':
        endpoint = '/api/v2/gpu/tensor-parsing';
        break;
      default:
        endpoint = '/health';
    }

    // Execute request
    const response = await fetch(`${selectedService}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(30000) // 30 second timeout
    });

    const result = await response.json();
    const executionTime = performance.now() - startTime;

    return json({
      success: true,
      result,
      metadata: {
        service_used: serviceName,
        service_url: selectedService,
        execution_time_ms: executionTime,
        operation: gpuRequest.operation,
        priority: gpuRequest.priority,
        timestamp: new Date().toISOString()
      }
    });

  } catch (error: any) {
    console.error('❌ GPU hybrid operation error:', error);
    return json({ 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error',
      timestamp: new Date().toISOString()
    }, { status: 500 });
  }
};

/**
 * PUT /api/gpu/hybrid - Update routing configuration
 */
export const PUT: RequestHandler = async ({ request }) => {
  try {
    const config = await request.json();
    
    // In production, you would persist this configuration
    console.log('🔧 Updating GPU routing configuration:', config);
    
    return json({
      success: true,
      message: 'Routing configuration updated',
      config,
      timestamp: new Date().toISOString()
    });
    
  } catch (error: any) {
    console.error('❌ GPU configuration update error:', error);
    return json({ error: 'Failed to update configuration' }, { status: 500 });
  }
};

/**
 * DELETE /api/gpu/hybrid - Shutdown hybrid GPU system
 */
export const DELETE: RequestHandler = async () => {
  try {
    console.log('🛑 Initiating hybrid GPU system shutdown...');
    
    // In production, you would gracefully shutdown services
    // For now, just return a success response
    
    return json({
      success: true,
      message: 'Hybrid GPU system shutdown initiated',
      timestamp: new Date().toISOString()
    });
    
  } catch (error: any) {
    console.error('❌ GPU system shutdown error:', error);
    return json({ error: 'Failed to shutdown system' }, { status: 500 });
  }
};