import type { RequestHandler } from './$types.js';

// Unified Service Orchestrator API Endpoint
// Provides HTTP interface to the complete GPU/WASM integration system

import { json, error } from '@sveltejs/kit';
import { getOrchestrator, type SystemHealth } from '$lib/services/unified-service-orchestrator';
import { TaskPriority } from '$lib/services/unified-service-orchestrator';
import { URL } from "url";

const orchestrator = getOrchestrator({
  enabledServices: ['wasmGPU', 'quicGateway', 'llamaOllama', 'nesGPUBridge'],
  performanceThresholds: {
    maxLatency: 2000,
    minThroughput: 5,
    maxCpuUsage: 85,
    maxMemoryUsage: 75
  },
  retryConfiguration: {
    maxRetries: 3,
    backoffMultiplier: 2,
    initialDelay: 100
  },
  monitoring: {
    healthCheckInterval: 30000,
    metricsRetentionPeriod: 3600000,
    alertThresholds: {
      latency: 3000,
      errorRate: 10,
      throughput: 3
    }
  }
});

// GET /api/v1/orchestrator - System health and status
export const GET: RequestHandler = async ({ url }) => {
  try {
    const endpoint = url.searchParams.get('endpoint') || 'health';
    
    switch (endpoint) {
      case 'health':
        const health = orchestrator.getSystemHealth();
        return json({
          success: true,
          data: health,
          timestamp: new Date().toISOString()
        });
        
      case 'metrics':
        const metrics = orchestrator.getPerformanceMetrics();
        return json({
          success: true,
          data: {
            metrics,
            count: metrics.length,
            latestMetric: metrics[metrics.length - 1] || null
          },
          timestamp: new Date().toISOString()
        });
        
      case 'tasks':
        const activeTasks = orchestrator.getActiveTasks();
        const taskQueue = orchestrator.getTaskQueue();
        return json({
          success: true,
          data: {
            activeTasks,
            taskQueue,
            activeCount: activeTasks.length,
            queueCount: taskQueue.length
          },
          timestamp: new Date().toISOString()
        });
        
      default:
        throw error(400, `Unknown endpoint: ${endpoint}`);
    }
  } catch (err: any) {
    console.error('[Orchestrator API] GET error:', err);
    throw error(500, `Internal server error: ${err}`);
  }
};

// POST /api/v1/orchestrator - Execute orchestrated operations
export const POST: RequestHandler = async ({ request, url }) => {
  try {
    const { operation, data, options = {} } = await request.json();
    
    if (!operation || !data) {
      throw error(400, 'Missing required fields: operation, data');
    }

    // Parse priority
    const priority = options.priority ? 
      TaskPriority[options.priority.toUpperCase() as keyof typeof TaskPriority] : 
      TaskPriority.NORMAL;

    let result;
    const startTime = Date.now();

    switch (operation) {
      case 'processDocument':
        if (typeof data.document !== 'string') {
          throw error(400, 'Document must be a string');
        }
        
        result = await orchestrator.processLegalDocument(data.document, {
          ...options,
          priority,
          analysisType: data.analysisType || 'comprehensive',
          maxTokens: data.maxTokens || 2048,
          temperature: data.temperature || 0.7
        });
        break;

      case 'performInference':
        if (!Array.isArray(data.input)) {
          throw error(400, 'Input must be an array');
        }
        
        const inputArray = new Float32Array(data.input);
        result = await orchestrator.performNeuralInference(inputArray, {
          ...options,
          priority,
          modelType: data.modelType || 'transformer',
          precision: data.precision || 'fp32'
        });
        break;

      case 'processCanvas':
        if (!data.canvasState || !data.canvasState.width || !data.canvasState.height) {
          throw error(400, 'Invalid canvas state');
        }
        
        // Convert array back to Uint8ClampedArray if needed
        const canvasData = Array.isArray(data.canvasState.data) ?
          new Uint8ClampedArray(data.canvasState.data) :
          data.canvasState.data;
          
        result = await orchestrator.processCanvasState({
          width: data.canvasState.width,
          height: data.canvasState.height,
          data: canvasData,
          format: data.canvasState.format || 'RGBA'
        }, {
          ...options,
          priority,
          targetBitDepth: data.targetBitDepth || 24,
          optimization: data.optimization || 'balanced'
        });
        break;

      case 'executeGPU':
        if (!data.operation) {
          throw error(400, 'GPU operation not specified');
        }
        
        result = await orchestrator.executeGPUComputation(data.operation, data.params || {}, {
          ...options,
          priority,
          precision: data.precision || 'fp32',
          timeout: data.timeout || 15000
        });
        break;

      case 'matmul':
        if (!Array.isArray(data.a) || !Array.isArray(data.b)) {
          throw error(400, 'Matrix multiplication requires arrays a and b');
        }
        
        result = await orchestrator.executeGPUComputation('matmul', {
          a: data.a,
          b: data.b,
          m: data.m || Math.sqrt(data.a.length),
          n: data.n || Math.sqrt(data.b.length),
          k: data.k || Math.sqrt(data.a.length)
        }, { ...options, priority });
        break;

      case 'attention':
        if (!Array.isArray(data.query) || !Array.isArray(data.key) || !Array.isArray(data.value)) {
          throw error(400, 'Attention requires query, key, and value arrays');
        }
        
        result = await orchestrator.executeGPUComputation('attention', {
          query: data.query,
          key: data.key,
          value: data.value,
          seq_len: data.seq_len || Math.sqrt(data.query.length),
          dim: data.dim || Math.sqrt(data.query.length)
        }, { ...options, priority });
        break;

      case 'conv2d':
        if (!Array.isArray(data.input) || !Array.isArray(data.kernel)) {
          throw error(400, 'Convolution requires input and kernel arrays');
        }
        
        result = await orchestrator.executeGPUComputation('conv2d', {
          input: data.input,
          kernel: data.kernel,
          width: data.width || Math.sqrt(data.input.length),
          height: data.height || Math.sqrt(data.input.length),
          kernel_size: data.kernel_size || Math.sqrt(data.kernel.length)
        }, { ...options, priority });
        break;

      default:
        throw error(400, `Unknown operation: ${operation}`);
    }

    const totalTime = Date.now() - startTime;

    return json({
      success: true,
      data: result,
      operation,
      totalProcessingTime: totalTime,
      timestamp: new Date().toISOString(),
      metadata: {
        operation,
        priority: TaskPriority[priority],
        servicesUsed: result.servicesUsed,
        fallbacksTriggered: result.fallbacksTriggered,
        performance: result.performance
      }
    });

  } catch (err: any) {
    console.error('[Orchestrator API] POST error:', err);
    
    if (err && typeof err === 'object' && 'status' in err) {
      throw err; // Re-throw SvelteKit errors
    }
    
    return json({
      success: false,
      error: String(err),
      timestamp: new Date().toISOString()
    }, { status: 500 });
  }
};

// PUT /api/v1/orchestrator - Update orchestrator configuration
export const PUT: RequestHandler = async ({ request }) => {
  try {
    const { config } = await request.json();
    
    if (!config) {
      throw error(400, 'Configuration object required');
    }

    // Note: In a full implementation, we'd need to reinitialize the orchestrator
    // with new configuration. For now, we'll just return the current health.
    const health = orchestrator.getSystemHealth();
    
    return json({
      success: true,
      message: 'Configuration update acknowledged',
      data: {
        health,
        requestedConfig: config
      },
      timestamp: new Date().toISOString()
    });

  } catch (err: any) {
    console.error('[Orchestrator API] PUT error:', err);
    throw error(500, `Configuration update failed: ${err}`);
  }
};

// DELETE /api/v1/orchestrator - Shutdown orchestrator (for maintenance)
export const DELETE: RequestHandler = async ({ url }) => {
  try {
    const confirm = url.searchParams.get('confirm');
    
    if (confirm !== 'true') {
      throw error(400, 'Must confirm shutdown with ?confirm=true');
    }

    // In a production environment, you'd want proper authentication here
    await orchestrator.shutdown();
    
    return json({
      success: true,
      message: 'Orchestrator shutdown initiated',
      timestamp: new Date().toISOString()
    });

  } catch (err: any) {
    console.error('[Orchestrator API] DELETE error:', err);
    throw error(500, `Shutdown failed: ${err}`);
  }
};