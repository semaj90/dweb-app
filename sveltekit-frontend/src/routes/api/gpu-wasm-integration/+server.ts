import type { RequestHandler } from './$types.js';

/**
 * Unified GPU/WASM Integration API Endpoint
 * Provides centralized access to GPU services and WASM bridge
 * Integrates FlashAttention2, LLVM-WASM bridge, and GPU error processor
 */

import { gpuServiceIntegration, GPUServiceUtils, type GPUProcessingTask, type GPUServiceStatus } from '$lib/services/gpu-service-integration';
import { llvmWasmBridge, initializeLLVMIntegration, type LLVMWASMBridge } from '$lib/wasm/llvm-wasm-bridge';
import { flashAttention2Service, gpuErrorProcessor, type GPUErrorContext } from '$lib/services/flashattention2-rtx3060';
import { URL } from "url";

export interface IntegrationStatus {
  gpuService: {
    available: boolean;
    initialized: boolean;
    status: GPUServiceStatus;
  };
  wasmBridge: {
    available: boolean;
    initialized: boolean;
    modules: Record<string, any>;
  };
  flashAttention: {
    available: boolean;
    status: any;
  };
  errorProcessor: {
    available: boolean;
    cacheStats: any;
  };
  overall: {
    healthy: boolean;
    readyForProcessing: boolean;
    integrationScore: number;
  };
}

// Initialize services on startup
let initializationPromise: Promise<void> | null = null;

function getInitializationPromise(): Promise<void> {
  if (!initializationPromise) {
    initializationPromise = initializeServices();
  }
  return initializationPromise;
}

async function initializeServices(): Promise<void> {
  try {
    console.log('🚀 Initializing GPU/WASM integration services...');
    
    // Initialize all services concurrently
    await Promise.all([
      gpuServiceIntegration.initialize(),
      initializeLLVMIntegration(),
      flashAttention2Service.initialize()
    ]);
    
    console.log('✅ GPU/WASM integration services initialized');
  } catch (error: any) {
    console.error('❌ Service initialization failed:', error);
    throw error;
  }
}

export const GET: RequestHandler = async ({ url }) => {
  const action = url.searchParams.get('action') || 'status';
  
  try {
    // Ensure services are initialized
    await getInitializationPromise();
    
    switch (action) {
      case 'status':
        return json(await getIntegrationStatus());
        
      case 'health':
        return json(await getHealthCheck());
        
      case 'modules':
        return json(await getModuleInformation());
        
      case 'metrics':
        return json(await getPerformanceMetrics());
        
      default:
        return json({ error: 'Invalid action parameter' }, { status: 400 });
    }
  } catch (error: any) {
    console.error('GPU/WASM Integration API error:', error);
    return json({
      error: 'Service unavailable',
      details: error instanceof Error ? error.message : String(error),
      timestamp: new Date().toISOString()
    }, { status: 503 });
  }
};

export const POST: RequestHandler = async ({ request, url }) => {
  const action = url.searchParams.get('action') || 'process';
  
  try {
    // Ensure services are initialized
    await getInitializationPromise();
    
    const body = await request.json().catch(() => ({}));
    
    switch (action) {
      case 'process':
        return await handleProcessing(body);
        
      case 'legal-analysis':
        return await handleLegalAnalysis(body);
        
      case 'embedding':
        return await handleEmbeddingGeneration(body);
        
      case 'error-processing':
        return await handleErrorProcessing(body);
        
      case 'compile-wasm':
        return await handleWASMCompilation(body);
        
      case 'test':
        return await handleIntegrationTest(body);
        
      default:
        return json({ error: 'Invalid action parameter' }, { status: 400 });
    }
  } catch (error: any) {
    console.error('GPU/WASM Integration API error:', error);
    return json({
      error: 'Processing failed',
      details: error instanceof Error ? error.message : String(error),
      timestamp: new Date().toISOString()
    }, { status: 500 });
  }
};

async function getIntegrationStatus(): Promise<IntegrationStatus> {
  try {
    // Get status from all services
    const [gpuStatus, wasmModules, flashStatus, errorStats] = await Promise.all([
      gpuServiceIntegration.getStatus(),
      llvmWasmBridge.getModuleStats(),
      flashAttention2Service.getStatus(),
      gpuErrorProcessor.getCacheStats()
    ]);
    
    const integrationScore = calculateIntegrationScore({
      gpuAvailable: gpuStatus.available,
      gpuInitialized: gpuStatus.initialized,
      wasmModulesLoaded: Object.keys(wasmModules).length,
      flashAttentionReady: flashStatus.initialized,
      errorProcessorReady: errorStats.cacheSize >= 0
    });
    
    return {
      gpuService: {
        available: gpuStatus.available,
        initialized: gpuStatus.initialized,
        status: gpuStatus
      },
      wasmBridge: {
        available: Object.keys(wasmModules).length > 0,
        initialized: Object.values(wasmModules).some((module: any) => module.isLoaded),
        modules: wasmModules
      },
      flashAttention: {
        available: flashStatus.initialized,
        status: flashStatus
      },
      errorProcessor: {
        available: true,
        cacheStats: errorStats
      },
      overall: {
        healthy: integrationScore > 0.7,
        readyForProcessing: integrationScore > 0.5,
        integrationScore
      }
    };
  } catch (error: any) {
    console.error('❌ Failed to get integration status:', error);
    throw error;
  }
}

function calculateIntegrationScore(metrics: {
  gpuAvailable: boolean;
  gpuInitialized: boolean;
  wasmModulesLoaded: number;
  flashAttentionReady: boolean;
  errorProcessorReady: boolean;
}): number {
  let score = 0;
  const maxScore = 5;
  
  if (metrics.gpuAvailable) score += 1;
  if (metrics.gpuInitialized) score += 1;
  if (metrics.wasmModulesLoaded > 0) score += Math.min(1, metrics.wasmModulesLoaded / 4);
  if (metrics.flashAttentionReady) score += 1;
  if (metrics.errorProcessorReady) score += 1;
  
  return score / maxScore;
}

async function getHealthCheck(): Promise<any> {
  const status = await getIntegrationStatus();
  
  return {
    status: status.overall.healthy ? 'healthy' : 'degraded',
    services: {
      gpu: status.gpuService.available ? 'operational' : 'unavailable',
      wasm: status.wasmBridge.available ? 'operational' : 'unavailable',
      flashAttention: status.flashAttention.available ? 'operational' : 'unavailable',
      errorProcessor: status.errorProcessor.available ? 'operational' : 'unavailable'
    },
    readyForProcessing: status.overall.readyForProcessing,
    integrationScore: status.overall.integrationScore,
    timestamp: new Date().toISOString()
  };
}

async function getModuleInformation(): Promise<any> {
  const wasmModules = llvmWasmBridge.getModuleStats();
  const gpuStatus = await gpuServiceIntegration.getStatus();
  
  return {
    wasmModules,
    gpuServiceConfig: {
      batchSize: gpuStatus.queuedTasks,
      memoryUsage: gpuStatus.memoryUsage,
      performance: gpuStatus.performance
    },
    availableOperations: [
      'legal-text-processing',
      'vector-embedding-generation',
      'similarity-calculation',
      'error-processing',
      'wasm-compilation'
    ]
  };
}

async function getPerformanceMetrics(): Promise<any> {
  const status = await getIntegrationStatus();
  
  return {
    gpu: {
      utilization: status.gpuService.status.currentLoad,
      throughput: status.gpuService.status.performance.throughput,
      avgProcessingTime: status.gpuService.status.performance.avgProcessingTime,
      efficiency: status.gpuService.status.performance.efficiency,
      errorRate: status.gpuService.status.errorRate
    },
    wasm: {
      modulesLoaded: Object.keys(status.wasmBridge.modules).length,
      memoryUsage: Object.values(status.wasmBridge.modules)
        .reduce((sum: number, module: any) => sum + (module.performance?.memoryUsage || 0), 0),
      avgCompileTime: Object.values(status.wasmBridge.modules)
        .reduce((sum: number, module: any) => sum + (module.performance?.compileTimeMs || 0), 0) / 
        Math.max(1, Object.keys(status.wasmBridge.modules).length)
    },
    flashAttention: {
      gpuEnabled: status.flashAttention.status.gpuEnabled,
      memoryOptimization: status.flashAttention.status.memoryOptimization,
      maxSequenceLength: status.flashAttention.status.maxSequenceLength
    },
    errorProcessor: status.errorProcessor.cacheStats,
    timestamp: new Date().toISOString()
  };
}

async function handleProcessing(body: any): Promise<any> {
  const { type, data, priority = 'medium', metadata = {} } = body;
  
  if (!type || !data) {
    return json({ error: 'Missing type or data parameters' }, { status: 400 });
  }
  
  const taskId = await gpuServiceIntegration.submitTask({
    type,
    data,
    priority,
    metadata: {
      ...metadata,
      requestTime: new Date().toISOString(),
      source: 'gpu-wasm-integration-api'
    }
  });
  
  return json({
    success: true,
    taskId,
    status: 'queued',
    message: `Task ${taskId} submitted for ${type} processing`,
    timestamp: new Date().toISOString()
  });
}

async function handleLegalAnalysis(body: any): Promise<any> {
  const { text, context = [], analysisType = 'legal' } = body;
  
  if (!text || typeof text !== 'string') {
    return json({ error: 'Missing or invalid text parameter' }, { status: 400 });
  }
  
  try {
    // Use both GPU service and WASM bridge for comprehensive analysis
    const [gpuResult, wasmResult] = await Promise.allSettled([
      gpuServiceIntegration.processLegalText(text, context, analysisType),
      llvmWasmBridge.processLegalText(text, {
        extractCitations: true,
        analyzePrecedents: true,
        riskAssessment: false
      })
    ]);
    
    const response: any = {
      success: true,
      analysis: {},
      processingTime: 0,
      sources: []
    };
    
    // Combine results from GPU processing
    if (gpuResult.status === 'fulfilled') {
      response.analysis.gpu = gpuResult.value;
      response.processingTime += gpuResult.value.processingTime || 0;
      response.sources.push('gpu-flash-attention');
    }
    
    // Combine results from WASM processing
    if (wasmResult.status === 'fulfilled') {
      response.analysis.wasm = wasmResult.value;
      response.processingTime += wasmResult.value.processingTime || 0;
      response.sources.push('llvm-wasm-bridge');
    }
    
    // Create unified result
    response.analysis.unified = {
      confidence: Math.max(
        response.analysis.gpu?.legalAnalysis?.confidenceMetrics?.semantic || 0,
        0.7 // Base confidence from WASM processing
      ),
      citations: response.analysis.wasm?.citations || [],
      legalEntities: response.analysis.gpu?.legalAnalysis?.legalEntities || [],
      conceptClusters: response.analysis.gpu?.legalAnalysis?.conceptClusters || [],
      processedText: response.analysis.wasm?.processedText || text
    };
    
    return json(response);
  } catch (error: any) {
    return json({
      success: false,
      error: 'Legal analysis failed',
      details: error instanceof Error ? error.message : String(error)
    }, { status: 500 });
  }
}

async function handleEmbeddingGeneration(body: any): Promise<any> {
  const { texts, dimensions = 384 } = body;
  
  if (!texts || !Array.isArray(texts)) {
    return json({ error: 'Missing or invalid texts parameter (must be array)' }, { status: 400 });
  }
  
  try {
    // Try GPU service first, fallback to WASM
    let embeddings: Float32Array[] = [];
    let processingTime = 0;
    let source = '';
    
    try {
      const startTime = performance.now();
      embeddings = await gpuServiceIntegration.generateEmbeddings(texts);
      processingTime = performance.now() - startTime;
      source = 'gpu-service';
    } catch (gpuError) {
      console.warn('GPU embedding failed, trying WASM:', gpuError);
      
      // Fallback to WASM processing
      const results = await Promise.all(
        texts.map(async (text: string) => {
          const result = await llvmWasmBridge.computeEmbedding(
            text.split('').map(char => char.charCodeAt(0)),
            dimensions
          );
          processingTime += result.processingTime;
          return new Float32Array(result.embedding);
        })
      );
      
      embeddings = results;
      source = 'wasm-bridge';
    }
    
    return json({
      success: true,
      embeddings: embeddings.map(emb => Array.from(emb)),
      dimensions,
      processingTime,
      source,
      timestamp: new Date().toISOString()
    });
  } catch (error: any) {
    return json({
      success: false,
      error: 'Embedding generation failed',
      details: error instanceof Error ? error.message : String(error)
    }, { status: 500 });
  }
}

async function handleErrorProcessing(body: any): Promise<any> {
  const { errorContext } = body;
  
  if (!errorContext) {
    return json({ error: 'Missing errorContext parameter' }, { status: 400 });
  }
  
  try {
    // Process error using GPU error processor
    const result = await gpuServiceIntegration.processGPUError(errorContext);
    
    return json({
      success: result.resolved,
      result,
      timestamp: new Date().toISOString()
    });
  } catch (error: any) {
    return json({
      success: false,
      error: 'Error processing failed',
      details: error instanceof Error ? error.message : String(error)
    }, { status: 500 });
  }
}

async function handleWASMCompilation(body: any): Promise<any> {
  const { moduleId, sources, options = {} } = body;
  
  if (!moduleId || !sources) {
    return json({ error: 'Missing moduleId or sources parameters' }, { status: 400 });
  }
  
  try {
    // Compile WASM module using LLVM bridge
    const module = await llvmWasmBridge.compileLegalModule(
      moduleId,
      `custom_${moduleId}`,
      {
        sources,
        exports: options.exports || [],
        memoryRequired: options.memoryRequired || 1024 * 1024
      }
    );
    
    if (!module) {
      throw new Error('Module compilation failed');
    }
    
    return json({
      success: true,
      module: {
        id: module.id,
        name: module.name,
        isLoaded: module.isLoaded,
        exports: Object.keys(module.exports),
        performance: module.performance
      },
      timestamp: new Date().toISOString()
    });
  } catch (error: any) {
    return json({
      success: false,
      error: 'WASM compilation failed',
      details: error instanceof Error ? error.message : String(error)
    }, { status: 500 });
  }
}

async function handleIntegrationTest(body: any): Promise<any> {
  const { testType = 'comprehensive' } = body;
  
  try {
    const results: any = {
      testType,
      timestamp: new Date().toISOString(),
      tests: {}
    };
    
    // Test GPU service integration
    try {
      const gpuTest = await GPUServiceUtils.testIntegration();
      results.tests.gpuService = gpuTest;
    } catch (error: any) {
      results.tests.gpuService = {
        success: false,
        details: error instanceof Error ? error.message : String(error)
      };
    }
    
    // Test WASM bridge
    try {
      const testText = 'This is a sample legal document for testing purposes.';
      const wasmTest = await llvmWasmBridge.processLegalText(testText, {
        extractCitations: true
      });
      results.tests.wasmBridge = {
        success: true,
        details: `Processed ${testText.length} characters in ${wasmTest.processingTime.toFixed(2)}ms`
      };
    } catch (error: any) {
      results.tests.wasmBridge = {
        success: false,
        details: error instanceof Error ? error.message : String(error)
      };
    }
    
    // Test FlashAttention2
    try {
      const flashStatus = flashAttention2Service.getStatus();
      results.tests.flashAttention = {
        success: flashStatus.initialized,
        details: `GPU enabled: ${flashStatus.gpuEnabled}, Memory optimization: ${flashStatus.memoryOptimization}`
      };
    } catch (error: any) {
      results.tests.flashAttention = {
        success: false,
        details: error instanceof Error ? error.message : String(error)
      };
    }
    
    // Calculate overall success
    const testResults = Object.values(results.tests);
    const successfulTests = testResults.filter((test: any) => test.success).length;
    results.overall = {
      success: successfulTests === testResults.length,
      successRate: successfulTests / testResults.length,
      summary: `${successfulTests}/${testResults.length} tests passed`
    };
    
    return json(results);
  } catch (error: any) {
    return json({
      success: false,
      error: 'Integration test failed',
      details: error instanceof Error ? error.message : String(error),
      timestamp: new Date().toISOString()
    }, { status: 500 });
  }
}
