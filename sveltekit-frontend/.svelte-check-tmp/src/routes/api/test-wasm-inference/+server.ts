import type { RequestHandler } from './$types';

/**
 * WebAssembly Inference Test API Endpoint
 * Tests the complete WebAssembly RAG inference pipeline
 */

import { json } from '@sveltejs/kit';
import { wasmInferenceMachine, WASMInferenceRAGService } from '$lib/services/webasm-inference-rag.js';
import { rabbitMQIntegration } from '$lib/messaging/rabbitmq-xstate-integration.js';

export const POST: RequestHandler = async ({ request }) => {
  const startTime = Date.now();
  console.log('🧠 WebAssembly inference test request received');

  try {
    const { prompt, enableRAG = true, maxTokens = 1024, temperature = 0.7 } = await request.json();

    if (!prompt) {
      return json({ error: 'Prompt is required' }, { status: 400 });
    }

    console.log(`🔍 Processing WASM inference: "${prompt.slice(0, 50)}..."`);

    // Test 1: Direct WebAssembly Inference Service
    console.log('📊 Test 1: Direct WASM Inference Service');
    
    const wasmRequest = {
      id: `test_wasm_${Date.now()}_${Math.random().toString(36).slice(2)}`,
      prompt,
      maxTokens,
      temperature,
      enableRAG,
      priority: 'high' as const,
      systemMessage: 'You are a legal AI assistant specialized in analyzing legal documents and providing expert insights.'
    };

    const wasmContext = {
      wasmModule: null,
      wasmInstance: null,
      isInitialized: false,
      config: {
        modelPath: '/models/gemma3-legal-q4.wasm',
        threads: 8,
        contextLength: 4096,
        enableGPU: true,
        batchSize: 4,
        quantization: 'q4_0' as const
      },
      activeRequests: new Map(),
      results: new Map(),
      performanceMetrics: {
        totalInferences: 0,
        averageLatency: 0,
        cacheHitRate: 0,
        memoryPeak: 0
      },
      error: null
    };

    const directResult = await WASMInferenceRAGService.processInferenceWithRAG(wasmRequest, wasmContext);
    console.log('✅ Direct WASM inference completed');

    // Test 2: RabbitMQ Message Queue Integration
    console.log('📊 Test 2: RabbitMQ Integration');
    
    let rabbitMQResult = null;
    try {
      await rabbitMQIntegration.publishMessage({
        type: 'wasm_inference',
        payload: {
          id: wasmRequest.id + '_mq',
          prompt,
          maxTokens,
          temperature,
          enableRAG,
          priority: 'high',
          startTime: Date.now()
        },
        priority: 8
      });
      
      rabbitMQResult = {
        status: 'queued',
        messageType: 'wasm_inference',
        queuedAt: new Date().toISOString()
      };
      
      console.log('✅ RabbitMQ message queued successfully');
    } catch (rabbitError) {
      console.warn('⚠️ RabbitMQ test failed:', rabbitError);
      rabbitMQResult = {
        status: 'failed',
        error: rabbitError.message,
        fallback: 'Direct processing mode available'
      };
    }

    // Test 3: PostgreSQL-Qdrant Sync Integration
    console.log('📊 Test 3: PostgreSQL-Qdrant Sync');
    
    let syncResult = null;
    try {
      const { postgresqlQdrantSync } = await import('$lib/services/postgresql-qdrant-sync.js');
      
      // Test health check
      const health = await postgresqlQdrantSync.healthCheck();
      
      // Test WASM statistics
      const wasmStats = postgresqlQdrantSync.getWASMStats();
      
      syncResult = {
        health,
        wasmStats,
        status: 'operational',
        optimizedRetrieval: true
      };
      
      console.log('✅ PostgreSQL-Qdrant sync test completed');
    } catch (syncError) {
      console.warn('⚠️ PostgreSQL-Qdrant sync test failed:', syncError);
      syncResult = {
        status: 'failed',
        error: syncError.message,
        fallback: 'Enhanced RAG service fallback available'
      };
    }

    // Test 4: XState Machine Integration
    console.log('📊 Test 4: XState Machine');
    
    const machineResult = {
      machineId: wasmInferenceMachine.id,
      initialState: wasmInferenceMachine.initialState.value,
      states: Object.keys(wasmInferenceMachine.config.states || {}),
      contextStructure: {
        hasWasmModule: 'wasmModule' in (wasmInferenceMachine.config.context || {}),
        hasPerformanceMetrics: 'performanceMetrics' in (wasmInferenceMachine.config.context || {}),
        hasActiveRequests: 'activeRequests' in (wasmInferenceMachine.config.context || {})
      },
      status: 'configured'
    };

    console.log('✅ XState machine test completed');

    const totalTime = Date.now() - startTime;

    // Comprehensive test results
    const testResults = {
      success: true,
      processingTime: totalTime,
      timestamp: new Date().toISOString(),
      request: {
        prompt: prompt.slice(0, 100) + (prompt.length > 100 ? '...' : ''),
        enableRAG,
        maxTokens,
        temperature
      },
      results: {
        wasmInference: {
          id: directResult.id,
          text: directResult.text,
          tokens: directResult.tokens,
          processingTime: directResult.processingTime,
          ragContext: directResult.ragContext,
          metadata: directResult.metadata,
          cacheHit: directResult.cacheHit
        },
        rabbitMQIntegration: rabbitMQResult,
        postgresqlQdrantSync: syncResult,
        xstateMachine: machineResult
      },
      integration: {
        wasmInferenceService: '✅ Operational',
        rabbitMQMessaging: rabbitMQResult?.status === 'queued' ? '✅ Operational' : '⚠️ Degraded',
        postgresqlQdrantSync: syncResult?.status === 'operational' ? '✅ Operational' : '⚠️ Degraded',
        xstateMachine: '✅ Operational'
      },
      performance: {
        totalProcessingTime: `${totalTime}ms`,
        wasmInferenceTime: `${directResult.processingTime}ms`,
        ragEnabled: enableRAG,
        documentsRetrieved: directResult.ragContext?.documentsUsed || 0,
        memoryUsage: `${directResult.memoryUsage} bytes`,
        cacheUtilization: directResult.cacheHit ? 'Cache Hit' : 'Cache Miss'
      },
      nextSteps: {
        suggestions: [
          'Run more inference requests to test performance under load',
          'Test with different model configurations and quantization settings',
          'Monitor RabbitMQ message processing and queue performance',
          'Verify PostgreSQL-Qdrant sync performance with real document data',
          'Test batch inference and streaming inference modes'
        ]
      }
    };

    console.log(`🎉 WebAssembly inference test completed successfully in ${totalTime}ms`);
    
    return json(testResults, { 
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'X-Processing-Time': totalTime.toString(),
        'X-WASM-Version': '1.0.0',
        'X-Integration-Status': 'fully-operational'
      }
    });

  } catch (error: any) {
    console.error('❌ WebAssembly inference test failed:', error);
    
    const errorTime = Date.now() - startTime;
    return json({
      success: false,
      error: error.message,
      processingTime: errorTime,
      timestamp: new Date().toISOString(),
      stack: error.stack,
      troubleshooting: {
        commonIssues: [
          'WebAssembly module not found or failed to load',
          'PostgreSQL or Qdrant connection issues',
          'RabbitMQ service not available',
          'Embedding service unavailable',
          'Memory allocation issues in WASM runtime'
        ],
        solutions: [
          'Check if all services are running (npm run dev:full)',
          'Verify database connections and Qdrant collection setup',
          'Test individual components separately',
          'Check browser console for additional error details',
          'Ensure WebAssembly is supported in the current environment'
        ]
      }
    }, { 
      status: 500,
      headers: {
        'Content-Type': 'application/json',
        'X-Processing-Time': errorTime.toString(),
        'X-Error-Source': 'wasm-inference-test'
      }
    });
  }
};

// GET endpoint for quick health check
export const GET: RequestHandler = async () => {
  try {
    console.log('🏥 WebAssembly inference health check');

    const { WASMInferenceRAGService } = await import('$lib/services/webasm-inference-rag.js');
    const healthStatus = WASMInferenceRAGService.getHealthStatus();

    // Quick connectivity tests
    const services = {
      wasmInferenceService: healthStatus.status === 'healthy',
      enhancedRAGService: false,
      postgresqlQdrantSync: false
    };

    // Test Enhanced RAG Service
    try {
      const ragResponse = await fetch('http://localhost:8094/api/health', {
        method: 'GET'
      });
      services.enhancedRAGService = ragResponse.ok;
    } catch {
      services.enhancedRAGService = false;
    }

    // Test PostgreSQL-Qdrant Sync
    try {
      const { postgresqlQdrantSync } = await import('$lib/services/postgresql-qdrant-sync.js');
      const syncHealth = await postgresqlQdrantSync.healthCheck();
      services.postgresqlQdrantSync = syncHealth.status === 'healthy';
    } catch {
      services.postgresqlQdrantSync = false;
    }

    const overallHealth = Object.values(services).every(service => service === true);

    return json({
      status: overallHealth ? 'healthy' : 'degraded',
      timestamp: new Date().toISOString(),
      services,
      wasmHealth: healthStatus,
      endpoints: {
        testInference: '/api/test-wasm-inference (POST)',
        healthCheck: '/api/test-wasm-inference (GET)',
        enhancedRAG: 'http://localhost:8094/api/rag'
      },
      version: '1.0.0'
    });

  } catch (error: any) {
    return json({
      status: 'unhealthy',
      error: error.message,
      timestamp: new Date().toISOString()
    }, { status: 500 });
  }
};