
import type { RequestHandler } from './$types';

// routes/api/ai-synthesizer/test/+server.ts
// Integration test and demo endpoint for AI synthesis system

import { aiAssistantSynthesizer } from "$lib/server/ai/ai-assistant-input-synthesizer";
import { feedbackLoop } from "$lib/server/ai/feedback-loop";
import { streamingService } from "$lib/server/ai/streaming-service";
import { logger } from "$lib/server/logger";

export const GET: RequestHandler = async () => {
  logger.info('[Test] Running AI Synthesizer integration test...');
  
  const results = {
    timestamp: new Date(),
    tests: [],
    health: {},
    performance: {},
    recommendations: []
  };

  try {
    // Test 1: Health Check
    const healthTest = await testHealthCheck();
    results.tests.push(healthTest);
    results.health = healthTest.result;

    // Test 2: Basic Synthesis
    const synthesisTest = await testBasicSynthesis();
    results.tests.push(synthesisTest);

    // Test 3: Caching
    const cacheTest = await testCaching();
    results.tests.push(cacheTest);

    // Test 4: Streaming
    const streamTest = await testStreaming();
    results.tests.push(streamTest);

    // Test 5: Ollama Local LLM
    const ollamaTest = await testOllama();
    results.tests.push(ollamaTest);

    // Test 6: Feedback Loop
    const feedbackTest = await testFeedbackLoop();
    results.tests.push(feedbackTest);

    // Test 7: Monitoring
    const monitoringTest = await testMonitoring();
    results.tests.push(monitoringTest);
    results.performance = monitoringTest.result;

    // Generate recommendations
    results.recommendations = generateRecommendations(results);

    return json({
      success: true,
      ...results
    });

  } catch (error: any) {
    logger.error('[Test] Integration test failed:', error);
    return json({
      success: false,
      error: error.message,
      ...results
    }, { status: 500 });
  }
};

async function testHealthCheck(): Promise<any> {
  const startTime = Date.now();
  
  try {
    const health = await aiAssistantSynthesizer.healthCheck();
    const cacheHealth = await cachingLayer.getStats();
    const ollamaHealth = await ollamaLLM.healthCheck();
    
    return {
      name: 'Health Check',
      status: 'passed',
      duration: Date.now() - startTime,
      result: {
        synthesizer: health,
        cache: cacheHealth,
        ollama: ollamaHealth
      }
    };
  } catch (error: any) {
    return {
      name: 'Health Check',
      status: 'failed',
      duration: Date.now() - startTime,
      error: error.message
    };
  }
}

async function testBasicSynthesis(): Promise<any> {
  const startTime = Date.now();
  
  try {
    const testQuery = 'What are the key elements of a valid contract under common law?';
    
    const result = await aiAssistantSynthesizer.synthesizeInput({
      query: testQuery,
      context: {
        userId: 'test_user',
        sessionId: 'test_session'
      },
      options: {
        enableMMR: true,
        enableCrossEncoder: true,
        enableLegalBERT: true,
        enableRAG: true,
        maxSources: 5
      }
    });
    
    const passed = result && 
                   result.processedQuery && 
                   result.retrievedContext &&
                   result.enhancedPrompt &&
                   result.metadata;
    
    return {
      name: 'Basic Synthesis',
      status: passed ? 'passed' : 'failed',
      duration: Date.now() - startTime,
      result: {
        queryProcessed: !!result.processedQuery,
        sourcesFound: result.retrievedContext?.sources?.length || 0,
        confidence: result.metadata?.confidence || 0,
        qualityScore: result.metadata?.qualityScore || 0
      }
    };
  } catch (error: any) {
    return {
      name: 'Basic Synthesis',
      status: 'failed',
      duration: Date.now() - startTime,
      error: error.message
    };
  }
}

async function testCaching(): Promise<any> {
  const startTime = Date.now();
  
  try {
    const testKey = 'test_key_' + Date.now();
    const testData = { test: 'data', timestamp: Date.now() };
    
    // Test set
    await cachingLayer.set(testKey, testData, { ttl: 60 });
    
    // Test get
    const retrieved = await cachingLayer.get(testKey);
    
    // Test stats
    const stats = await cachingLayer.getStats();
    
    const passed = retrieved && 
                   retrieved.test === testData.test &&
                   stats.hits >= 0;
    
    return {
      name: 'Caching Layer',
      status: passed ? 'passed' : 'failed',
      duration: Date.now() - startTime,
      result: {
        cacheWorking: passed,
        hitRate: stats.hitRate,
        memoryUsage: stats.memoryUsage,
        redisConnected: stats.redisConnected
      }
    };
  } catch (error: any) {
    return {
      name: 'Caching Layer',
      status: 'failed',
      duration: Date.now() - startTime,
      error: error.message
    };
  }
}

async function testStreaming(): Promise<any> {
  const startTime = Date.now();
  
  try {
    let progressUpdates = 0;
    let stagesCompleted = [];
    
    const result = await streamingService.synthesizeWithProgress({
      input: {
        query: 'Test streaming query',
        context: { userId: 'test' },
        options: {}
      },
      onProgress: (stage, progress) => {
        progressUpdates++;
      },
      onStage: (stage, data) => {
        stagesCompleted.push(stage);
      }
    });
    
    const passed = progressUpdates > 0 && stagesCompleted.length > 0;
    
    return {
      name: 'Streaming Service',
      status: passed ? 'passed' : 'failed',
      duration: Date.now() - startTime,
      result: {
        progressUpdates,
        stagesCompleted,
        activeStreams: streamingService.getActiveStreams().length
      }
    };
  } catch (error: any) {
    return {
      name: 'Streaming Service',
      status: 'failed',
      duration: Date.now() - startTime,
      error: error.message
    };
  }
}

async function testOllama(): Promise<any> {
  const startTime = Date.now();
  
  try {
    const available = await ollamaLLM.checkAvailability();
    
    let generateResult = null;
    let embeddingResult = null;
    
    if (available) {
      // Test generation
      generateResult = await ollamaLLM.generate({
        model: 'llama2',
        prompt: 'What is a contract?',
        options: {
          temperature: 0.3,
          num_predict: 50
        }
      });
      
      // Test embeddings
      embeddingResult = await ollamaLLM.generateEmbeddings('test text');
    }
    
    return {
      name: 'Ollama Local LLM',
      status: available ? 'passed' : 'warning',
      duration: Date.now() - startTime,
      result: {
        available,
        models: available ? (await ollamaLLM.healthCheck()).models : [],
        generationWorks: !!generateResult,
        embeddingsWork: !!embeddingResult
      }
    };
  } catch (error: any) {
    return {
      name: 'Ollama Local LLM',
      status: 'failed',
      duration: Date.now() - startTime,
      error: error.message
    };
  }
}

async function testFeedbackLoop(): Promise<any> {
  const startTime = Date.now();
  
  try {
    // Record test interaction
    await feedbackLoop.recordInteraction({
      requestId: 'test_request_' + Date.now(),
      query: 'Test query',
      result: { metadata: { confidence: 0.8 } },
      userId: 'test_user',
      timestamp: new Date()
    });
    
    // Process test feedback
    await feedbackLoop.processFeedback({
      requestId: 'test_request_' + Date.now(),
      userId: 'test_user',
      rating: 4,
      feedback: 'Test feedback'
    });
    
    // Get personalized recommendations
    const recommendations = await feedbackLoop.getPersonalizedRecommendations('test_user');
    
    // Get stats
    const stats = feedbackLoop.getStats();
    
    return {
      name: 'Feedback Loop',
      status: 'passed',
      duration: Date.now() - startTime,
      result: {
        interactionCount: stats.interactionCount,
        queueSize: stats.queueSize,
        hasRecommendations: !!recommendations,
        modelWeights: stats.modelWeights
      }
    };
  } catch (error: any) {
    return {
      name: 'Feedback Loop',
      status: 'failed',
      duration: Date.now() - startTime,
      error: error.message
    };
  }
}

async function testMonitoring(): Promise<any> {
  const startTime = Date.now();
  
  try {
    // Track test request
    monitoringService.trackRequest({
      requestId: 'test_' + Date.now(),
      userId: 'test_user',
      query: 'Test query',
      timestamp: new Date()
    });
    
    // Track test metrics
    monitoringService.trackMetrics({
      requestId: 'test_' + Date.now(),
      processingTime: 1234,
      confidence: 0.85,
      sourceCount: 5,
      strategies: ['rag', 'mmr'],
      qualityScore: 0.9
    });
    
    // Get stats
    const stats = monitoringService.getStats();
    
    // Export Prometheus metrics
    const prometheusMetrics = monitoringService.exportPrometheusMetrics();
    
    return {
      name: 'Monitoring Service',
      status: 'passed',
      duration: Date.now() - startTime,
      result: {
        totalRequests: stats.counters.totalRequests,
        successRate: stats.rates.successRate,
        cacheHitRate: stats.rates.cacheHitRate,
        performance: stats.performance,
        hasPrometheusMetrics: prometheusMetrics.length > 0
      }
    };
  } catch (error: any) {
    return {
      name: 'Monitoring Service',
      status: 'failed',
      duration: Date.now() - startTime,
      error: error.message
    };
  }
}

function generateRecommendations(results) {
  const recommendations = [];
  
  // Check Ollama availability
  const ollamaTest = results.tests.find(t => t.name === 'Ollama Local LLM');
  if (ollamaTest && !ollamaTest.result?.available) {
    recommendations.push({
      priority: 'high',
      category: 'infrastructure',
      message: 'Install and run Ollama for local LLM support',
      action: 'Run: curl -fsSL https://ollama.ai/install.sh | sh && ollama serve'
    });
  }
  
  // Check Redis connection
  const cacheTest = results.tests.find(t => t.name === 'Caching Layer');
  if (cacheTest && !cacheTest.result?.redisConnected) {
    recommendations.push({
      priority: 'medium',
      category: 'performance',
      message: 'Connect Redis for distributed caching',
      action: 'Install Redis and set REDIS_HOST environment variable'
    });
  }
  
  // Check cache hit rate
  if (cacheTest && parseFloat(cacheTest.result?.hitRate) < 0.3) {
    recommendations.push({
      priority: 'low',
      category: 'optimization',
      message: 'Low cache hit rate detected',
      action: 'Consider warming cache with frequently accessed data'
    });
  }
  
  // Check performance metrics
  const monitoringTest = results.tests.find(t => t.name === 'Monitoring Service');
  if (monitoringTest?.result?.performance?.overall?.p95 > 5000) {
    recommendations.push({
      priority: 'high',
      category: 'performance',
      message: 'High P95 latency detected',
      action: 'Optimize slow queries and consider adding more resources'
    });
  }
  
  // Check synthesis quality
  const synthesisTest = results.tests.find(t => t.name === 'Basic Synthesis');
  if (synthesisTest?.result?.confidence < 0.7) {
    recommendations.push({
      priority: 'medium',
      category: 'quality',
      message: 'Low confidence scores in synthesis',
      action: 'Improve data quality and consider adding more training data'
    });
  }
  
  return recommendations;
}

// POST endpoint for manual testing with custom queries
export const POST: RequestHandler = async ({ request }) => {
  try {
    const { query, options } = await request.json();
    
    if (!query) {
      return json({ error: 'Query is required' }, { status: 400 });
    }
    
    logger.info(`[Test] Running manual test with query: "${query}"`);
    
    const result = await aiAssistantSynthesizer.synthesizeInput({
      query,
      context: {
        userId: 'test_user',
        sessionId: 'test_session_' + Date.now()
      },
      options: {
        enableMMR: true,
        enableCrossEncoder: true,
        enableLegalBERT: true,
        enableRAG: true,
        maxSources: 10,
        ...options
      }
    });
    
    // Also test with Ollama if available
    let ollamaResult = null;
    if (await ollamaLLM.checkAvailability()) {
      ollamaResult = await ollamaLLM.processLegalDocument(
        query,
        'analyze',
        { format: 'json' }
      );
    }
    
    return json({
      success: true,
      synthesis: result,
      ollama: ollamaResult,
      stats: {
        cache: await cachingLayer.getStats(),
        monitoring: monitoringService.getStats(),
        feedback: feedbackLoop.getStats()
      }
    });
    
  } catch (error: any) {
    logger.error('[Test] Manual test failed:', error);
    return json({
      success: false,
      error: error.message
    }, { status: 500 });
  }
};
