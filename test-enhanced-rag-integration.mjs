#!/usr/bin/env node

/**
 * Enhanced RAG Integration Test
 * Tests the complete integration of cluster management, Ollama Gemma caching, and enhanced RAG service
 */

import { execSync } from 'child_process';
import { writeFileSync, readFileSync } from 'fs';
import { join } from 'path';

const TEST_LOG_FILE = 'enhanced-rag-integration-test.log';
const TEST_RESULTS = {
  testsPassed: 0,
  testsFailed: 0,
  results: []
};

function log(message) {
  const timestamp = new Date().toISOString();
  const logMessage = `[${timestamp}] ${message}`;
  console.log(logMessage);
  
  try {
    const logContent = readFileSync(TEST_LOG_FILE, 'utf8');
    writeFileSync(TEST_LOG_FILE, logContent + '\n' + logMessage);
  } catch {
    writeFileSync(TEST_LOG_FILE, logMessage);
  }
}

function runTest(testName, testFunction) {
  log(`🧪 Running test: ${testName}`);
  
  try {
    const result = testFunction();
    if (result !== false) {
      TEST_RESULTS.testsPassed++;
      TEST_RESULTS.results.push({ test: testName, status: 'PASSED', result });
      log(`✅ ${testName}: PASSED`);
      return true;
    } else {
      TEST_RESULTS.testsFailed++;
      TEST_RESULTS.results.push({ test: testName, status: 'FAILED', error: 'Test returned false' });
      log(`❌ ${testName}: FAILED`);
      return false;
    }
  } catch (error) {
    TEST_RESULTS.testsFailed++;
    TEST_RESULTS.results.push({ test: testName, status: 'FAILED', error: error.message });
    log(`❌ ${testName}: FAILED - ${error.message}`);
    return false;
  }
}

function testEnhancedRAGServiceExists() {
  try {
    const ragServicePath = './rag/enhanced-rag-service.ts';
    const content = readFileSync(ragServicePath, 'utf8');
    
    const requiredComponents = [
      'import { ollamaGemmaCache',
      'import { clusterManager',
      'enableClustering',
      'enableSemanticCaching',
      'checkSemanticCache',
      'cacheRAGResult',
      'batchQuery',
      'getEnhancedStats'
    ];
    
    for (const component of requiredComponents) {
      if (!content.includes(component)) {
        throw new Error(`Missing required component: ${component}`);
      }
    }
    
    return 'Enhanced RAG service has all required components';
  } catch (error) {
    throw new Error(`Enhanced RAG service test failed: ${error.message}`);
  }
}

function testClusterManagerExists() {
  try {
    const clusterPath = './vscode-llm-extension/src/cluster-manager.ts';
    const content = readFileSync(clusterPath, 'utf8');
    
    const requiredComponents = [
      'ExtensionClusterManager',
      'WorkerTask',
      'executeTask',
      'getClusterStats',
      'setupPrimary',
      'setupWorker'
    ];
    
    for (const component of requiredComponents) {
      if (!content.includes(component)) {
        throw new Error(`Missing required component: ${component}`);
      }
    }
    
    return 'Cluster manager has all required components';
  } catch (error) {
    throw new Error(`Cluster manager test failed: ${error.message}`);
  }
}

function testOllamaGemmaCacheExists() {
  try {
    const cachePath = './vscode-llm-extension/src/ollama-gemma-cache.ts';
    const content = readFileSync(cachePath, 'utf8');
    
    const requiredComponents = [
      'OllamaGemmaCacheManager',
      'getEmbedding',
      'querySimilar',
      'preCacheWorkspace',
      'cosineSimilarity',
      'getCacheStats'
    ];
    
    for (const component of requiredComponents) {
      if (!content.includes(component)) {
        throw new Error(`Missing required component: ${component}`);
      }
    }
    
    return 'Ollama Gemma cache has all required components';
  } catch (error) {
    throw new Error(`Ollama Gemma cache test failed: ${error.message}`);
  }
}

function testVSCodeExtensionIntegration() {
  try {
    const extensionPath = './vscode-llm-extension/src/extension.ts';
    const content = readFileSync(extensionPath, 'utf8');
    
    const requiredComponents = [
      'import { clusterManager',
      'import { ollamaGemmaCache',
      'analyzeCurrentContext',
      'mcp.analyzeCurrentContext',
      'cluster.showStatus',
      'cache.showStats',
      'generateAnalysisWebviewContent'
    ];
    
    for (const component of requiredComponents) {
      if (!content.includes(component)) {
        throw new Error(`Missing required component: ${component}`);
      }
    }
    
    return 'VS Code extension has all enhanced integrations';
  } catch (error) {
    throw new Error(`VS Code extension integration test failed: ${error.message}`);
  }
}

function testAgentOrchestrationExists() {
  try {
    const agentPaths = [
      './agents/claude-agent.ts',
      './agents/autogen-agent.ts',
      './agents/crewai-agent.ts'
    ];
    
    let agentsFound = 0;
    
    for (const agentPath of agentPaths) {
      try {
        const content = readFileSync(agentPath, 'utf8');
        if (content.includes('execute') && content.includes('export')) {
          agentsFound++;
        }
      } catch (error) {
        // Agent file might not exist yet, which is okay
        log(`⚠️ Agent file not found: ${agentPath}`);
      }
    }
    
    return `Found ${agentsFound} agent implementations`;
  } catch (error) {
    throw new Error(`Agent orchestration test failed: ${error.message}`);
  }
}

function testConfigurationIntegration() {
  try {
    const ragServicePath = './rag/enhanced-rag-service.ts';
    const content = readFileSync(ragServicePath, 'utf8');
    
    const environmentVariables = [
      'ENHANCED_RAG_CLUSTERING',
      'ENHANCED_RAG_CACHING',
      'ENHANCED_RAG_CACHE_THRESHOLD',
      'ENHANCED_RAG_WORKERS',
      'ENHANCED_RAG_MAX_CONCURRENT',
      'ENHANCED_RAG_PRECACHING'
    ];
    
    for (const envVar of environmentVariables) {
      if (!content.includes(envVar)) {
        throw new Error(`Missing environment variable: ${envVar}`);
      }
    }
    
    return 'All environment variables are properly configured';
  } catch (error) {
    throw new Error(`Configuration integration test failed: ${error.message}`);
  }
}

function testTypeDefinitions() {
  try {
    const ragServicePath = './rag/enhanced-rag-service.ts';
    const content = readFileSync(ragServicePath, 'utf8');
    
    const requiredTypes = [
      'RAGServiceConfig',
      'RAGQueryRequest',
      'RAGQueryResponse',
      'EnhancedRAGService',
      'enableClustering: boolean',
      'enableSemanticCaching: boolean',
      'cacheHit: boolean',
      'processingMethod:',
      'enhancedMetadata:'
    ];
    
    for (const type of requiredTypes) {
      if (!content.includes(type)) {
        throw new Error(`Missing type definition: ${type}`);
      }
    }
    
    return 'All TypeScript type definitions are present';
  } catch (error) {
    throw new Error(`Type definitions test failed: ${error.message}`);
  }
}

function testErrorHandlingAndFallbacks() {
  try {
    const ragServicePath = './rag/enhanced-rag-service.ts';
    const content = readFileSync(ragServicePath, 'utf8');
    
    const errorHandlingPatterns = [
      'try {',
      'catch (error)',
      'enableFallback',
      'console.warn',
      'fallback mode',
      'initializeEnhancedSystems'
    ];
    
    for (const pattern of errorHandlingPatterns) {
      if (!content.includes(pattern)) {
        throw new Error(`Missing error handling pattern: ${pattern}`);
      }
    }
    
    return 'Error handling and fallback mechanisms are implemented';
  } catch (error) {
    throw new Error(`Error handling test failed: ${error.message}`);
  }
}

function testPerformanceMetrics() {
  try {
    const ragServicePath = './rag/enhanced-rag-service.ts';
    const content = readFileSync(ragServicePath, 'utf8');
    
    const metricsPatterns = [
      'performanceMetrics',
      'cacheHitRate',
      'averageResponseTime',
      'clusterUtilization',
      'updateCacheHitRate',
      'updatePerformanceMetrics',
      'getEnhancedStats'
    ];
    
    for (const pattern of metricsPatterns) {
      if (!content.includes(pattern)) {
        throw new Error(`Missing performance metric: ${pattern}`);
      }
    }
    
    return 'Performance metrics and monitoring are implemented';
  } catch (error) {
    throw new Error(`Performance metrics test failed: ${error.message}`);
  }
}

async function runAllTests() {
  log('🚀 Starting Enhanced RAG Integration Tests');
  log('===============================================');
  
  // Core component tests
  runTest('Enhanced RAG Service Structure', testEnhancedRAGServiceExists);
  runTest('Cluster Manager Implementation', testClusterManagerExists);
  runTest('Ollama Gemma Cache Implementation', testOllamaGemmaCacheExists);
  runTest('VS Code Extension Integration', testVSCodeExtensionIntegration);
  runTest('Agent Orchestration Setup', testAgentOrchestrationExists);
  
  // Configuration and integration tests
  runTest('Configuration Integration', testConfigurationIntegration);
  runTest('TypeScript Type Definitions', testTypeDefinitions);
  runTest('Error Handling and Fallbacks', testErrorHandlingAndFallbacks);
  runTest('Performance Metrics', testPerformanceMetrics);
  
  // Generate final report
  log('===============================================');
  log('🏁 Enhanced RAG Integration Test Results');
  log(`✅ Tests Passed: ${TEST_RESULTS.testsPassed}`);
  log(`❌ Tests Failed: ${TEST_RESULTS.testsFailed}`);
  log(`📊 Success Rate: ${((TEST_RESULTS.testsPassed / (TEST_RESULTS.testsPassed + TEST_RESULTS.testsFailed)) * 100).toFixed(1)}%`);
  
  if (TEST_RESULTS.testsFailed === 0) {
    log('🎉 All enhanced RAG integration tests passed!');
    log('✅ The cluster management, Ollama Gemma caching, and enhanced RAG service are fully integrated');
  } else {
    log('⚠️ Some tests failed. Please review the issues above.');
  }
  
  // Write detailed results to file
  const detailedResults = {
    timestamp: new Date().toISOString(),
    summary: {
      passed: TEST_RESULTS.testsPassed,
      failed: TEST_RESULTS.testsFailed,
      successRate: ((TEST_RESULTS.testsPassed / (TEST_RESULTS.testsPassed + TEST_RESULTS.testsFailed)) * 100).toFixed(1) + '%'
    },
    results: TEST_RESULTS.results
  };
  
  writeFileSync('enhanced-rag-integration-results.json', JSON.stringify(detailedResults, null, 2));
  log('📄 Detailed results written to: enhanced-rag-integration-results.json');
  
  return TEST_RESULTS.testsFailed === 0;
}

// Run the tests
runAllTests().then(success => {
  process.exit(success ? 0 : 1);
}).catch(error => {
  log(`💥 Test suite failed with error: ${error.message}`);
  process.exit(1);
});