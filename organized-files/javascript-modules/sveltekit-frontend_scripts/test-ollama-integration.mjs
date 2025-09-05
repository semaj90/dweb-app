#!/usr/bin/env node

/**
 * Ollama Integration Test with Memory Monitoring
 * Tests local LLM functionality and logs performance metrics
 */

import { readFile, writeFile } from 'fs/promises';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL || 'http://localhost:11434';
const TEST_MODEL = 'gemma2-legal';

/**
 * Memory usage tracking
 */
function getMemoryUsage() {
  const usage = process.memoryUsage();
  return {
    heapUsedMB: Math.round(usage.heapUsed / 1024 / 1024),
    heapTotalMB: Math.round(usage.heapTotal / 1024 / 1024),
    externalMB: Math.round(usage.external / 1024 / 1024),
    rss: Math.round(usage.rss / 1024 / 1024)
  };
}

/**
 * Test Ollama connection and model availability
 */
async function testOllamaHealth() {
  console.log('🔍 Testing Ollama health...');
  
  try {
    const response = await fetch(`${OLLAMA_BASE_URL}/api/tags`);
    
    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.status}`);
    }
    
    const data = await response.json();
    const models = data.models || [];
    
    console.log(`✅ Ollama healthy: ${models.length} models available`);
    console.log(`📋 Models: ${models.map(m => m.name.split(':')[0]).join(', ')}`);
    
    // Check if test model exists
    const hasTestModel = models.some(m => m.name.includes(TEST_MODEL));
    if (!hasTestModel) {
      console.warn(`⚠️  Test model '${TEST_MODEL}' not found. Available models:`);
      models.forEach(m => console.log(`   - ${m.name}`));
    }
    
    return { healthy: true, models, hasTestModel };
    
  } catch (error) {
    console.error('❌ Ollama health check failed:', error.message);
    return { healthy: false, error: error.message };
  }
}

/**
 * Test LLM generation with memory monitoring
 */
async function testLLMGeneration(model = TEST_MODEL) {
  console.log(`🧠 Testing LLM generation with ${model}...`);
  
  const testPrompt = `Please provide a brief legal analysis of contract liability. Keep response under 100 words.`;
  
  const startTime = Date.now();
  const startMemory = getMemoryUsage();
  
  try {
    const response = await fetch(`${OLLAMA_BASE_URL}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        prompt: testPrompt,
        stream: false,
        options: {
          temperature: 0.2,
          num_predict: 150,
          num_ctx: 4096,
          num_gpu: -1 // Use all GPU layers
        }
      })
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Generation failed: ${response.status} - ${errorText}`);
    }
    
    const data = await response.json();
    const endTime = Date.now();
    const endMemory = getMemoryUsage();
    
    const metrics = {
      duration: endTime - startTime,
      tokensGenerated: data.eval_count || 0,
      responseLength: data.response?.length || 0,
      memoryDelta: {
        heapUsed: endMemory.heapUsedMB - startMemory.heapUsedMB,
        heapTotal: endMemory.heapTotalMB - startMemory.heapTotalMB
      },
      startMemory,
      endMemory
    };
    
    console.log('✅ LLM generation successful');
    console.log(`⏱️  Duration: ${metrics.duration}ms`);
    console.log(`🎯 Tokens: ${metrics.tokensGenerated}`);
    console.log(`📊 Memory delta: +${metrics.memoryDelta.heapUsed}MB heap`);
    console.log(`💬 Response preview: "${data.response?.substring(0, 100)}..."`);
    
    return { success: true, metrics, response: data.response };
    
  } catch (error) {
    console.error('❌ LLM generation failed:', error.message);
    return { success: false, error: error.message };
  }
}

/**
 * Test streaming generation
 */
async function testStreamingGeneration(model = TEST_MODEL) {
  console.log(`🌊 Testing streaming generation with ${model}...`);
  
  const testPrompt = `Explain the concept of legal precedent in 2-3 sentences.`;
  
  try {
    const response = await fetch(`${OLLAMA_BASE_URL}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        prompt: testPrompt,
        stream: true,
        options: {
          temperature: 0.1,
          num_predict: 100
        }
      })
    });
    
    if (!response.ok) {
      throw new Error(`Streaming failed: ${response.status}`);
    }
    
    const reader = response.body?.getReader();
    if (!reader) throw new Error('No response body');
    
    const decoder = new TextDecoder();
    let fullResponse = '';
    let chunkCount = 0;
    
    console.log('📡 Streaming response:');
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n').filter(line => line.trim());
      
      for (const line of lines) {
        try {
          const data = JSON.parse(line);
          if (data.response) {
            process.stdout.write(data.response);
            fullResponse += data.response;
            chunkCount++;
          }
        } catch {
          // Skip malformed JSON
        }
      }
    }
    
    console.log(`\n✅ Streaming complete: ${chunkCount} chunks, ${fullResponse.length} chars`);
    return { success: true, chunkCount, responseLength: fullResponse.length };
    
  } catch (error) {
    console.error('❌ Streaming test failed:', error.message);
    return { success: false, error: error.message };
  }
}

/**
 * Performance stress test
 */
async function stressTest(iterations = 3) {
  console.log(`🔥 Running stress test (${iterations} iterations)...`);
  
  const results = [];
  
  for (let i = 1; i <= iterations; i++) {
    console.log(`\n--- Iteration ${i}/${iterations} ---`);
    
    const result = await testLLMGeneration();
    if (result.success) {
      results.push(result.metrics);
    }
    
    // Brief pause between iterations
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  
  if (results.length > 0) {
    const avgDuration = results.reduce((sum, r) => sum + r.duration, 0) / results.length;
    const avgTokens = results.reduce((sum, r) => sum + r.tokensGenerated, 0) / results.length;
    const maxMemory = Math.max(...results.map(r => r.endMemory.heapUsedMB));
    
    console.log(`\n📊 Stress Test Results:`);
    console.log(`   Average duration: ${Math.round(avgDuration)}ms`);
    console.log(`   Average tokens: ${Math.round(avgTokens)}`);
    console.log(`   Peak memory: ${maxMemory}MB`);
    console.log(`   Success rate: ${results.length}/${iterations}`);
  }
  
  return results;
}

/**
 * Main test runner
 */
async function main() {
  console.log('🚀 Starting Ollama Integration Test');
  console.log(`🔗 Base URL: ${OLLAMA_BASE_URL}`);
  console.log(`🏷️  Test Model: ${TEST_MODEL}`);
  console.log(`💾 Initial Memory: ${JSON.stringify(getMemoryUsage())}`);
  
  const results = {
    timestamp: new Date().toISOString(),
    nodeOptions: process.env.NODE_OPTIONS,
    tests: {}
  };
  
  // Test 1: Health Check
  results.tests.health = await testOllamaHealth();
  
  if (!results.tests.health.healthy) {
    console.log('\n❌ Ollama not available. Skipping further tests.');
    process.exit(1);
  }
  
  // Determine best available model
  const availableModel = results.tests.health.hasTestModel ? 
    TEST_MODEL : 
    results.tests.health.models[0]?.name?.split(':')[0] || 'llama3.2';
  
  console.log(`\n🎯 Using model: ${availableModel}`);
  
  // Test 2: Basic Generation
  results.tests.generation = await testLLMGeneration(availableModel);
  
  // Test 3: Streaming
  results.tests.streaming = await testStreamingGeneration(availableModel);
  
  // Test 4: Stress Test
  results.tests.stress = await stressTest(3);
  
  // Final memory check
  results.finalMemory = getMemoryUsage();
  
  // Save test results
  const reportPath = join(__dirname, '..', 'test-results', `ollama-test-${Date.now()}.json`);
  try {
    await writeFile(reportPath, JSON.stringify(results, null, 2));
    console.log(`\n📄 Test report saved: ${reportPath}`);
  } catch (error) {
    console.warn(`⚠️  Could not save test report: ${error.message}`);
  }
  
  // Summary
  const allPassed = Object.values(results.tests).every(test => 
    Array.isArray(test) ? test.length > 0 : test.success !== false
  );
  
  console.log(`\n${allPassed ? '✅' : '❌'} Integration test ${allPassed ? 'PASSED' : 'FAILED'}`);
  console.log(`💾 Final Memory: ${JSON.stringify(results.finalMemory)}`);
  
  process.exit(allPassed ? 0 : 1);
}

// Handle uncaught errors
process.on('uncaughtException', (error) => {
  console.error('❌ Uncaught exception:', error);
  process.exit(1);
});

process.on('unhandledRejection', (reason) => {
  console.error('❌ Unhandled rejection:', reason);
  process.exit(1);
});

// Run tests
main().catch(console.error);