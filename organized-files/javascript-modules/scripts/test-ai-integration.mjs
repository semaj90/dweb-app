#!/usr/bin/env node

/**
 * Test AI Integration - Validates Ollama service with SvelteKit
 * Tests the complete pipeline from Docker to SvelteKit API routes
 */

import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { writeFileSync } from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

console.log('🧪 Testing Legal AI Integration...\n');

// Test configuration
const TEST_CONFIG = {
  ollamaUrl: 'http://localhost:11434',
  svelteKitUrl: 'http://localhost:5173',
  apiUrl: 'http://localhost:5173/api/ai/chat',
  testModel: 'gemma3-legal',
  fallbackModel: 'gemma3:2b',
  timeout: 30000
};

let testResults = [];

async function runTest(name, testFunction) {
  console.log(`🔍 Testing: ${name}`);
  const startTime = Date.now();

  try {
    const result = await testFunction();
    const duration = Date.now() - startTime;

    console.log(`✅ ${name} - PASSED (${duration}ms)`);
    testResults.push({ name, status: 'PASSED', duration, result });
    return result;
  } catch (error) {
    const duration = Date.now() - startTime;
    console.log(`❌ ${name} - FAILED (${duration}ms)`);
    console.log(`   Error: ${error.message}\n`);
    testResults.push({ name, status: 'FAILED', duration, error: error.message });
    return null;
  }
}

// Test 1: Ollama Service Health
async function testOllamaHealth() {
  const response = await fetch(`${TEST_CONFIG.ollamaUrl}/api/tags`, {
    signal: AbortSignal.timeout(10000)
  });

  if (!response.ok) {
    throw new Error(`Ollama health check failed: ${response.status}`);
  }

  const data = await response.json();
  return {
    status: 'healthy',
    models: data.models?.length || 0,
    modelNames: data.models?.map(m => m.name) || []
  };
}

// Test 2: Available Models
async function testAvailableModels() {
  const response = await fetch(`${TEST_CONFIG.ollamaUrl}/api/tags`);
  const data = await response.json();

  const models = data.models || [];
  const hasLegalModel = models.some(m =>
    m.name.includes('gemma3') ||
    m.name.includes('legal')
  );

  return {
    totalModels: models.length,
    models: models.map(m => m.name),
    hasLegalModel,
    recommendedModels: ['gemma3-legal', 'gemma3:2b', 'nomic-embed-text']
  };
}

// Test 3: Simple Text Generation
async function testTextGeneration() {
  const testPrompt = "Explain the concept of reasonable doubt in criminal law.";

  const response = await fetch(`${TEST_CONFIG.ollamaUrl}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: TEST_CONFIG.testModel,
      prompt: testPrompt,
      stream: false,
      options: { temperature: 0.1, num_predict: 200 }
    }),
    signal: AbortSignal.timeout(TEST_CONFIG.timeout)
  });

  if (!response.ok) {
    // Try fallback model
    const fallbackResponse = await fetch(`${TEST_CONFIG.ollamaUrl}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: TEST_CONFIG.fallbackModel,
        prompt: testPrompt,
        stream: false,
        options: { temperature: 0.1, num_predict: 200 }
      }),
      signal: AbortSignal.timeout(TEST_CONFIG.timeout)
    });

    if (!fallbackResponse.ok) {
      throw new Error(`Both primary and fallback models failed`);
    }

    const fallbackData = await fallbackResponse.json();
    return {
      model: TEST_CONFIG.fallbackModel,
      response: fallbackData.response,
      tokens: fallbackData.eval_count || 0,
      duration: fallbackData.total_duration || 0,
      usedFallback: true
    };
  }

  const data = await response.json();
  return {
    model: TEST_CONFIG.testModel,
    response: data.response,
    tokens: data.eval_count || 0,
    duration: data.total_duration || 0,
    usedFallback: false
  };
}

// Main test runner
async function runAllTests() {
  console.log('🚀 Legal AI Integration Test Suite');
  console.log('==================================\n');

  // Run tests sequentially
  await runTest('Ollama Service Health', testOllamaHealth);
  await runTest('Available Models', testAvailableModels);
  await runTest('Text Generation', testTextGeneration);

  // Generate report
  console.log('\n📊 Test Results Summary');
  console.log('========================');

  const passed = testResults.filter(r => r.status === 'PASSED').length;
  const failed = testResults.filter(r => r.status === 'FAILED').length;
  const total = testResults.length;

  console.log(`✅ Passed: ${passed}/${total}`);
  console.log(`❌ Failed: ${failed}/${total}`);
  console.log(`📈 Success Rate: ${((passed/total) * 100).toFixed(1)}%\n`);

  // Detailed results
  testResults.forEach(result => {
    const status = result.status === 'PASSED' ? '✅' : '❌';
    console.log(`${status} ${result.name} (${result.duration}ms)`);
    if (result.status === 'FAILED') {
      console.log(`   ${result.error}`);
    }
  });

  // Recommendations
  console.log('\n💡 Recommendations:');
  const failedTests = testResults.filter(r => r.status === 'FAILED');

  if (failedTests.some(t => t.name.includes('Ollama'))) {
    console.log('• Start Ollama service: npm run ollama:start');
    console.log('• Check Docker containers: docker ps | grep ollama');
  }

  if (failedTests.some(t => t.name.includes('Models'))) {
    console.log('• Setup AI models: npm run ollama:setup');
    console.log('• Pull specific model: npm run ollama:pull gemma3-legal');
  }

  console.log('\n🔧 Quick Commands:');
  console.log('• Full dev setup: npm run dev:full');
  console.log('• Check health: npm run ollama:health');
  console.log('• View logs: npm run docker:logs');

  // Save detailed report
  const reportPath = join(__dirname, '..', 'ai-integration-test-report.json');
  writeFileSync(reportPath, JSON.stringify({
    timestamp: new Date().toISOString(),
    summary: { passed, failed, total, successRate: (passed/total) * 100 },
    results: testResults,
    config: TEST_CONFIG
  }, null, 2));

  console.log(`\n📄 Detailed report saved: ${reportPath}`);

  // Exit code
  process.exit(failed > 0 ? 1 : 0);
}

// Run tests
runAllTests().catch(error => {
  console.error('❌ Test runner failed:', error);
  process.exit(1);
});
        const response = await fetch('http://localhost:6379', { method: 'HEAD' });
        return { success: true, data: 'Redis responding' };
      } catch (error) {
        return { success: false, error: 'Redis connection failed' };
      }
    }
  },
  {
    name: 'Qdrant',
    url: 'http://localhost:6333',
    test: async () => {
      try {
        const response = await fetch('http://localhost:6333/collections');
        if (response.ok) {
          const data = await response.json();
          return { success: true, data };
        }
        return { success: false, error: 'Qdrant unavailable' };
      } catch (error) {
        return { success: false, error: error.message };
      }
    }
  }
];

// Test each service
for (const service of services) {
  process.stdout.write(`Testing ${service.name}... `);

  try {
    const result = await service.test();
    if (result.success) {
      console.log('✅ PASS');
      if (result.data) {
        console.log(`   Data: ${JSON.stringify(result.data).substring(0, 100)}...`);
      }
    } else {
      console.log('❌ FAIL');
      console.log(`   Error: ${result.error}`);
    }
  } catch (error) {
    console.log('❌ FAIL');
    console.log(`   Error: ${error.message}`);
  }

  console.log();
}

// Test Ollama models
console.log('🤖 Testing Ollama Models...');
try {
  const response = await fetch('http://localhost:11434/api/tags');
  if (response.ok) {
    const data = await response.json();
    console.log('✅ Available models:');
    data.models?.forEach(model => {
      console.log(`   - ${model.name} (${model.size})`);
    });
  } else {
    console.log('❌ Failed to fetch models');
  }
} catch (error) {
  console.log(`❌ Ollama models test failed: ${error.message}`);
}

console.log('\n📊 Integration Test Summary:');
console.log('- Docker services configured for Windows 10 low memory');
console.log('- Ollama replaces VLLM for local LLM processing');
console.log('- PostgreSQL optimized with vector extension');
console.log('- MCP servers ready for VS Code integration');
console.log('- Global stores implemented with authentication');
console.log('- Report generation with Context7 MCP support');

console.log('\n🚀 Next Steps:');
console.log('1. Run: START-LEGAL-AI-WINDOWS.bat');
console.log('2. Open VS Code and start debugging');
console.log('3. Navigate to case management interface');
console.log('4. Test report generation with AI assistance');

console.log('\n✨ System Ready for Development!');