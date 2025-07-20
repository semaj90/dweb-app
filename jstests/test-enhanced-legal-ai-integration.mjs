#!/usr/bin/env node

/**
 * Comprehensive Integration Test for Enhanced Legal AI System
 * Tests: Chat API, Vector Embeddings, Qdrant Integration, Evidence Analysis, Case Summaries
 */

import { createRequire } from 'module';
const require = createRequire(import.meta.url);

// Test configuration
const CONFIG = {
  BASE_URL: 'http://localhost:5173',
  OLLAMA_URL: 'http://localhost:11434',
  QDRANT_URL: 'http://localhost:6333',
  TEST_USER_ID: 'test-user-' + Date.now(),
  TEST_CASE_ID: 'test-case-' + Date.now(),
  TEST_CONVERSATION_ID: 'test-conv-' + Date.now()
};

// Test utilities
function log(message, level = 'info') {
  const timestamp = new Date().toISOString();
  const colors = {
    info: '\x1b[36m',    // cyan
    success: '\x1b[32m', // green
    error: '\x1b[31m',   // red
    warning: '\x1b[33m', // yellow
    reset: '\x1b[0m'     // reset
  };
  
  console.log(`${colors[level]}[${timestamp}] ${message}${colors.reset}`);
}

async function apiCall(endpoint, options = {}) {
  const url = `${CONFIG.BASE_URL}${endpoint}`;
  const defaultOptions = {
    headers: {
      'Content-Type': 'application/json',
      'Cookie': 'session_id=test-session-' + Date.now()
    },
    ...options
  };

  try {
    const response = await fetch(url, defaultOptions);
    const data = await response.json();
    return { success: response.ok, status: response.status, data };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Test suite
class LegalAITestSuite {
  constructor() {
    this.tests = [];
    this.results = {
      passed: 0,
      failed: 0,
      total: 0
    };
  }

  addTest(name, testFn) {
    this.tests.push({ name, testFn });
  }

  async runTests() {
    log('🧪 Starting Legal AI System Integration Tests', 'info');
    log(`Test Configuration: ${JSON.stringify(CONFIG, null, 2)}`, 'info');

    for (const test of this.tests) {
      this.results.total++;
      try {
        log(`\n🔍 Running: ${test.name}`, 'info');
        await test.testFn();
        this.results.passed++;
        log(`✅ PASSED: ${test.name}`, 'success');
      } catch (error) {
        this.results.failed++;
        log(`❌ FAILED: ${test.name} - ${error.message}`, 'error');
        if (error.details) {
          log(`   Details: ${JSON.stringify(error.details, null, 2)}`, 'error');
        }
      }
    }

    this.printSummary();
  }

  printSummary() {
    log('\n📊 Test Summary', 'info');
    log(`Total Tests: ${this.results.total}`, 'info');
    log(`Passed: ${this.results.passed}`, 'success');
    log(`Failed: ${this.results.failed}`, this.results.failed > 0 ? 'error' : 'success');
    log(`Success Rate: ${((this.results.passed / this.results.total) * 100).toFixed(1)}%`, 
        this.results.failed > 0 ? 'warning' : 'success');
  }
}

// Initialize test suite
const testSuite = new LegalAITestSuite();

// Test 1: Service Health Checks
testSuite.addTest('Service Health Checks', async () => {
  // Check Ollama
  const ollamaResponse = await fetch(`${CONFIG.OLLAMA_URL}/api/tags`);
  if (!ollamaResponse.ok) {
    throw new Error('Ollama service not available');
  }
  
  // Check Qdrant
  const qdrantResponse = await fetch(`${CONFIG.QDRANT_URL}/health`);
  if (!qdrantResponse.ok) {
    throw new Error('Qdrant service not available');
  }
  
  log('All core services are healthy', 'success');
});

// Test 2: Vector Embedding API
testSuite.addTest('Vector Embedding API', async () => {
  const testText = 'This is a test evidence document about a criminal case involving theft and fraud.';
  
  const result = await apiCall('/api/embed', {
    method: 'POST',
    body: JSON.stringify({
      text: testText,
      type: 'evidence',
      metadata: {
        userId: CONFIG.TEST_USER_ID,
        caseId: CONFIG.TEST_CASE_ID,
        evidenceId: 'test-evidence-1',
        category: 'document'
      }
    })
  });

  if (!result.success) {
    throw new Error(`Embedding API failed: ${result.data?.error || 'Unknown error'}`);
  }

  if (!result.data.vector || !Array.isArray(result.data.vector)) {
    throw new Error('Invalid vector response');
  }

  log(`Embedding created with ${result.data.vector.length} dimensions`, 'success');
});

// Test 3: Chat API with Vector Context
testSuite.addTest('Chat API with Vector Context', async () => {
  const testMessage = 'Can you help me analyze evidence in this criminal case?';
  
  const result = await apiCall('/api/chat', {
    method: 'POST',
    body: JSON.stringify({
      message: testMessage,
      conversationId: CONFIG.TEST_CONVERSATION_ID,
      userId: CONFIG.TEST_USER_ID,
      caseId: CONFIG.TEST_CASE_ID,
      mode: 'professional',
      useContext: true
    })
  });

  if (!result.success) {
    throw new Error(`Chat API failed: ${result.data?.error || 'Unknown error'}`);
  }

  if (!result.data.message || !result.data.message.content) {
    throw new Error('Invalid chat response');
  }

  log(`Chat response: "${result.data.message.content.substring(0, 100)}..."`, 'success');
  
  if (result.data.suggestions && result.data.suggestions.length > 0) {
    log(`Received ${result.data.suggestions.length} suggestions`, 'success');
  }
});

// Test 4: Evidence Analysis API
testSuite.addTest('Evidence Analysis API', async () => {
  const testEvidence = {
    caseId: CONFIG.TEST_CASE_ID,
    content: 'Security camera footage showing the defendant entering the building at 10:45 PM on January 15th, 2024. The footage clearly shows the defendant carrying a bag and using tools to break the lock.',
    type: 'video',
    generateAnalysis: true,
    metadata: {
      filename: 'security_footage_01_15_2024.mp4',
      timestamp: '2024-01-15T22:45:00Z',
      location: 'Main entrance camera'
    }
  };
  
  const result = await apiCall('/api/evidence', {
    method: 'POST',
    body: JSON.stringify(testEvidence)
  });

  if (!result.success) {
    throw new Error(`Evidence API failed: ${result.data?.error || 'Unknown error'}`);
  }

  if (!result.data.evidence) {
    throw new Error('No evidence object returned');
  }

  if (result.data.analysis) {
    log(`Evidence analysis generated: ${result.data.analysis.summary}`, 'success');
    log(`Admissibility: ${result.data.analysis.admissibility}`, 'success');
    log(`Relevance: ${result.data.analysis.relevance}/10`, 'success');
  }
});

// Test 5: Case Summary Generation
testSuite.addTest('Case Summary Generation', async () => {
  const result = await apiCall('/api/cases/summary', {
    method: 'POST',
    body: JSON.stringify({
      caseId: CONFIG.TEST_CASE_ID,
      includeEvidence: true,
      includeTimeline: true,
      analysisDepth: 'comprehensive'
    })
  });

  if (!result.success) {
    throw new Error(`Case summary API failed: ${result.data?.error || 'Unknown error'}`);
  }

  if (!result.data.summary) {
    throw new Error('No summary generated');
  }

  log(`Case summary generated: "${result.data.summary.overview.substring(0, 100)}..."`, 'success');
  log(`Key findings: ${result.data.summary.keyFindings?.length || 0}`, 'success');
  log(`Risk level: ${result.data.summary.riskAssessment?.level || 'unknown'}`, 'success');
});

// Test 6: Conversation History Retrieval
testSuite.addTest('Conversation History Retrieval', async () => {
  // First, ensure we have some conversation history
  await sleep(1000); // Wait for previous chat to be stored
  
  const result = await apiCall(`/api/chat?conversationId=${CONFIG.TEST_CONVERSATION_ID}&userId=${CONFIG.TEST_USER_ID}&limit=10`);

  if (!result.success) {
    throw new Error(`Conversation history API failed: ${result.data?.error || 'Unknown error'}`);
  }

  if (!result.data.conversation || !Array.isArray(result.data.conversation)) {
    throw new Error('Invalid conversation history format');
  }

  log(`Retrieved ${result.data.conversation.length} conversation messages`, 'success');
});

// Test 7: Vector Similarity Search
testSuite.addTest('Vector Similarity Search', async () => {
  // Create a test vector for search
  const testVector = Array.from({length: 1536}, () => Math.random() * 2 - 1);
  
  const result = await apiCall(`/api/embed?vector=${JSON.stringify(testVector)}&limit=5&threshold=0.5`);

  if (!result.success) {
    throw new Error(`Vector similarity search failed: ${result.data?.error || 'Unknown error'}`);
  }

  if (!result.data.similarity_results || !Array.isArray(result.data.similarity_results)) {
    throw new Error('Invalid similarity search results');
  }

  log(`Found ${result.data.similarity_results.length} similar items`, 'success');
});

// Test 8: End-to-End Workflow
testSuite.addTest('End-to-End Legal AI Workflow', async () => {
  log('Testing complete legal AI workflow...', 'info');
  
  // Step 1: Create evidence
  const evidenceResult = await apiCall('/api/evidence', {
    method: 'POST',
    body: JSON.stringify({
      caseId: CONFIG.TEST_CASE_ID,
      content: 'Witness statement from John Doe: "I saw the defendant leaving the scene at approximately 11:15 PM. He was wearing a dark jacket and carrying a backpack."',
      type: 'document',
      generateAnalysis: true,
      tags: ['witness-statement', 'eyewitness']
    })
  });

  if (!evidenceResult.success) {
    throw new Error(`Evidence creation failed: ${evidenceResult.data?.error}`);
  }

  // Step 2: Ask AI about the evidence
  await sleep(2000); // Wait for embedding to be processed
  
  const chatResult = await apiCall('/api/chat', {
    method: 'POST',
    body: JSON.stringify({
      message: 'What can you tell me about the witness statement evidence in this case?',
      conversationId: CONFIG.TEST_CONVERSATION_ID + '-workflow',
      userId: CONFIG.TEST_USER_ID,
      caseId: CONFIG.TEST_CASE_ID,
      mode: 'investigative',
      useContext: true
    })
  });

  if (!chatResult.success) {
    throw new Error(`Chat query failed: ${chatResult.data?.error}`);
  }

  // Step 3: Generate case summary
  const summaryResult = await apiCall('/api/cases/summary', {
    method: 'POST',
    body: JSON.stringify({
      caseId: CONFIG.TEST_CASE_ID,
      includeEvidence: true,
      includeTimeline: true,
      analysisDepth: 'detailed'
    })
  });

  if (!summaryResult.success) {
    throw new Error(`Case summary failed: ${summaryResult.data?.error}`);
  }

  log('End-to-end workflow completed successfully', 'success');
  log(`- Evidence created and analyzed`, 'success');
  log(`- AI provided contextual response`, 'success');
  log(`- Case summary generated with ${summaryResult.data.summary.keyFindings?.length || 0} key findings`, 'success');
});

// Test 9: Error Handling and Edge Cases
testSuite.addTest('Error Handling and Edge Cases', async () => {
  // Test invalid embedding request
  const invalidEmbedResult = await apiCall('/api/embed', {
    method: 'POST',
    body: JSON.stringify({
      text: '', // Empty text
      type: 'invalid_type'
    })
  });

  if (invalidEmbedResult.success) {
    throw new Error('API should have rejected invalid embedding request');
  }

  // Test chat without required fields
  const invalidChatResult = await apiCall('/api/chat', {
    method: 'POST',
    body: JSON.stringify({
      message: 'Test message'
      // Missing required fields
    })
  });

  if (invalidChatResult.success) {
    throw new Error('API should have rejected invalid chat request');
  }

  log('Error handling working correctly', 'success');
});

// Test 10: Performance and Load Test
testSuite.addTest('Performance and Load Test', async () => {
  const startTime = Date.now();
  const concurrentRequests = 5;
  
  const promises = Array.from({ length: concurrentRequests }, (_, i) => 
    apiCall('/api/chat', {
      method: 'POST',
      body: JSON.stringify({
        message: `Performance test message ${i + 1}`,
        conversationId: CONFIG.TEST_CONVERSATION_ID + '-perf-' + i,
        userId: CONFIG.TEST_USER_ID,
        caseId: CONFIG.TEST_CASE_ID,
        mode: 'professional'
      })
    })
  );

  const results = await Promise.all(promises);
  const endTime = Date.now();
  const totalTime = endTime - startTime;

  const successfulRequests = results.filter(r => r.success).length;
  
  if (successfulRequests < concurrentRequests * 0.8) {
    throw new Error(`Only ${successfulRequests}/${concurrentRequests} requests succeeded`);
  }

  log(`${successfulRequests}/${concurrentRequests} concurrent requests completed in ${totalTime}ms`, 'success');
  log(`Average response time: ${(totalTime / concurrentRequests).toFixed(2)}ms`, 'success');
});

// Run all tests
async function main() {
  try {
    await testSuite.runTests();
    
    if (testSuite.results.failed > 0) {
      process.exit(1);
    } else {
      log('\n🎉 All tests passed! Legal AI System is working correctly.', 'success');
      process.exit(0);
    }
  } catch (error) {
    log(`\n💥 Test runner error: ${error.message}`, 'error');
    process.exit(1);
  }
}

main();
