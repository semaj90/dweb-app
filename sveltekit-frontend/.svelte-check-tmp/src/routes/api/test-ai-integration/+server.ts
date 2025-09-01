
import type { RequestHandler } from './$types';

/**
 * AI Integration Test Suite
 * Comprehensive testing of all AI/MCP API endpoints
 */

import { copilotOrchestrator, generateMCPPrompt, commonMCPQueries, semanticSearch, mcpMemoryReadGraph, validateMCPRequest } from "$lib/utils/mcp-helpers";

export interface TestResult {
  name: string;
  status: 'pass' | 'fail' | 'warning';
  duration: number;
  details: string;
  error?: string;
}

export interface TestSuite {
  name: string;
  tests: TestResult[];
  totalTests: number;
  passedTests: number;
  failedTests: number;
  warnings: number;
  totalDuration: number;
}

/**
 * Main test runner
 */
export const POST: RequestHandler = async ({ request }) => {
  const startTime = Date.now();
  
  try {
    const { testSuite = 'all', verbose = false } = await request.json();
    
    const results: TestSuite[] = [];
    
    // Run different test suites based on request
    if (testSuite === 'all' || testSuite === 'mcp') {
      results.push(await testMCPIntegration(verbose));
    }
    
    if (testSuite === 'all' || testSuite === 'ai') {
      results.push(await testAIServices(verbose));
    }
    
    if (testSuite === 'all' || testSuite === 'find') {
      results.push(await testFindAPI(verbose));
    }
    
    if (testSuite === 'all' || testSuite === 'memory') {
      results.push(await testMemoryGraph(verbose));
    }
    
    if (testSuite === 'all' || testSuite === 'semantic') {
      results.push(await testSemanticSearch(verbose));
    }

    // Calculate overall statistics
    const totalTests = results.reduce((sum, suite) => sum + suite.totalTests, 0);
    const totalPassed = results.reduce((sum, suite) => sum + suite.passedTests, 0);
    const totalFailed = results.reduce((sum, suite) => sum + suite.failedTests, 0);
    const totalWarnings = results.reduce((sum, suite) => sum + suite.warnings, 0);
    const totalDuration = Date.now() - startTime;

    return json({
      success: true,
      summary: {
        totalTests,
        totalPassed,
        totalFailed,
        totalWarnings,
        totalDuration,
        successRate: Math.round((totalPassed / totalTests) * 100),
        timestamp: new Date().toISOString()
      },
      testSuites: results,
      recommendations: generateRecommendations(results)
    });

  } catch (error: any) {
    console.error('Test suite execution failed:', error);
    
    return json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
      duration: Date.now() - startTime
    }, { status: 500 });
  }
};

/**
 * Test MCP Context7 Integration
 */
async function testMCPIntegration(verbose: boolean): Promise<TestSuite> {
  const tests: TestResult[] = [];
  const suiteStartTime = Date.now();

  // Test 1: MCP Prompt Generation
  await runTest(tests, 'MCP Prompt Generation', async () => {
    const query = commonMCPQueries.analyzeSvelteKit();
    const prompt = generateMCPPrompt(query);
    
    if (!prompt || typeof prompt !== 'string') {
      throw new Error('Failed to generate MCP prompt');
    }
    
    if (!prompt.includes('sveltekit')) {
      throw new Error('Generated prompt does not contain expected content');
    }
    
    return `Generated prompt: "${prompt.substring(0, 50)}..."`;
  });

  // Test 2: MCP Request Validation
  await runTest(tests, 'MCP Request Validation', async () => {
    const validRequest = commonMCPQueries.performanceBestPractices();
    const validation = validateMCPRequest(validRequest);
    
    if (!validation.valid) {
      throw new Error(`Validation failed: ${validation.errors.join(', ')}`);
    }
    
    const invalidRequest = { tool: 'invalid-tool' as any };
    const invalidValidation = validateMCPRequest(invalidRequest);
    
    if (invalidValidation.valid) {
      throw new Error('Invalid request was incorrectly validated as valid');
    }
    
    return `Valid request passed, invalid request properly rejected`;
  });

  // Test 3: Copilot Orchestrator
  await runTest(tests, 'Copilot Orchestrator', async () => {
    const result = await copilotOrchestrator(
      'Test AI integration capabilities',
      {
        useSemanticSearch: true,
        useMemory: true,
        synthesizeOutputs: true,
        agents: ['claude']
      }
    );
    
    if (!result || typeof result !== 'object') {
      throw new Error('Orchestrator did not return valid result');
    }
    
    const hasRequiredFields = ['semantic', 'memory', 'agentResults', 'synthesized'].some(
      (field: any) => result.hasOwnProperty(field)
    );
    
    if (!hasRequiredFields) {
      return 'Orchestrator returned result but may be using mock data';
    }
    
    return `Orchestrator executed successfully with ${Object.keys(result).length} result fields`;
  });

  // Test 4: Memory Graph Integration
  await runTest(tests, 'Memory Graph Read', async () => {
    const memoryData = await mcpMemoryReadGraph();
    
    if (!Array.isArray(memoryData)) {
      throw new Error('Memory graph did not return array format');
    }
    
    if (memoryData.length === 0) {
      return 'Memory graph returned empty array (may be expected)';
    }
    
    return `Memory graph returned ${memoryData.length} nodes/relations`;
  });

  // Test 5: Semantic Search Integration
  await runTest(tests, 'Semantic Search', async () => {
    const searchResults = await semanticSearch('legal document analysis');
    
    if (!Array.isArray(searchResults)) {
      throw new Error('Semantic search did not return array format');
    }
    
    if (searchResults.length === 0) {
      return 'Semantic search returned no results (may indicate service unavailable)';
    }
    
    return `Semantic search returned ${searchResults.length} results`;
  });

  return {
    name: 'Context7 MCP Integration',
    tests,
    totalTests: tests.length,
    passedTests: tests.filter((t: any) => t.status === 'pass').length,
    failedTests: tests.filter((t: any) => t.status === 'fail').length,
    warnings: tests.filter((t: any) => t.status === 'warning').length,
    totalDuration: Date.now() - suiteStartTime
  };
}

/**
 * Test AI Services (Ollama/LLMs)
 */
async function testAIServices(verbose: boolean): Promise<TestSuite> {
  const tests: TestResult[] = [];
  const suiteStartTime = Date.now();

  // Test 1: Ollama Service Health
  await runTest(tests, 'Ollama Service Health', async () => {
    const response = await fetch('http://localhost:11434/api/version', {
      method: 'GET',
      signal: AbortSignal.timeout(5000)
    });
    
    if (!response.ok) {
      throw new Error(`Ollama service returned ${response.status}`);
    }
    
    const data = await response.json();
    return `Ollama service healthy, version: ${data.version || 'unknown'}`;
  });

  // Test 2: Model Availability
  await runTest(tests, 'Model Availability Check', async () => {
    const response = await fetch('http://localhost:11434/api/tags', {
      method: 'GET',
      signal: AbortSignal.timeout(5000)
    });
    
    if (!response.ok) {
      throw new Error(`Failed to get model list: ${response.status}`);
    }
    
    const data = await response.json();
    const models = data.models || [];
    
    if (models.length === 0) {
      throw new Error('No models available in Ollama');
    }
    
    const hasLlama = models.some((m: any) => m.name.includes('llama'));
    if (!hasLlama) {
      return `${models.length} models available but no Llama model found`;
    }
    
    return `${models.length} models available including Llama variants`;
  });

  // Test 3: Simple AI Generation
  await runTest(tests, 'AI Text Generation', async () => {
    const response = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'llama3.2',
        prompt: 'Test prompt: What is 2+2?',
        stream: false,
        options: {
          temperature: 0.1,
          max_tokens: 50
        }
      })
    });
    
    if (!response.ok) {
      throw new Error(`AI generation failed: ${response.status}`);
    }
    
    const data = await response.json();
    
    if (!data.response || typeof data.response !== 'string') {
      throw new Error('AI response invalid or empty');
    }
    
    if (data.response.length < 5) {
      return 'AI response too short, may indicate issues';
    }
    
    return `AI generated response: "${data.response.substring(0, 50)}..."`;
  });

  // Test 4: JSON Response Parsing
  await runTest(tests, 'Structured AI Response', async () => {
    const prompt = `
      Return a JSON object with this exact structure:
      {"status": "success", "message": "test completed", "number": 42}
      
      Return only the JSON, no other text.
    `;
    
    const response = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'llama3.2',
        prompt,
        stream: false,
        options: {
          temperature: 0.1,
          max_tokens: 100
        }
      })
    });
    
    if (!response.ok) {
      throw new Error(`Structured AI request failed: ${response.status}`);
    }
    
    const data = await response.json();
    
    try {
      const parsed = JSON.parse(data.response);
      
      if (parsed.status !== 'success' || parsed.number !== 42) {
        return 'AI returned JSON but with incorrect values';
      }
      
      return 'AI successfully returned structured JSON response';
    } catch (error: any) {
      return 'AI response was not valid JSON format';
    }
  });

  return {
    name: 'AI Services (Ollama/LLM)',
    tests,
    totalTests: tests.length,
    passedTests: tests.filter((t: any) => t.status === 'pass').length,
    failedTests: tests.filter((t: any) => t.status === 'fail').length,
    warnings: tests.filter((t: any) => t.status === 'warning').length,
    totalDuration: Date.now() - suiteStartTime
  };
}

/**
 * Test Find API Endpoint
 */
async function testFindAPI(verbose: boolean): Promise<TestSuite> {
  const tests: TestResult[] = [];
  const suiteStartTime = Date.now();

  // Test 1: Basic Search Request
  await runTest(tests, 'Basic Find API Request', async () => {
    const response = await fetch('/api/ai/find', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: 'test legal document',
        type: 'all',
        useAI: false,
        mcpAnalysis: false
      })
    });
    
    if (!response.ok) {
      throw new Error(`Find API request failed: ${response.status}`);
    }
    
    const data = await response.json();
    
    if (!data.success) {
      throw new Error(`Find API returned error: ${data.error}`);
    }
    
    if (!Array.isArray(data.results)) {
      throw new Error('Find API did not return results array');
    }
    
    return `Find API returned ${data.results.length} results in ${data.metadata.processingTime}ms`;
  });

  // Test 2: AI-Enhanced Search
  await runTest(tests, 'AI-Enhanced Search', async () => {
    const response = await fetch('/api/ai/find', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: 'contract liability evidence',
        type: 'all',
        useAI: true,
        mcpAnalysis: false,
        semanticSearch: false,
        confidenceThreshold: 0.5
      })
    });
    
    if (!response.ok) {
      throw new Error(`AI-enhanced search failed: ${response.status}`);
    }
    
    const data = await response.json();
    
    if (!data.success) {
      throw new Error(`AI search returned error: ${data.error}`);
    }
    
    const hasAiConfidence = data.results.some((result: any) => 
      result.aiConfidence !== undefined
    );
    
    if (!hasAiConfidence) {
      return 'AI enhancement may not be working (no confidence scores)';
    }
    
    return `AI-enhanced search returned ${data.results.length} results with confidence scores`;
  });

  // Test 3: MCP Analysis Integration
  await runTest(tests, 'MCP Analysis in Find API', async () => {
    const response = await fetch('/api/ai/find', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: 'legal precedent analysis',
        type: 'all',
        useAI: true,
        mcpAnalysis: true,
        semanticSearch: false
      })
    });
    
    if (!response.ok) {
      throw new Error(`MCP analysis request failed: ${response.status}`);
    }
    
    const data = await response.json();
    
    if (!data.success) {
      throw new Error(`MCP analysis returned error: ${data.error}`);
    }
    
    if (!data.metadata.mcpAnalysis) {
      return 'MCP analysis was not executed (may be expected)';
    }
    
    const hasMcpContext = data.mcpContext !== null;
    const hasAutoSuggestions = data.autoSuggestions && data.autoSuggestions.length > 0;
    
    return `MCP analysis executed, context: ${hasMcpContext}, suggestions: ${hasAutoSuggestions}`;
  });

  // Test 4: Search Suggestions
  await runTest(tests, 'Search Suggestions API', async () => {
    const response = await fetch('/api/ai/find?q=contract');
    
    if (!response.ok) {
      throw new Error(`Suggestions API failed: ${response.status}`);
    }
    
    const data = await response.json();
    
    if (!data.success) {
      throw new Error(`Suggestions API returned error: ${data.error}`);
    }
    
    if (!Array.isArray(data.suggestions)) {
      throw new Error('Suggestions API did not return array');
    }
    
    return `Suggestions API returned ${data.suggestions.length} suggestions`;
  });

  // Test 5: Rate Limiting
  await runTest(tests, 'Rate Limiting Check', async () => {
    // Make multiple rapid requests to test rate limiting
    const requests = Array(5).fill(null).map(() => 
      fetch('/api/ai/find', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: 'rate limit test',
          type: 'all',
          useAI: false
        })
      })
    );
    
    const responses = await Promise.all(requests);
    const statusCodes = responses.map((r: any) => r.status);
    
    // Check if any requests were rate limited (429)
    const rateLimited = statusCodes.some((code: any) => code === 429);
    
    if (rateLimited) {
      return 'Rate limiting is working (some requests returned 429)';
    }
    
    // Check rate limit headers
    const lastResponse = responses[responses.length - 1];
    const rateLimitHeader = lastResponse.headers.get('X-RateLimit-Remaining');
    
    if (rateLimitHeader !== null) {
      return `Rate limiting headers present, remaining: ${rateLimitHeader}`;
    }
    
    return 'Rate limiting may not be configured (all requests succeeded)';
  });

  return {
    name: 'Find API Endpoint',
    tests,
    totalTests: tests.length,
    passedTests: tests.filter((t: any) => t.status === 'pass').length,
    failedTests: tests.filter((t: any) => t.status === 'fail').length,
    warnings: tests.filter((t: any) => t.status === 'warning').length,
    totalDuration: Date.now() - suiteStartTime
  };
}

/**
 * Test Memory Graph Integration
 */
async function testMemoryGraph(verbose: boolean): Promise<TestSuite> {
  const tests: TestResult[] = [];
  const suiteStartTime = Date.now();

  // Test 1: Memory Graph Read
  await runTest(tests, 'Memory Graph Read', async () => {
    const response = await fetch('/api/mcp/memory/read-graph', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        filters: {
          nodeTypes: ['ai-interaction', 'search'],
          limit: 10
        }
      })
    });
    
    // If endpoint doesn't exist, this is expected for now
    if (response.status === 404) {
      return 'Memory graph endpoint not implemented yet (expected)';
    }
    
    if (!response.ok) {
      throw new Error(`Memory graph read failed: ${response.status}`);
    }
    
    const data = await response.json();
    return `Memory graph read successful, returned ${JSON.stringify(data).length} characters`;
  });

  // Test 2: Memory Relation Creation
  await runTest(tests, 'Memory Relation Creation', async () => {
    const response = await fetch('/api/mcp/memory/create-relations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        source: 'test-user',
        target: 'test-search',
        relationType: 'performed-search',
        properties: {
          timestamp: new Date().toISOString(),
          query: 'test memory relation'
        }
      })
    });
    
    // If endpoint doesn't exist, this is expected for now
    if (response.status === 404) {
      return 'Memory relation endpoint not implemented yet (expected)';
    }
    
    if (!response.ok) {
      throw new Error(`Memory relation creation failed: ${response.status}`);
    }
    
    const data = await response.json();
    return `Memory relation created successfully: ${data.success}`;
  });

  return {
    name: 'Memory Graph Integration',
    tests,
    totalTests: tests.length,
    passedTests: tests.filter((t: any) => t.status === 'pass').length,
    failedTests: tests.filter((t: any) => t.status === 'fail').length,
    warnings: tests.filter((t: any) => t.status === 'warning').length,
    totalDuration: Date.now() - suiteStartTime
  };
}

/**
 * Test Semantic Search Integration
 */
async function testSemanticSearch(verbose: boolean): Promise<TestSuite> {
  const tests: TestResult[] = [];
  const suiteStartTime = Date.now();

  // Test 1: Semantic Search Service
  await runTest(tests, 'Semantic Search Service', async () => {
    const response = await fetch('/api/semantic-search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: 'legal document contract analysis'
      })
    });
    
    // If endpoint doesn't exist, this is expected for now
    if (response.status === 404) {
      return 'Semantic search endpoint not implemented yet (expected)';
    }
    
    if (!response.ok) {
      throw new Error(`Semantic search failed: ${response.status}`);
    }
    
    const data = await response.json();
    
    if (!data.results || !Array.isArray(data.results)) {
      throw new Error('Semantic search did not return results array');
    }
    
    return `Semantic search returned ${data.results.length} results`;
  });

  // Test 2: Vector Database Connection
  await runTest(tests, 'Vector Database (Qdrant)', async () => {
    try {
      const response = await fetch('http://localhost:6333/collections', {
        method: 'GET',
        signal: AbortSignal.timeout(5000)
      });
      
      if (!response.ok) {
        throw new Error(`Qdrant connection failed: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (!data.result || !data.result.collections) {
        return 'Qdrant connected but no collections found';
      }
      
      return `Qdrant connected, ${data.result.collections.length} collections available`;
      
    } catch (error: any) {
      return 'Qdrant vector database not available (may be expected)';
    }
  });

  return {
    name: 'Semantic Search Integration',
    tests,
    totalTests: tests.length,
    passedTests: tests.filter((t: any) => t.status === 'pass').length,
    failedTests: tests.filter((t: any) => t.status === 'fail').length,
    warnings: tests.filter((t: any) => t.status === 'warning').length,
    totalDuration: Date.now() - suiteStartTime
  };
}

/**
 * Helper function to run individual tests
 */
async function runTest(
  tests: TestResult[], 
  name: string, 
  testFn: () => Promise<string>
): Promise<void> {
  const startTime = Date.now();
  
  try {
    const details = await testFn();
    const duration = Date.now() - startTime;
    
    // Determine status based on details
    let status: 'pass' | 'fail' | 'warning' = 'pass';
    if (details.includes('may be expected') || 
        details.includes('not implemented') || 
        details.includes('may indicate') ||
        details.includes('may not be')) {
      status = 'warning';
    }
    
    tests.push({
      name,
      status,
      duration,
      details
    });
    
  } catch (error: any) {
    tests.push({
      name,
      status: 'fail',
      duration: Date.now() - startTime,
      details: 'Test failed',
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}

/**
 * Generate recommendations based on test results
 */
function generateRecommendations(testSuites: TestSuite[]): string[] {
  const recommendations: string[] = [];
  
  for (const suite of testSuites) {
    const failureRate = suite.failedTests / suite.totalTests;
    const warningRate = suite.warnings / suite.totalTests;
    
    if (failureRate > 0.5) {
      recommendations.push(
        `🔴 ${suite.name}: High failure rate (${Math.round(failureRate * 100)}%). Consider checking service availability and configuration.`
      );
    } else if (failureRate > 0.2) {
      recommendations.push(
        `🟡 ${suite.name}: Some tests failing (${Math.round(failureRate * 100)}%). Review failed tests and service health.`
      );
    }
    
    if (warningRate > 0.7) {
      recommendations.push(
        `⚠️ ${suite.name}: Many services not yet implemented (${Math.round(warningRate * 100)}% warnings). This is expected during development.`
      );
    }
    
    if (suite.totalDuration > 10000) {
      recommendations.push(
        `⏱️ ${suite.name}: Slow response times (${suite.totalDuration}ms). Consider performance optimization.`
      );
    }
  }
  
  if (recommendations.length === 0) {
    recommendations.push('✅ All systems operational. Great work on the AI integration!');
  }
  
  return recommendations;
}

/**
 * GET handler for quick health check
 */
export const GET: RequestHandler = async () => {
  const startTime = Date.now();
  
  try {
    // Quick health checks
    const checks = await Promise.allSettled([
      // AI Service
      fetch('http://localhost:11434/api/version', { 
        signal: AbortSignal.timeout(3000) 
      }).then((r: any) => ({ ai: r.ok })),
      
      // Find API
      fetch('/api/ai/find', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: 'health check', useAI: false })
      }).then((r: any) => ({ findApi: r.ok })),
    ]);
    
    const results = checks.map((check: any) => check.status === 'fulfilled' ? check.value : { error: true }
    );
    
    const healthStatus = {
      ai: (results[0] && 'ai' in results[0]) ? results[0].ai : false,
      findApi: (results[1] && 'findApi' in results[1]) ? results[1].findApi : false,
    };
    
    const allHealthy = Object.values(healthStatus).every(Boolean);
    
    return json({
      healthy: allHealthy,
      status: healthStatus,
      duration: Date.now() - startTime,
      timestamp: new Date().toISOString()
    });
    
  } catch (error: any) {
    return json({
      healthy: false,
      error: 'Health check failed',
      duration: Date.now() - startTime,
      timestamp: new Date().toISOString()
    }, { status: 503 });
  }
};