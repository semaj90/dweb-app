<script lang="ts">

  import { onMount } from 'svelte';
  import { writable } from 'svelte/store';
  
  // Test state
  const testResults = writable<unknown[]>([]);
  const isRunning = writable(false);
  const currentTest = writable<string>('');
  
  // Test configuration
  let selectedComponent = $state('sveltekit');
  let testQuery = $state('Context7 legal AI stack analysis');
  
  const components = [
    'sveltekit', 'drizzle', 'unocss', 'bits-ui', 'xstate', 
    'typescript', 'postgresql', 'autogen', 'crewai', 'vllm'
  ];

  // Test the real Context7 semantic audit API
  async function runSemanticAuditTest() {
    $isRunning = true;
    $currentTest = 'semantic-audit';
    $testResults = [];
    
    try {
      console.log('[Context7 Test] Starting semantic audit test...');
      
      const response = await fetch('/api/audit/semantic', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: testQuery,
          component: selectedComponent
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      console.log('[Context7 Test] Semantic audit completed:', data);
      
      $testResults = [
        {
          test: 'Semantic Audit API',
          status: 'success',
          timestamp: new Date().toISOString(),
          data: data,
          summary: `Analyzed ${selectedComponent} with ${data.results?.length || 0} results, ${data.triggeredAgents?.length || 0} agent triggers`
        }
      ];
      
    } catch (error) {
      console.error('[Context7 Test] Semantic audit failed:', error);
      
      $testResults = [
        {
          test: 'Semantic Audit API',
          status: 'error',
          timestamp: new Date().toISOString(),
          error: String(error),
          summary: `Failed to run semantic audit for ${selectedComponent}`
        }
      ];
    } finally {
      $isRunning = false;
      $currentTest = '';
    }
  }

  // Test Context7 semantic search directly
  async function runSemanticSearchTest() {
    $isRunning = true;
    $currentTest = 'semantic-search';
    
    try {
      console.log('[Context7 Test] Testing semantic search...');
      
      // Import the performContext7Search function dynamically
      const { performContext7Search } = await import('$lib/ai/types');
      
      const searchResults = await performContext7Search({
        query: testQuery,
        maxResults: 5,
        confidenceThreshold: 0.7,
        includeCode: true,
        includeDocs: true
      });
      
      console.log('[Context7 Test] Semantic search completed:', searchResults);
      
      $testResults = [
        ...$testResults,
        {
          test: 'Direct Semantic Search',
          status: 'success',
          timestamp: new Date().toISOString(),
          data: searchResults,
          summary: `Found ${searchResults.length} search results`
        }
      ];
      
    } catch (error) {
      console.error('[Context7 Test] Semantic search failed:', error);
      
      $testResults = [
        ...$testResults,
        {
          test: 'Direct Semantic Search', 
          status: 'error',
          timestamp: new Date().toISOString(),
          error: String(error),
          summary: 'Failed to run semantic search'
        }
      ];
    } finally {
      $isRunning = false;
      $currentTest = '';
    }
  }

  // Test Context7 agent orchestration
  async function runAgentOrchestrationTest() {
    $isRunning = true;
    $currentTest = 'agent-orchestration';
    
    try {
      console.log('[Context7 Test] Testing agent orchestration...');
      
      const { context7AgentOrchestrator } = await import('$lib/ai/types');
      
      const trigger = {
        todoId: `test_${Date.now()}`,
        action: 'analyze' as const,
        status: 'pending' as const
      };
      
      const result = await context7AgentOrchestrator.triggerAgent(trigger);
      
      console.log('[Context7 Test] Agent orchestration completed:', result);
      
      $testResults = [
        ...$testResults,
        {
          test: 'Agent Orchestration',
          status: 'success',
          timestamp: new Date().toISOString(),
          data: result,
          summary: `Agent ${trigger.action} completed for ${trigger.todoId}`
        }
      ];
      
    } catch (error) {
      console.error('[Context7 Test] Agent orchestration failed:', error);
      
      $testResults = [
        ...$testResults,
        {
          test: 'Agent Orchestration',
          status: 'error', 
          timestamp: new Date().toISOString(),
          error: String(error),
          summary: 'Failed to run agent orchestration'
        }
      ];
    } finally {
      $isRunning = false;
      $currentTest = '';
    }
  }

  // Run all tests in sequence
  async function runAllTests() {
    await runSemanticAuditTest();
    await runSemanticSearchTest();
    await runAgentOrchestrationTest();
  }

  // Clear test results
  function clearResults() {
    $testResults = [];
  }
</script>

<div class="space-y-6 p-6 max-w-6xl mx-auto">
  <div class="border-b border-gray-200 pb-4">
    <h2 class="text-2xl font-bold text-gray-900">Context7 Integration Test Suite</h2>
    <p class="text-gray-600 mt-2">
      Test the real Context7 semantic search, logging, and agent trigger implementations.
    </p>
  </div>

  <!-- Test Configuration -->
  <div class="bg-white border border-gray-200 rounded-lg p-6">
    <h3 class="text-lg font-semibold mb-4">Test Configuration</h3>
    
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div>
        <label for="component" class="block text-sm font-medium mb-2">Component to Test</label>
        <select
          id="component"
          bind:value={selectedComponent}
          class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          {#each components as component}
            <option value={component}>{component}</option>
          {/each}
        </select>
      </div>
      
      <div>
        <label for="query" class="block text-sm font-medium mb-2">Test Query</label>
        <input
          id="query"
          type="text"
          bind:value={testQuery}
          placeholder="Enter your test query..."
          class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>
    </div>
  </div>

  <!-- Test Controls -->
  <div class="bg-white border border-gray-200 rounded-lg p-6">
    <h3 class="text-lg font-semibold mb-4">Test Controls</h3>
    
    <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
      <button
        type="button"
        onclick={runSemanticAuditTest}
        disabled={$isRunning}
        class="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {#if $currentTest === 'semantic-audit'}
          Testing...
        {:else}
          Semantic Audit
        {/if}
      </button>
      
      <button
        type="button"
        onclick={runSemanticSearchTest}
        disabled={$isRunning}
        class="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {#if $currentTest === 'semantic-search'}
          Testing...
        {:else}
          Semantic Search
        {/if}
      </button>
      
      <button
        type="button"
        onclick={runAgentOrchestrationTest}
        disabled={$isRunning}
        class="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {#if $currentTest === 'agent-orchestration'}
          Testing...
        {:else}
          Agent Orchestration
        {/if}
      </button>
      
      <button
        type="button"
        onclick={runAllTests}
        disabled={$isRunning}
        class="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {#if $isRunning}
          Running Tests...
        {:else}
          Run All Tests
        {/if}
      </button>
    </div>
    
    <div class="mt-4">
      <button
        type="button"
        onclick={clearResults}
        class="px-4 py-2 border border-gray-300 rounded-md hover:bg-gray-50"
      >
        Clear Results
      </button>
    </div>
  </div>

  <!-- Test Results -->
  {#if $testResults.length > 0}
    <div class="bg-white border border-gray-200 rounded-lg p-6">
      <h3 class="text-lg font-semibold mb-4">Test Results</h3>
      
      <div class="space-y-4">
        {#each $testResults as result}
          <div class="border border-gray-200 rounded-lg p-4">
            <div class="flex items-center justify-between mb-2">
              <h4 class="font-medium text-gray-900">{result.test}</h4>
              <span class="px-2 py-1 rounded-full text-xs font-medium
                {result.status === 'success' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}">
                {result.status}
              </span>
            </div>
            
            <p class="text-sm text-gray-600 mb-2">{result.summary}</p>
            <p class="text-xs text-gray-500">{result.timestamp}</p>
            
            {#if result.error}
              <div class="mt-2 p-2 bg-red-50 rounded text-sm text-red-700">
                Error: {result.error}
              </div>
            {/if}
            
            {#if result.data}
              <details class="mt-2">
                <summary class="text-sm font-medium text-gray-700 cursor-pointer">View Details</summary>
                <pre class="mt-2 p-2 bg-gray-50 rounded text-xs overflow-x-auto">{JSON.stringify(result.data, null, 2)}</pre>
              </details>
            {/if}
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Integration Status -->
  <div class="bg-green-50 border border-green-200 rounded-lg p-6">
    <h3 class="text-lg font-semibold text-green-800 mb-2">Context7 Integration Status</h3>
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
      <div>
        <h4 class="font-medium text-green-700 mb-2">✅ Implemented Features</h4>
        <ul class="space-y-1 text-green-600">
          <li>• Real semantic search with Context7 MCP</li>
          <li>• Agent orchestration (AutoGen, CrewAI, Claude)</li>
          <li>• Self-prompting workflow automation</li>
          <li>• Comprehensive audit logging</li>
          <li>• Context7 MCP tools integration</li>
        </ul>
      </div>
      
      <div>
        <h4 class="font-medium text-green-700 mb-2">🔧 Available Tools</h4>
        <ul class="space-y-1 text-green-600">
          <li>• analyze-stack (component analysis)</li>
          <li>• generate-best-practices (security, performance, UI/UX)</li>
          <li>• suggest-integration (feature integration)</li>
          <li>• resolve-library-id (library documentation)</li>
          <li>• RAG tools (legal document search)</li>
        </ul>
      </div>
    </div>
  </div>
</div>

