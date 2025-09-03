<script lang="ts">
  import { aiAssistantManager, isProcessing, currentResponse, aiError } from '$lib/stores/aiAssistant.svelte.js';
  import { webAssemblyLangChainBridge } from '$lib/services/webasm-langchain-bridge.js';
  import { webAssemblyAIAdapter } from '$lib/adapters/webasm-ai-adapter.js';

  // Test state
  let testQuery = $state('What are the key legal risks in a standard employment contract?');
  let testResults = $state<any[]>([]);
  let showHealthStatus = $state(false);

  // Test methods
  const testMethods = [
    {
      name: 'WebAssembly Only',
      description: 'Direct WebAssembly llama.cpp inference',
      options: { useWebAssembly: true, useLangChain: false, useHybridRAG: false }
    },
    {
      name: 'LangChain + WebAssembly',
      description: 'LangChain document retrieval with WebAssembly generation',
      options: { useWebAssembly: true, useLangChain: true, useHybridRAG: false }
    },
    {
      name: 'Hybrid RAG',
      description: 'Hybrid WebAssembly + Ollama processing with LangChain',
      options: { useWebAssembly: true, useLangChain: true, useHybridRAG: true }
    },
    {
      name: 'Ollama Fallback',
      description: 'Traditional Ollama processing (fallback)',
      options: { useWebAssembly: false, useLangChain: false, useHybridRAG: false }
    }
  ];

  // Run test with specific method
  async function runTest(method: typeof testMethods[0]) {
    if (!testQuery.trim()) {
      alert('Please enter a test query');
      return;
    }

    const startTime = performance.now();
    
    try {
      console.log(`[Test] Running test: ${method.name}`);
      
      await aiAssistantManager.sendMessage(testQuery, method.options);
      
      const endTime = performance.now();
      const duration = endTime - startTime;

      const result = {
        method: method.name,
        description: method.description,
        query: testQuery,
        response: $currentResponse,
        error: $aiError,
        duration: Math.round(duration),
        timestamp: new Date().toLocaleTimeString(),
        success: !$aiError
      };

      testResults = [result, ...testResults];
      
      console.log(`[Test] ${method.name} completed in ${duration}ms`);

    } catch (error: any) {
      const result = {
        method: method.name,
        description: method.description,
        query: testQuery,
        response: '',
        error: error.message,
        duration: Math.round(performance.now() - startTime),
        timestamp: new Date().toLocaleTimeString(),
        success: false
      };

      testResults = [result, ...testResults];
      console.error(`[Test] ${method.name} failed:`, error);
    }
  }

  // Run all tests sequentially
  async function runAllTests() {
    if (!testQuery.trim()) {
      alert('Please enter a test query');
      return;
    }

    testResults = [];
    
    for (const method of testMethods) {
      await runTest(method);
      // Wait between tests to avoid overwhelming the system
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }

  // Clear test results
  function clearResults() {
    testResults = [];
  }

  // Get health status
  async function getHealthStatus() {
    try {
      const webAsmHealth = webAssemblyAIAdapter.getHealthStatus();
      const bridgeHealth = webAssemblyLangChainBridge.getHealthStatus();
      const aiAssistantState = aiAssistantManager.state;

      return {
        webAssemblyAdapter: webAsmHealth,
        langChainBridge: bridgeHealth,
        aiAssistantManager: {
          initialized: !!aiAssistantManager,
          isProcessing: aiAssistantState.isProcessing,
          currentModel: aiAssistantState.model,
          clusterHealth: aiAssistantState.ollamaClusterHealth,
          totalQueries: aiAssistantState.usage.totalQueries
        }
      };
    } catch (error: any) {
      console.error('[Test] Health check failed:', error);
      return { error: error.message };
    }
  }

  // Sample legal queries
  const sampleQueries = [
    'What are the key legal risks in a standard employment contract?',
    'Explain the difference between indemnification and limitation of liability clauses.',
    'What are the essential elements of a valid contract under common law?',
    'How does force majeure typically apply in commercial agreements?',
    'What are the main considerations for data privacy compliance in contracts?'
  ];
</script>

<div class="webassembly-langchain-test max-w-6xl mx-auto p-6">
  <div class="mb-8">
    <h1 class="text-3xl font-bold mb-4">WebAssembly + LangChain Integration Test</h1>
    <p class="text-gray-600 mb-4">
      Test suite for the integrated WebAssembly llama.cpp + LangChain RAG system. 
      Compare different processing methods and evaluate performance.
    </p>
    
    <!-- Health Status Toggle -->
    <button 
      onclick={() => showHealthStatus = !showHealthStatus}
      class="mb-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
    >
      {showHealthStatus ? 'Hide' : 'Show'} Health Status
    </button>

    {#if showHealthStatus}
      {#await getHealthStatus()}
        <div class="bg-gray-100 p-4 rounded mb-4">
          <p>Loading health status...</p>
        </div>
      {:then health}
        <div class="bg-green-50 border border-green-200 p-4 rounded mb-4">
          <h3 class="font-semibold mb-2">System Health Status</h3>
          <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div>
              <h4 class="font-medium">WebAssembly Adapter</h4>
              <ul class="text-gray-600">
                <li>Initialized: {health.webAssemblyAdapter.initialized ? '✅' : '❌'}</li>
                <li>Model Loaded: {health.webAssemblyAdapter.modelLoaded ? '✅' : '❌'}</li>
                <li>WebGPU: {health.webAssemblyAdapter.webgpuEnabled ? '✅' : '❌'}</li>
                <li>Worker: {health.webAssemblyAdapter.workerEnabled ? '✅' : '❌'}</li>
              </ul>
            </div>
            <div>
              <h4 class="font-medium">LangChain Bridge</h4>
              <ul class="text-gray-600">
                <li>Initialized: {health.langChainBridge.bridgeInitialized ? '✅' : '❌'}</li>
                <li>WebAssembly: {health.langChainBridge.webAssemblyAvailable ? '✅' : '❌'}</li>
                <li>LangChain: {health.langChainBridge.langChainHealthy ? '✅' : '❌'}</li>
              </ul>
            </div>
            <div>
              <h4 class="font-medium">AI Assistant Manager</h4>
              <ul class="text-gray-600">
                <li>Initialized: {health.aiAssistantManager.initialized ? '✅' : '❌'}</li>
                <li>Processing: {health.aiAssistantManager.isProcessing ? '🔄' : '✅'}</li>
                <li>Total Queries: {health.aiAssistantManager.totalQueries}</li>
                <li>Model: {health.aiAssistantManager.currentModel}</li>
              </ul>
            </div>
          </div>
        </div>
      {:catch error}
        <div class="bg-red-50 border border-red-200 p-4 rounded mb-4">
          <p class="text-red-700">Health check failed: {error}</p>
        </div>
      {/await}
    {/if}
  </div>

  <!-- Test Query Input -->
  <div class="mb-6">
    <label for="testQuery" class="block text-sm font-medium mb-2">Test Query:</label>
    <div class="flex gap-2 mb-2">
      <input
        id="testQuery"
        bind:value={testQuery}
        placeholder="Enter your legal AI test query..."
        class="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
      />
      <button
        onclick={runAllTests}
        disabled={$isProcessing}
        class="px-6 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {$isProcessing ? 'Testing...' : 'Run All Tests'}
      </button>
      <button
        onclick={clearResults}
        class="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700"
      >
        Clear
      </button>
    </div>

    <!-- Sample Queries -->
    <div class="text-sm">
      <span class="font-medium">Sample queries:</span>
      <div class="flex flex-wrap gap-2 mt-1">
        {#each sampleQueries as query}
          <button
            onclick={() => testQuery = query}
            class="px-2 py-1 bg-blue-100 text-blue-800 rounded hover:bg-blue-200 text-xs"
          >
            {query.substring(0, 50)}...
          </button>
        {/each}
      </div>
    </div>
  </div>

  <!-- Individual Test Buttons -->
  <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
    {#each testMethods as method}
      <div class="border border-gray-200 rounded-lg p-4">
        <h3 class="font-semibold text-sm mb-2">{method.name}</h3>
        <p class="text-xs text-gray-600 mb-3">{method.description}</p>
        <button
          onclick={() => runTest(method)}
          disabled={$isProcessing}
          class="w-full px-3 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 text-sm"
        >
          {$isProcessing ? 'Running...' : 'Test'}
        </button>
      </div>
    {/each}
  </div>

  <!-- Current Processing Status -->
  {#if $isProcessing}
    <div class="bg-yellow-50 border border-yellow-200 p-4 rounded mb-6">
      <div class="flex items-center gap-2">
        <div class="animate-spin h-5 w-5 border-2 border-yellow-600 border-t-transparent rounded-full"></div>
        <span class="font-medium">Processing query...</span>
      </div>
      <p class="text-sm text-gray-600 mt-1">
        Query: "{aiAssistantManager.state.currentQuery}"
      </p>
    </div>
  {/if}

  <!-- Test Results -->
  {#if testResults.length > 0}
    <div class="space-y-4">
      <h2 class="text-2xl font-semibold">Test Results</h2>
      
      {#each testResults as result}
        <div class="border rounded-lg p-4 {result.success ? 'border-green-200 bg-green-50' : 'border-red-200 bg-red-50'}">
          <div class="flex justify-between items-start mb-3">
            <div>
              <h3 class="font-semibold {result.success ? 'text-green-800' : 'text-red-800'}">
                {result.method} {result.success ? '✅' : '❌'}
              </h3>
              <p class="text-sm text-gray-600">{result.description}</p>
            </div>
            <div class="text-right text-sm text-gray-500">
              <div>{result.timestamp}</div>
              <div>{result.duration}ms</div>
            </div>
          </div>

          <div class="mb-3">
            <h4 class="font-medium text-sm mb-1">Query:</h4>
            <p class="text-sm text-gray-700 bg-gray-100 p-2 rounded">{result.query}</p>
          </div>

          {#if result.success && result.response}
            <div class="mb-3">
              <h4 class="font-medium text-sm mb-1">Response:</h4>
              <div class="text-sm bg-white p-3 rounded border max-h-48 overflow-y-auto">
                {result.response}
              </div>
            </div>
          {/if}

          {#if result.error}
            <div>
              <h4 class="font-medium text-sm mb-1">Error:</h4>
              <p class="text-sm text-red-700 bg-red-100 p-2 rounded">{result.error}</p>
            </div>
          {/if}
        </div>
      {/each}
    </div>
  {:else}
    <div class="text-center py-12 text-gray-500">
      <p>No test results yet. Run some tests to see the comparison!</p>
    </div>
  {/if}
</div>

<style>
  .webassembly-langchain-test {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }
</style>