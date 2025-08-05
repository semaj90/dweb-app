<script lang="ts">
  import { onMount } from 'svelte';
  import { OllamaService } from '$lib/services/ollamaService';
  import { multiLayerCache } from '$lib/services/multiLayerCache';
  
  // Initialize services
  const ollamaService = new OllamaService();
  
  let testResults = $state<Array<{
    test: string;
    status: 'pending' | 'success' | 'error';
    message: string;
    duration?: number;
  }>>([]);
  
  let isRunning = $state(false);
  
  async function runAITests() {
    isRunning = true;
    testResults = [];
    
    // Test 1: Ollama Service Health Check
    await runTest('Ollama Health Check', async () => {
      const health = await ollamaService.healthCheck();
      if (health.status === 'healthy') {
        return `Ollama is healthy. Models: ${health.models.join(', ')}`;
      } else {
        throw new Error(`Ollama unhealthy: ${health.error}`);
      }
    });
    
    // Test 2: Cache System
    await runTest('Cache System Test', async () => {
      await multiLayerCache.set('test-key', { message: 'Hello AI!' }, {
        type: 'query',
        ttl: 300
      });
      
      const retrieved = await multiLayerCache.get('test-key');
      if (retrieved?.message === 'Hello AI!') {
        return 'Cache system working correctly';
      } else {
        throw new Error('Cache retrieval failed');
      }
    });
    
    // Test 3: Text Analysis
    await runTest('Text Analysis Test', async () => {
      const testText = "This is a legal document regarding evidence in case 2024-001.";
      const analysis = await ollamaService.analyzeDocument(testText, 'summary');
      
      if (analysis && analysis.length > 0) {
        return `Analysis completed: ${analysis.substring(0, 100)}...`;
      } else {
        throw new Error('No analysis returned');
      }
    });
    
    // Test 4: Embedding Generation
    await runTest('Embedding Generation Test', async () => {
      const testText = "Legal document embedding test";
      const embedding = await ollamaService.generateEmbedding(testText);
      
      if (embedding && embedding.length > 0) {
        return `Embedding generated: ${embedding.length} dimensions`;
      } else {
        throw new Error('No embedding generated');
      }
    });
    
    isRunning = false;
  }
  
  async function runTest(testName: string, testFn: () => Promise<string>) {
    const startTime = Date.now();
    testResults = [...testResults, { test: testName, status: 'pending', message: 'Running...' }];
    
    try {
      const result = await testFn();
      const duration = Date.now() - startTime;
      
      testResults = testResults.map(t => 
        t.test === testName 
          ? { ...t, status: 'success' as const, message: result, duration }
          : t
      );
    } catch (error) {
      const duration = Date.now() - startTime;
      testResults = testResults.map(t => 
        t.test === testName 
          ? { ...t, status: 'error' as const, message: error.message, duration }
          : t
      );
    }
  }
  
  onMount(() => {
    // Auto-run tests on mount
    runAITests();
  });
</script>

<div class="container mx-auto py-8 max-w-4xl">
  <div class="mb-8">
    <h1 class="text-4xl font-bold mb-2 flex items-center gap-2">
      🤖 AI Pipeline Testing
    </h1>
    <p class="text-lg text-muted-foreground">
      Testing core AI components and services
    </p>
  </div>

  <div class="mb-6">
    <button 
      onclick={runAITests}
      disabled={isRunning}
      class="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
    >
      {isRunning ? 'Running Tests...' : 'Run AI Tests'}
    </button>
  </div>

  <div class="space-y-4">
    {#each testResults as result}
      <div class="p-4 border rounded-lg {
        result.status === 'success' ? 'border-green-200 bg-green-50' :
        result.status === 'error' ? 'border-red-200 bg-red-50' :
        'border-gray-200 bg-gray-50'
      }">
        <div class="flex items-center justify-between mb-2">
          <h3 class="font-semibold">{result.test}</h3>
          <div class="flex items-center gap-2">
            {#if result.duration}
              <span class="text-sm text-gray-500">{result.duration}ms</span>
            {/if}
            <span class="px-2 py-1 text-xs rounded {
              result.status === 'success' ? 'bg-green-200 text-green-800' :
              result.status === 'error' ? 'bg-red-200 text-red-800' :
              'bg-gray-200 text-gray-800'
            }">
              {result.status.toUpperCase()}
            </span>
          </div>
        </div>
        <p class="text-sm {
          result.status === 'error' ? 'text-red-700' : 'text-gray-700'
        }">
          {result.message}
        </p>
      </div>
    {/each}
  </div>
  
  {#if testResults.length === 0 && !isRunning}
    <div class="text-center py-8 text-gray-500">
      Click "Run AI Tests" to start testing the AI pipeline
    </div>
  {/if}
</div>

<style>
  .container {
    font-family: 'Inter', sans-serif;
  }
</style>