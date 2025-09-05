<script lang="ts">
  import { onMount } from 'svelte';
  
  let testResults = $state<Array<{
    test: string
    status: 'pending' | 'success' | 'error';
    message: string
    duration?: number;
  }>>([]);
  
  let isRunning = $state(false);
  
  async function runSystemTests() {
    isRunning = true;
    testResults = [];
    
    // Test 1: Basic API connectivity 
    await runTest('API Health Check', async () => {
      const response = await fetch('/api/health');
      if (response.ok) {
        const data = await response.json();
        return `API is healthy: ${data.status}`;
      } else {
        throw new Error(`API returned ${response.status}`);
      }
    });
    
    // Test 2: Database connectivity
    await runTest('Database Connectivity', async () => {
      const response = await fetch('/api/database/health');
      if (response.ok) {
        const data = await response.json();
        return `Database: ${data.status}`;
      } else {
        throw new Error('Database connection failed');
      }
    });
    
    // Test 3: Basic AI endpoint
    await runTest('AI Endpoint Test', async () => {
      const response = await fetch('/api/ai/health');
      if (response.ok) {
        const data = await response.json();
        return `AI service: ${data.status || 'Available'}`;
      } else {
        throw new Error('AI endpoint unavailable');
      }
    });
    
    // Test 4: Frontend functionality
    await runTest('Frontend Systems', async () => {
      // Test local storage
      localStorage.setItem('test-key', 'test-value');
      const retrieved = localStorage.getItem('test-key');
      localStorage.removeItem('test-key');
      
      if (retrieved === 'test-value') {
        return 'Local storage working correctly';
      } else {
        throw new Error('Local storage failed');
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
    runSystemTests();
  });
</script>

<div class="container mx-auto py-8 max-w-4xl">
  <div class="mb-8">
    <h1 class="text-4xl font-bold mb-2 flex items-center gap-2">
      🧪 System Testing Dashboard
    </h1>
    <p class="text-lg text-muted-foreground">
      Testing core system components and connectivity
    </p>
  </div>

  <div class="mb-6">
    <button 
      onclick={runSystemTests}
      disabled={isRunning}
      class="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
    >
      {isRunning ? 'Running Tests...' : 'Run System Tests'}
    </button>
  </div>

  <div class="space-y-4">
    {#each testResults as result}
      <div class="p-4 border rounded-lg transition-colors {
        result.status === 'success' ? 'border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-950' :
        result.status === 'error' ? 'border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-950' :
        'border-gray-200 bg-gray-50 dark:border-gray-700 dark:bg-gray-900'
      }">
        <div class="flex items-center justify-between mb-2">
          <h3 class="font-semibold">{result.test}</h3>
          <div class="flex items-center gap-2">
            {#if result.duration}
              <span class="text-sm text-gray-500">{result.duration}ms</span>
            {/if}
            <span class="px-2 py-1 text-xs rounded font-medium {
              result.status === 'success' ? 'bg-green-200 text-green-800 dark:bg-green-800 dark:text-green-200' :
              result.status === 'error' ? 'bg-red-200 text-red-800 dark:bg-red-800 dark:text-red-200' :
              'bg-gray-200 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
            }">
              {result.status.toUpperCase()}
            </span>
          </div>
        </div>
        <p class="text-sm {
          result.status === 'error' ? 'text-red-700 dark:text-red-300' : 'text-gray-700 dark:text-gray-300'
        }">
          {result.message}
        </p>
      </div>
    {/each}
  </div>
  
  {#if testResults.length === 0 && !isRunning}
    <div class="text-center py-12 text-gray-500">
      <div class="text-6xl mb-4">🚀</div>
      <p class="text-lg">Click "Run System Tests" to start testing</p>
    </div>
  {/if}
  
  {#if !isRunning && testResults.length > 0}
    <div class="mt-8 p-4 bg-blue-50 dark:bg-blue-950 rounded-lg">
      <div class="flex items-center gap-2 mb-2">
        <span class="text-2xl">📊</span>
        <h3 class="font-semibold text-blue-900 dark:text-blue-100">Test Summary</h3>
      </div>
      <div class="grid grid-cols-3 gap-4 text-center">
        <div>
          <div class="text-2xl font-bold text-green-600">{testResults.filter(t => t.status === 'success').length}</div>
          <div class="text-sm text-gray-600 dark:text-gray-400">Passed</div>
        </div>
        <div>
          <div class="text-2xl font-bold text-red-600">{testResults.filter(t => t.status === 'error').length}</div>
          <div class="text-sm text-gray-600 dark:text-gray-400">Failed</div>
        </div>
        <div>
          <div class="text-2xl font-bold text-gray-600">{testResults.length}</div>
          <div class="text-sm text-gray-600 dark:text-gray-400">Total</div>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .container {
    font-family: 'Inter', system-ui, sans-serif;
  }
</style>