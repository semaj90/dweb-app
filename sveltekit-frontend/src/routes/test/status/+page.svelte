<!-- Route Status Check - Verify All Routes Work -->
<script>
  import { onMount } from 'svelte';
  import Button from '$lib/components/ui/Button.svelte';
let routeTests = $state([
    { name: 'Test Hub', path: '/test', status: 'pending', responseTime: null },
    { name: 'CRUD Interface', path: '/test/crud', status: 'pending', responseTime: null },
    { name: 'CRUD API Health', path: '/test/crud', method: 'GET', headers: {'Accept': 'application/json'}, status: 'pending', responseTime: null },
    { name: 'Route Status', path: '/test/status', status: 'pending', responseTime: null }
  ]);
let isTestingRoutes = $state(false);
  
  const testRoute = async (test) => {
    const startTime = Date.now();
    try {
      const options = {
        method: test.method || 'GET',
        headers: test.headers || {}
      };
      
      const response = await fetch(test.path, options);
      const endTime = Date.now();
      
      return {
        ...test,
        status: response.ok ? 'success' : `error-${response.status}`,
        responseTime: endTime - startTime,
        statusCode: response.status
      };
    } catch (error) {
      const endTime = Date.now();
      return {
        ...test,
        status: 'failed',
        responseTime: endTime - startTime,
        error: error.message
      };
    }
  };
  
  const runAllRouteTests = async () => {
    if (isTestingRoutes) return;
    
    isTestingRoutes = true;
    
    // Reset all to pending
    routeTests = routeTests.map(test => ({ ...test, status: 'pending', responseTime: null }));
    
    // Test each route
    for (let i = 0; i < routeTests.length; i++) {
      const result = await testRoute(routeTests[i]);
      routeTests[i] = result;
      routeTests = [...routeTests]; // trigger reactivity
    }
    
    isTestingRoutes = false;
  };
  
  const getStatusColor = (status) => {
    if (status === 'success') return 'text-green-600';
    if (status === 'pending') return 'text-yellow-600';
    if (status.startsWith('error-')) return 'text-orange-600';
    if (status === 'failed') return 'text-red-600';
    return 'text-gray-600';
  };
  
  const getStatusIcon = (status) => {
    if (status === 'success') return '✅';
    if (status === 'pending') return '⏳';
    if (status.startsWith('error-')) return '⚠️';
    if (status === 'failed') return '❌';
    return '❓';
  };
  
  onMount(() => {
    // Auto-run tests on page load
    runAllRouteTests();
  });
</script>

<svelte:head>
  <title>Route Status Check</title>
</svelte:head>

<div class="container mx-auto p-6 space-y-6">
  <div class="text-center">
    <h1 class="text-3xl font-bold">🛣️ Route Status Check</h1>
    <p class="text-gray-600 mt-2">Verifying all test routes are accessible and working</p>
  </div>
  
  <div class="flex justify-center gap-4">
    <Button 
      on:on:on:click={runAllRouteTests}
      disabled={isTestingRoutes}
      variant="default"
    >
      {isTestingRoutes ? '🔄 Testing...' : '🚀 Test All Routes'}
    </Button>
    
    <Button 
      on:on:on:click={() => window.location.href = '/test'}
      variant="outline"
    >
      ← Back to Test Hub
    </Button>
  </div>

  <!-- Route Test Results -->
  <div class="space-y-4">
    {#each routeTests as test}
      <div class="border rounded-lg p-4 {test.status === 'success' ? 'border-green-200 bg-green-50' : test.status === 'failed' ? 'border-red-200 bg-red-50' : test.status.startsWith('error-') ? 'border-orange-200 bg-orange-50' : 'border-gray-200'}">
        <div class="flex justify-between items-center">
          <div>
            <h3 class="font-semibold flex items-center gap-2">
              {getStatusIcon(test.status)}
              {test.name}
            </h3>
            <div class="text-sm text-gray-600 flex gap-4">
              <span>Path: <code class="bg-gray-100 px-1 rounded">{test.path}</code></span>
              {#if test.method && test.method !== 'GET'}
                <span>Method: <code class="bg-gray-100 px-1 rounded">{test.method}</code></span>
              {/if}
              {#if test.responseTime !== null}
                <span>Response: <strong>{test.responseTime}ms</strong></span>
              {/if}
            </div>
          </div>
          
          <div class="text-right">
            <div class="font-semibold {getStatusColor(test.status)}">
              {test.status === 'success' ? 'OK' : 
               test.status === 'pending' ? 'Testing...' :
               test.status.startsWith('error-') ? `HTTP ${test.statusCode}` :
               'Failed'}
            </div>
            {#if test.error}
              <div class="text-xs text-red-600 mt-1 max-w-xs truncate">
                {test.error}
              </div>
            {/if}
          </div>
        </div>
      </div>
    {/each}
  </div>

  <!-- Test Summary -->
  <div class="border-t pt-6">
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
      <div>
        <div class="text-2xl font-bold text-green-600">
          {routeTests.filter(t => t.status === 'success').length}
        </div>
        <div class="text-sm text-gray-600">Successful</div>
      </div>
      <div>
        <div class="text-2xl font-bold text-orange-600">
          {routeTests.filter(t => t.status.startsWith('error-')).length}
        </div>
        <div class="text-sm text-gray-600">Errors</div>
      </div>
      <div>
        <div class="text-2xl font-bold text-red-600">
          {routeTests.filter(t => t.status === 'failed').length}
        </div>
        <div class="text-sm text-gray-600">Failed</div>
      </div>
      <div>
        <div class="text-2xl font-bold text-yellow-600">
          {routeTests.filter(t => t.status === 'pending').length}
        </div>
        <div class="text-sm text-gray-600">Pending</div>
      </div>
    </div>
  </div>

  <!-- Notes -->
  <div class="bg-blue-50 border border-blue-200 rounded-lg p-4">
    <h4 class="font-semibold text-blue-800 mb-2">📝 Test Notes</h4>
    <ul class="text-sm text-blue-700 space-y-1">
      <li>• <strong>HTTP 503</strong> on CRUD routes indicates database connection issues (expected behavior)</li>
      <li>• <strong>HTTP 500</strong> may indicate missing dependencies or configuration issues</li>
      <li>• Routes should be accessible even when services are unavailable</li>
      <li>• SSR functionality is verified by successful page loads</li>
    </ul>
  </div>
</div>

<style>
  .container {
    max-width: 4xl;
  }
</style>