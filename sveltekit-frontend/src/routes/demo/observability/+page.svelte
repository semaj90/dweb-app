<!-- Observability Integration Demo -->
<script>
  import { onMount } from 'svelte';
  import { browser } from '$app/environment';
  import { 
    getObservabilityStatus, 
    trackCustomEvent, 
    createObservableFetch 
  } from '$lib/utils/observability-init.js';

  let observabilityStatus = $state(null);
  let clientMetrics = $state(null);
  let serverHealth = $state(null);
  let isLoading = $state(false);
  let demoResults = $state([]);

  // Enhanced fetch with observability
  const observableFetch = browser ? createObservableFetch() : fetch;

  onMount(async () => {
    // Initialize observability status
    observabilityStatus = getObservabilityStatus();
    
    // Perform initial health check
    await checkServerHealth();
    
    // Load initial metrics
    await loadClientMetrics();
    
    console.log('🔍 Observability Demo Loaded:', {
      status: observabilityStatus,
      timestamp: new Date().toISOString()
    });
  });

  async function checkServerHealth() {
    if (!browser) return;
    
    isLoading = true;
    const startTime = performance.now();
    
    try {
      const response = await observableFetch('/api/v1/observability/client?action=health');
      const data = await response.json();
      
      serverHealth = data;
      
      const duration = performance.now() - startTime;
      addDemoResult('Health Check', `${data.status} (${Math.round(duration)}ms)`, 'success');
      
      // Track custom event
      trackCustomEvent('health-check-completed', { 
        status: data.status, 
        score: data.score,
        duration 
      });
      
    } catch (error) {
      console.error('Health check failed:', error);
      addDemoResult('Health Check', `Failed: ${error.message}`, 'error');
      serverHealth = { status: 'error', error: error.message };
    } finally {
      isLoading = false;
    }
  }

  async function loadClientMetrics() {
    if (!browser) return;
    
    isLoading = true;
    const startTime = performance.now();
    
    try {
      const response = await observableFetch('/api/v1/observability/client?action=stats');
      const data = await response.json();
      
      clientMetrics = data;
      
      const duration = performance.now() - startTime;
      addDemoResult('Load Metrics', `${data.totalStoredMetrics} metrics (${Math.round(duration)}ms)`, 'info');
      
    } catch (error) {
      console.error('Failed to load metrics:', error);
      addDemoResult('Load Metrics', `Failed: ${error.message}`, 'error');
    } finally {
      isLoading = false;
    }
  }

  async function simulateSlowOperation() {
    isLoading = true;
    const startTime = performance.now();
    
    // Track the start of custom operation
    trackCustomEvent('slow-operation-start');
    
    try {
      // Simulate a slow operation
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Make an API call to test Server-Timing headers
      const response = await observableFetch('/api/v1/observability/client?action=performance');
      const data = await response.json();
      
      const duration = performance.now() - startTime;
      addDemoResult('Slow Operation', `Completed in ${Math.round(duration)}ms`, 'success');
      
      // Track completion
      trackCustomEvent('slow-operation-complete', { duration });
      
      // Trigger metrics update
      await loadClientMetrics();
      
    } catch (error) {
      const duration = performance.now() - startTime;
      addDemoResult('Slow Operation', `Failed after ${Math.round(duration)}ms`, 'error');
    } finally {
      isLoading = false;
    }
  }

  async function clearMetrics() {
    if (!browser) return;
    
    isLoading = true;
    
    try {
      const response = await observableFetch('/api/v1/observability/client?action=clear', {
        method: 'GET'
      });
      const data = await response.json();
      
      addDemoResult('Clear Metrics', data.message || 'Success', 'info');
      
      // Reload metrics
      await loadClientMetrics();
      
    } catch (error) {
      addDemoResult('Clear Metrics', `Failed: ${error.message}`, 'error');
    } finally {
      isLoading = false;
    }
  }

  function addDemoResult(action, result, type) {
    demoResults = [...demoResults, {
      id: Date.now(),
      action,
      result,
      type,
      timestamp: new Date().toLocaleTimeString()
    }];
    
    // Keep only last 10 results
    if (demoResults.length > 10) {
      demoResults = demoResults.slice(-10);
    }
  }

  function getStatusColor(status) {
    switch (status) {
      case 'excellent': return 'text-green-400';
      case 'good': return 'text-blue-400';
      case 'fair': return 'text-yellow-400';
      case 'poor': return 'text-red-400';
      default: return 'text-gray-400';
    }
  }

  function getResultColor(type) {
    switch (type) {
      case 'success': return 'text-green-400';
      case 'error': return 'text-red-400';
      case 'info': return 'text-blue-400';
      default: return 'text-gray-400';
    }
  }
</script>

<div class="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-800 p-6">
  <div class="max-w-6xl mx-auto space-y-6">
    <!-- Header -->
    <div class="text-center mb-8">
      <h1 class="text-4xl font-bold text-yellow-400 mb-4">
        🔍 OBSERVABILITY INTEGRATION DEMO
      </h1>
      <p class="text-gray-300 text-lg">
        Full-stack observability with client-side timing, server-side metrics, and cognitive system integration
      </p>
    </div>

    <!-- Status Cards -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
      <!-- Observability Status -->
      <div class="bg-gray-800 border border-gray-700 rounded-lg p-6">
        <h3 class="text-xl font-semibold text-yellow-400 mb-4">System Status</h3>
        {#if observabilityStatus}
          <div class="space-y-2">
            <div class="flex justify-between">
              <span class="text-gray-400">Initialized:</span>
              <span class="text-{observabilityStatus.initialized ? 'green' : 'red'}-400">
                {observabilityStatus.initialized ? 'Yes' : 'No'}
              </span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-400">Browser:</span>
              <span class="text-{observabilityStatus.browser ? 'green' : 'red'}-400">
                {observabilityStatus.browser ? 'Yes' : 'No'}
              </span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-400">Current Route:</span>
              <span class="text-blue-400 text-sm">
                {observabilityStatus.currentRoute || 'Unknown'}
              </span>
            </div>
          </div>
        {:else}
          <div class="text-gray-400">Loading...</div>
        {/if}
      </div>

      <!-- Server Health -->
      <div class="bg-gray-800 border border-gray-700 rounded-lg p-6">
        <h3 class="text-xl font-semibent text-yellow-400 mb-4">Server Health</h3>
        {#if serverHealth}
          <div class="space-y-2">
            <div class="flex justify-between">
              <span class="text-gray-400">Status:</span>
              <span class="{getStatusColor(serverHealth.status)}">
                {serverHealth.status || 'Unknown'}
              </span>
            </div>
            {#if serverHealth.score !== undefined}
              <div class="flex justify-between">
                <span class="text-gray-400">Score:</span>
                <span class="{getStatusColor(serverHealth.status)}">
                  {Math.round(serverHealth.score)}/100
                </span>
              </div>
            {/if}
            {#if serverHealth.checks}
              <div class="text-sm text-gray-500">
                <div>Load Time: {serverHealth.checks.averageLoadTime ? '✅' : '❌'}</div>
                <div>LCP: {serverHealth.checks.lcpUnder2_5s ? '✅' : '❌'}</div>
              </div>
            {/if}
          </div>
        {:else}
          <div class="text-gray-400">Loading...</div>
        {/if}
      </div>

      <!-- Client Metrics -->
      <div class="bg-gray-800 border border-gray-700 rounded-lg p-6">
        <h3 class="text-xl font-semibold text-yellow-400 mb-4">Client Metrics</h3>
        {#if clientMetrics}
          <div class="space-y-2">
            <div class="flex justify-between">
              <span class="text-gray-400">Total Requests:</span>
              <span class="text-blue-400">
                {clientMetrics.aggregatedStats?.totalRequests || 0}
              </span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-400">Avg Load Time:</span>
              <span class="text-blue-400">
                {clientMetrics.aggregatedStats?.averageLoadTime 
                  ? Math.round(clientMetrics.aggregatedStats.averageLoadTime) + 'ms' 
                  : 'N/A'}
              </span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-400">Stored Metrics:</span>
              <span class="text-green-400">
                {clientMetrics.totalStoredMetrics || 0}
              </span>
            </div>
          </div>
        {:else}
          <div class="text-gray-400">Loading...</div>
        {/if}
      </div>
    </div>

    <!-- Demo Actions -->
    <div class="bg-gray-800 border border-gray-700 rounded-lg p-6">
      <h3 class="text-xl font-semibold text-yellow-400 mb-4">Demo Actions</h3>
      <div class="flex flex-wrap gap-4">
        <button 
          onclick={checkServerHealth}
          disabled={isLoading}
          class="bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 text-white px-4 py-2 rounded-md transition-colors"
        >
          🏥 Check Health
        </button>
        
        <button 
          onclick={loadClientMetrics}
          disabled={isLoading}
          class="bg-green-600 hover:bg-green-500 disabled:bg-gray-600 text-white px-4 py-2 rounded-md transition-colors"
        >
          📊 Load Metrics
        </button>
        
        <button 
          onclick={simulateSlowOperation}
          disabled={isLoading}
          class="bg-yellow-600 hover:bg-yellow-500 disabled:bg-gray-600 text-white px-4 py-2 rounded-md transition-colors"
        >
          🐌 Slow Operation
        </button>
        
        <button 
          onclick={clearMetrics}
          disabled={isLoading}
          class="bg-red-600 hover:bg-red-500 disabled:bg-gray-600 text-white px-4 py-2 rounded-md transition-colors"
        >
          🧹 Clear Metrics
        </button>
      </div>
      
      {#if isLoading}
        <div class="mt-4 flex items-center text-yellow-400">
          <div class="animate-spin rounded-full h-4 w-4 border-b-2 border-yellow-400 mr-2"></div>
          Processing...
        </div>
      {/if}
    </div>

    <!-- Demo Results -->
    {#if demoResults.length > 0}
      <div class="bg-gray-800 border border-gray-700 rounded-lg p-6">
        <h3 class="text-xl font-semibold text-yellow-400 mb-4">Recent Operations</h3>
        <div class="space-y-2 max-h-64 overflow-y-auto">
          {#each demoResults.reverse() as result (result.id)}
            <div class="flex justify-between items-center py-2 border-b border-gray-700 last:border-b-0">
              <div class="flex items-center space-x-3">
                <span class="text-gray-400 text-sm">{result.timestamp}</span>
                <span class="text-white font-medium">{result.action}</span>
              </div>
              <span class="{getResultColor(result.type)} text-sm">
                {result.result}
              </span>
            </div>
          {/each}
        </div>
      </div>
    {/if}

    <!-- Integration Notes -->
    <div class="bg-gray-800 border border-gray-700 rounded-lg p-6">
      <h3 class="text-xl font-semibold text-yellow-400 mb-4">🎯 Integration Features</h3>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h4 class="text-lg font-semibold text-green-400 mb-2">✅ Client-Side</h4>
          <ul class="text-gray-300 space-y-1 text-sm">
            <li>• Timing metrics collection</li>
            <li>• Web Vitals monitoring (LCP, FID, CLS)</li>
            <li>• Performance snapshots</li>
            <li>• Route navigation tracking</li>
            <li>• Custom event tracking</li>
            <li>• Enhanced fetch with timing</li>
          </ul>
        </div>
        <div>
          <h4 class="text-lg font-semibold text-green-400 mb-2">✅ Server-Side</h4>
          <ul class="text-gray-300 space-y-1 text-sm">
            <li>• Request correlation (X-Request-ID)</li>
            <li>• Server-Timing headers</li>
            <li>• Cognitive metrics integration</li>
            <li>• Health scoring algorithm</li>
            <li>• Metrics aggregation</li>
            <li>• Performance monitoring API</li>
          </ul>
        </div>
      </div>
    </div>

    <!-- Footer -->
    <div class="text-center text-gray-500 text-sm">
      <p>
        🔍 Observability System Integrated • 
        Server: <span class="text-yellow-400">Port 5181</span> • 
        API: <span class="text-blue-400">/api/v1/observability/client</span>
      </p>
    </div>
  </div>
</div>

<style>
  /* Custom scrollbar for results */
  ::-webkit-scrollbar {
    width: 6px;
  }
  
  ::-webkit-scrollbar-track {
    background: #374151;
  }
  
  ::-webkit-scrollbar-thumb {
    background: #6B7280;
    border-radius: 3px;
  }
  
  ::-webkit-scrollbar-thumb:hover {
    background: #9CA3AF;
  }
</style>