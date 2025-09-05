<script lang="ts">
  import { onMount } from 'svelte';
  import { browser } from '$app/environment';
  
  let systemStatus = null;
  let memoryPrediction = null;
  let isLoading = true;
  let error = null;

  onMount(async () => {
    if (browser) {
      await loadData();
      
      // Refresh data every 30 seconds
      setInterval(loadData, 30000);
    }
  });

  async function loadData() {
    try {
      error = null;
      
      // Fetch system status and memory prediction
      const [statusResponse, predictionResponse] = await Promise.all([
        fetch('/api/memory/neural?action=status'),
        fetch('/api/memory/neural?action=predict&horizon=30')
      ]);

      if (statusResponse.ok) {
        const statusResult = await statusResponse.json();
        systemStatus = statusResult.success ? statusResult.data: null
      }

      if (predictionResponse.ok) {
        const predictionResult = await predictionResponse.json();
        memoryPrediction = predictionResult.success ? predictionResult.data: null
      }
    } catch (err) {
      console.error('Failed to load memory data:', err);
      error = err.message;
    } finally {
      isLoading = false;
    }
  }

  async function triggerOptimization() {
    isLoading = true;
    try {
      const response = await fetch('/api/memory/neural?action=optimize');
      const result = await response.json();
      
      if (result.success) {
        // Reload data after optimization
        await loadData();
      } else {
        error = result.error || 'Optimization failed';
      }
    } catch (err) {
      console.error('Optimization failed:', err);
      error = err.message;
    } finally {
      isLoading = false;
    }
  }

  function getHealthColor(value: number): string {
    if (value >= 0.8) return 'text-green-600';
    if (value >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  }

  function formatBytes(bytes: number): string {
    const sizes = ['B', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 B';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  }
</script>

<svelte:head>
  <title>Memory Optimization Dashboard</title>
</svelte:head>

<div class="container mx-auto p-6 space-y-6">
  <div class="flex items-center justify-between">
    <h1 class="text-3xl font-bold text-gray-900">Memory Optimization Dashboard</h1>
    <div class="flex items-center gap-4">
      {#if isLoading}
        <div class="text-sm text-gray-500">Loading...</div>
      {/if}
      <button 
        class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        onclick={triggerOptimization}
        disabled={isLoading}
      >
        {isLoading ? 'Optimizing...' : 'Run Optimization'}
      </button>
    </div>
  </div>

  {#if error}
    <div class="bg-red-50 border border-red-200 rounded-lg p-4">
      <div class="flex">
        <div class="text-red-800">
          <h3 class="text-sm font-medium">Error</h3>
          <p class="mt-1 text-sm">{error}</p>
        </div>
      </div>
    </div>
  {/if}

  {#if systemStatus}
    <!-- System Overview -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
      <div class="bg-white rounded-lg border p-6">
        <h3 class="text-lg font-semibold mb-3">Memory Efficiency</h3>
        <div class="text-3xl font-bold {getHealthColor(systemStatus.memoryEfficiency)}">
          {(systemStatus.memoryEfficiency * 100).toFixed(1)}%
        </div>
        <p class="text-sm text-gray-600 mt-1">System efficiency</p>
      </div>

      <div class="bg-white rounded-lg border p-6">
        <h3 class="text-lg font-semibold mb-3">LOD Level</h3>
        <div class="text-3xl font-bold text-blue-600">
          {systemStatus.lodLevel?.name || 'medium'}
        </div>
        <p class="text-sm text-gray-600 mt-1">
          Level {systemStatus.lodLevel?.level || 2}
        </p>
      </div>

      <div class="bg-white rounded-lg border p-6">
        <h3 class="text-lg font-semibold mb-3">Active Clusters</h3>
        <div class="text-3xl font-bold text-purple-600">
          {systemStatus.clusterCount || 0}
        </div>
        <p class="text-sm text-gray-600 mt-1">Memory clusters</p>
      </div>
    </div>

    <!-- Memory Pool Utilization -->
    {#if systemStatus.poolUtilization}
      <div class="bg-white rounded-lg border p-6">
        <h3 class="text-xl font-semibold mb-4">Memory Pool Utilization</h3>
        <div class="space-y-4">
          {#each Object.entries(systemStatus.poolUtilization) as [poolName, utilizationValue]}
            {@const utilization = Number(utilizationValue)}
            <div class="flex items-center justify-between">
              <span class="font-medium capitalize">{poolName}</span>
              <div class="flex items-center gap-3">
                <div class="w-32 h-3 bg-gray-200 rounded-full overflow-hidden">
                  <div 
                    class="h-full transition-all duration-300 {utilization > 0.8 ? 'bg-red-500' : utilization > 0.6 ? 'bg-yellow-500' : 'bg-green-500'}"
                    style="width: {Math.min(100, utilization * 100)}%"
                  ></div>
                </div>
                <span class="text-sm font-mono {getHealthColor(1 - utilization)}">
                  {(utilization * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          {/each}
        </div>
      </div>
    {/if}
  {/if}

  {#if memoryPrediction}
    <!-- Memory Prediction -->
    <div class="bg-white rounded-lg border p-6">
      <h3 class="text-xl font-semibold mb-4">Memory Prediction (30 minutes)</h3>
      
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <div>
          <h4 class="font-medium mb-2">Expected Usage</h4>
          <div class="text-2xl font-bold">
            {formatBytes(memoryPrediction.expectedUsage)}
          </div>
        </div>
        
        <div>
          <h4 class="font-medium mb-2">Confidence</h4>
          <div class="text-2xl font-bold {getHealthColor(memoryPrediction.confidence)}">
            {(memoryPrediction.confidence * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      {#if memoryPrediction.recommendations?.length > 0}
        <div>
          <h4 class="font-medium mb-3">Recommendations</h4>
          <ul class="space-y-2">
            {#each memoryPrediction.recommendations as recommendation}
              <li class="flex items-start gap-2">
                <div class="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                <span class="text-sm text-gray-700">{recommendation}</span>
              </li>
            {/each}
          </ul>
        </div>
      {/if}

      {#if memoryPrediction.optimizations?.length > 0}
        <div class="mt-6">
          <h4 class="font-medium mb-3">Suggested Optimizations</h4>
          <div class="space-y-3">
            {#each memoryPrediction.optimizations as optimization}
              <div class="bg-gray-50 rounded-lg p-3">
                <div class="flex items-center justify-between mb-1">
                  <span class="font-medium capitalize">{optimization.type}</span>
                  <span class="text-sm px-2 py-1 rounded-full bg-blue-100 text-blue-800">
                    Priority: {optimization.priority}
                  </span>
                </div>
                <div class="text-sm text-gray-600">
                  Estimated savings: {formatBytes(optimization.estimatedSavings)}
                </div>
              </div>
            {/each}
          </div>
        </div>
      {/if}
    </div>
  {/if}

  <!-- Quick Actions -->
  <div class="bg-white rounded-lg border p-6">
    <h3 class="text-xl font-semibold mb-4">Quick Actions</h3>
    <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
      <button 
        class="p-4 border border-gray-300 rounded-lg hover:bg-gray-50 text-left transition-colors"
        onclick={loadData}
        disabled={isLoading}
      >
        <div class="font-medium mb-1">Refresh Data</div>
        <div class="text-sm text-gray-600">Update all metrics</div>
      </button>
      
      <button 
        class="p-4 border border-blue-300 rounded-lg hover:bg-blue-50 text-left transition-colors"
        onclick={triggerOptimization}
        disabled={isLoading}
      >
        <div class="font-medium mb-1 text-blue-700">Run Optimization</div>
        <div class="text-sm text-gray-600">Optimize memory usage</div>
      </button>
      
      <button 
        class="p-4 border border-gray-300 rounded-lg hover:bg-gray-50 text-left transition-colors"
        disabled
      >
        <div class="font-medium mb-1">System Benchmark</div>
        <div class="text-sm text-gray-600">Coming soon</div>
      </button>
    </div>
  </div>

  <!-- Footer Info -->
  <div class="text-center text-sm text-gray-500">
    <p>Memory Optimization Dashboard - Legal AI System</p>
    <p>Last updated: {new Date().toLocaleString()}</p>
  </div>
</div>