<script lang="ts">
  interface Props {
    showDetails?: any;
  }
  let {
    showDetails = false
  } = $props();



  import { onMount, onDestroy } from 'svelte';
  import { memoryMonitoring } from '$lib/services/memory-monitoring.service';
  
    
  let memoryData = {
    currentLOD: { name: 'medium', level: 2 },
    memoryPressure: 0.5,
    pools: [],
    clusters: [],
    cacheLayers: []
  };
  
  let updateCount = 0;
  let isOptimizing = false;

  onMount(() => {
    memoryMonitoring.start(10000); // Update every 10 seconds
    
    memoryMonitoring.onUpdate((data) => {
      memoryData = data;
      updateCount++;
    });
  });

  onDestroy(() => {
    memoryMonitoring.stop();
  });

  async function triggerOptimization() {
    isOptimizing = true;
    try {
      const success = await memoryMonitoring.triggerOptimization();
      if (success) {
        console.log('✅ Optimization triggered successfully');
      }
    } catch (error) {
      console.error('❌ Optimization failed:', error);
    } finally {
      isOptimizing = false;
    }
  }

  function getMemoryPressureColor(pressure: number): string {
    if (pressure > 0.9) return 'text-red-600';
    if (pressure > 0.7) return 'text-yellow-600';
    return 'text-green-600';
  }

  function formatBytes(bytes: number): string {
    const sizes = ['B', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 B';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  }
</script>

<div class="memory-monitor bg-white border rounded-lg p-4 shadow-sm">
  <div class="flex items-center justify-between mb-4">
    <h3 class="text-lg font-semibold">Memory Monitor</h3>
    <div class="flex items-center gap-2">
      <div class="text-xs text-gray-500">Updates: {updateCount}</div>
      <button 
        class="optimize-btn px-3 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
        onclick={triggerOptimization}
        disabled={isOptimizing}
      >
        {isOptimizing ? 'Optimizing...' : 'Optimize'}
      </button>
    </div>
  </div>

  <!-- Key Metrics -->
  <div class="grid grid-cols-3 gap-4 mb-4">
    <div class="metric">
      <div class="text-xs text-gray-500">LOD Level</div>
      <div class="text-lg font-bold">{memoryData.currentLOD.name}</div>
    </div>
    
    <div class="metric">
      <div class="text-xs text-gray-500">Memory Pressure</div>
      <div class="text-lg font-bold {getMemoryPressureColor(memoryData.memoryPressure)}">
        {(memoryData.memoryPressure * 100).toFixed(1)}%
      </div>
    </div>
    
    <div class="metric">
      <div class="text-xs text-gray-500">Active Clusters</div>
      <div class="text-lg font-bold">{memoryData.clusters.length}</div>
    </div>
  </div>

  <!-- Memory Pools -->
  {#if showDetails && memoryData.pools.length > 0}
    <div class="pools mb-4">
      <h4 class="font-semibold mb-2">Memory Pools</h4>
      <div class="space-y-2">
        {#each memoryData.pools as pool}
          <div class="pool-item flex justify-between items-center text-sm">
            <span class="font-medium">{pool.id}</span>
            <div class="flex items-center gap-2">
              <div class="usage-bar w-20 h-2 bg-gray-200 rounded">
                <div 
                  class="usage-fill h-full bg-blue-600 rounded"
                  style="width: {pool.percentage}%"
                ></div>
              </div>
              <span class="text-xs text-gray-500">{pool.percentage.toFixed(1)}%</span>
            </div>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Cache Layers -->
  {#if showDetails && memoryData.cacheLayers.length > 0}
    <div class="cache-layers">
      <h4 class="font-semibold mb-2">Cache Layers</h4>
      <div class="grid grid-cols-2 gap-2 text-xs">
        {#each memoryData.cacheLayers as layer}
          <div class="layer-item p-2 bg-gray-50 rounded">
            <div class="font-medium">{layer.name}</div>
            <div class="text-gray-600">Hit Rate: {(layer.hitRate * 100).toFixed(1)}%</div>
          </div>
        {/each}
      </div>
    </div>
  {/if}
</div>

<style>
  .usage-fill {
    transition: width 0.3s ease;
  }
</style>