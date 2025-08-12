<script lang="ts">
  import { onMount } from 'svelte';
  
  // Props
  let { health = {}, metrics = {} } = $props();
  
  // Derived values for display
  let systemStatus = $derived(health.components ? 'Online' : 'Offline');
  let totalComponents = $derived(Object.keys(health.components || {}).length);
  let activeComponents = $derived(Object.values(health.components || {}).filter(status => status === 'active').length);
  let cpuUsage = $derived(Math.round((health.performance?.cpuUsage?.user || 0) / 10000)); // Convert microseconds to percentage
  let memoryUsage = $derived(Math.round((health.performance?.memoryUsage?.heapUsed || 0) / 1024 / 1024)); // Convert to MB
  let uptime = $derived(Math.round((health.system?.uptime || 0) / 3600)); // Convert to hours
  
  // Component status colors
  function getStatusColor(status: string) {
    switch (status) {
      case 'active': return 'text-green-400';
      case 'inactive': return 'text-yellow-400';
      case 'error': return 'text-red-400';
      default: return 'text-gray-400';
    }
  }
  
  function getStatusBg(status: string) {
    switch (status) {
      case 'active': return 'bg-green-400/20 border-green-400/30';
      case 'inactive': return 'bg-yellow-400/20 border-yellow-400/30';
      case 'error': return 'bg-red-400/20 border-red-400/30';
      default: return 'bg-gray-400/20 border-gray-400/30';
    }
  }
  
  // Health score calculation
  let healthScore = $derived(() => {
    if (!health.components) return 0;
    
    const total = totalComponents;
    const active = activeComponents;
    const cpuScore = Math.max(0, 100 - cpuUsage);
    const memoryScore = Math.max(0, 100 - Math.min(memoryUsage / 10, 100)); // Assume 1GB is 100%
    
    const componentScore = total > 0 ? (active / total) * 100 : 0;
    return Math.round((componentScore + cpuScore + memoryScore) / 3);
  });
</script>

<div class="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-6">
  <div class="flex items-center justify-between mb-6">
    <h2 class="text-xl font-semibold text-white">🏥 System Health Dashboard</h2>
    
    <!-- Health Score Badge -->
    <div class="flex items-center gap-3">
      <div class="flex items-center gap-2">
        <div 
          class="w-3 h-3 rounded-full"
          class:bg-green-400={healthScore >= 80}
          class:bg-yellow-400={healthScore >= 60 && healthScore < 80}
          class:bg-red-400={healthScore < 60}
        ></div>
        <span class="text-white font-medium">Health Score: {healthScore}%</span>
      </div>
    </div>
  </div>

  <!-- Key Metrics Grid -->
  <div class="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
    <!-- System Status -->
    <div class="bg-slate-700/50 border border-slate-600 rounded-lg p-4">
      <div class="flex items-center gap-2 mb-2">
        <span class="text-2xl">🖥️</span>
        <span class="text-slate-300 text-sm">System</span>
      </div>
      <div class="text-white font-semibold">{systemStatus}</div>
      <div class="text-slate-400 text-xs">Uptime: {uptime}h</div>
    </div>

    <!-- CPU Usage -->
    <div class="bg-slate-700/50 border border-slate-600 rounded-lg p-4">
      <div class="flex items-center gap-2 mb-2">
        <span class="text-2xl">⚡</span>
        <span class="text-slate-300 text-sm">CPU</span>
      </div>
      <div class="text-white font-semibold">{cpuUsage}%</div>
      <div class="w-full bg-slate-600 rounded-full h-2 mt-2">
        <div 
          class="h-2 rounded-full transition-all duration-300"
          class:bg-green-400={cpuUsage < 70}
          class:bg-yellow-400={cpuUsage >= 70 && cpuUsage < 85}
          class:bg-red-400={cpuUsage >= 85}
          style="width: {Math.min(cpuUsage, 100)}%"
        ></div>
      </div>
    </div>

    <!-- Memory Usage -->
    <div class="bg-slate-700/50 border border-slate-600 rounded-lg p-4">
      <div class="flex items-center gap-2 mb-2">
        <span class="text-2xl">💾</span>
        <span class="text-slate-300 text-sm">Memory</span>
      </div>
      <div class="text-white font-semibold">{memoryUsage} MB</div>
      <div class="w-full bg-slate-600 rounded-full h-2 mt-2">
        <div 
          class="h-2 rounded-full transition-all duration-300"
          class:bg-green-400={memoryUsage < 512}
          class:bg-yellow-400={memoryUsage >= 512 && memoryUsage < 1024}
          class:bg-red-400={memoryUsage >= 1024}
          style="width: {Math.min((memoryUsage / 2048) * 100, 100)}%"
        ></div>
      </div>
    </div>

    <!-- Active Components -->
    <div class="bg-slate-700/50 border border-slate-600 rounded-lg p-4">
      <div class="flex items-center gap-2 mb-2">
        <span class="text-2xl">🧩</span>
        <span class="text-slate-300 text-sm">Components</span>
      </div>
      <div class="text-white font-semibold">{activeComponents}/{totalComponents}</div>
      <div class="text-slate-400 text-xs">Active services</div>
    </div>
  </div>

  <!-- Component Status Grid -->
  {#if health.components && Object.keys(health.components).length > 0}
    <div class="mb-6">
      <h3 class="text-lg font-medium text-white mb-4">🔧 Component Status</h3>
      <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
        {#each Object.entries(health.components) as [name, status]}
          <div class="border rounded-lg p-3 {getStatusBg(status)}">
            <div class="flex items-center gap-2">
              <div class="w-2 h-2 rounded-full {status === 'active' ? 'bg-green-400' : status === 'inactive' ? 'bg-yellow-400' : 'bg-red-400'}"></div>
              <span class="text-white text-sm font-medium capitalize">
                {name.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
              </span>
            </div>
            <div class="text-xs {getStatusColor(status)} mt-1 capitalize">{status}</div>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Performance Metrics -->
  {#if metrics && Object.keys(metrics).length > 0}
    <div>
      <h3 class="text-lg font-medium text-white mb-4">📊 Performance Metrics</h3>
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        
        <!-- Cache Performance -->
        {#if metrics.cache}
          <div class="bg-slate-700/30 border border-slate-600 rounded-lg p-4">
            <div class="flex items-center gap-2 mb-3">
              <span class="text-xl">🗄️</span>
              <span class="text-white font-medium">Cache Performance</span>
            </div>
            <div class="space-y-2 text-sm">
              <div class="flex justify-between">
                <span class="text-slate-300">Hit Rate:</span>
                <span class="text-green-400">{Math.round(metrics.cache.hitRate || 0)}%</span>
              </div>
              <div class="flex justify-between">
                <span class="text-slate-300">Size:</span>
                <span class="text-slate-300">{Math.round((metrics.cache.size || 0) / 1024 / 1024)} MB</span>
              </div>
            </div>
          </div>
        {/if}

        <!-- Worker Threads -->
        <div class="bg-slate-700/30 border border-slate-600 rounded-lg p-4">
          <div class="flex items-center gap-2 mb-3">
            <span class="text-xl">👷</span>
            <span class="text-white font-medium">Worker Threads</span>
          </div>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span class="text-slate-300">Active:</span>
              <span class="text-blue-400">{metrics.workers || 0}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-slate-300">Queue:</span>
              <span class="text-slate-300">{health.performance?.queueDepths?.total || 0}</span>
            </div>
          </div>
        </div>

        <!-- Response Times -->
        {#if metrics.averageResponseTime}
          <div class="bg-slate-700/30 border border-slate-600 rounded-lg p-4">
            <div class="flex items-center gap-2 mb-3">
              <span class="text-xl">⏱️</span>
              <span class="text-white font-medium">Response Times</span>
            </div>
            <div class="space-y-2 text-sm">
              <div class="flex justify-between">
                <span class="text-slate-300">Average:</span>
                <span class="text-blue-400">{Math.round(metrics.averageResponseTime)}ms</span>
              </div>
              <div class="flex justify-between">
                <span class="text-slate-300">P95:</span>
                <span class="text-slate-300">{Math.round(metrics.p95ResponseTime || metrics.averageResponseTime * 1.5)}ms</span>
              </div>
            </div>
          </div>
        {/if}
      </div>
    </div>
  {/if}

  <!-- System Alerts -->
  {#if cpuUsage > 85 || memoryUsage > 1024 || activeComponents < totalComponents}
    <div class="mt-6 p-4 bg-red-900/20 border border-red-700 rounded-lg">
      <div class="flex items-center gap-2 mb-2">
        <span class="text-xl">⚠️</span>
        <span class="text-red-400 font-medium">System Alerts</span>
      </div>
      <div class="space-y-1 text-sm">
        {#if cpuUsage > 85}
          <div class="text-red-300">• High CPU usage detected ({cpuUsage}%)</div>
        {/if}
        {#if memoryUsage > 1024}
          <div class="text-red-300">• High memory usage detected ({memoryUsage} MB)</div>
        {/if}
        {#if activeComponents < totalComponents}
          <div class="text-red-300">• {totalComponents - activeComponents} component(s) offline</div>
        {/if}
      </div>
    </div>
  {/if}
</div>

<style>
  /* Add smooth transitions for all metrics */
  .transition-all {
    transition: all 0.3s ease;
  }
</style>