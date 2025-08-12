<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  
  // Props
  let { metrics = {}, health = {} } = $props();
  
  // Chart data state
  let cpuData = $state<number[]>([]);
  let memoryData = $state<number[]>([]);
  let timestamps = $state<string[]>([]);
  let maxDataPoints = 50;
  
  // Current values
  let currentCPU = $derived(Math.round((health.performance?.cpuUsage?.user || 0) / 10000));
  let currentMemory = $derived(Math.round((health.performance?.memoryUsage?.heapUsed || 0) / 1024 / 1024));
  let currentWorkers = $derived(health.performance?.activeWorkers || 0);
  let cacheHitRate = $derived(Math.round(metrics.cache?.hitRate || 0));
  
  // Performance indicators
  let cpuStatus = $derived(currentCPU < 70 ? 'good' : currentCPU < 85 ? 'warning' : 'critical');
  let memoryStatus = $derived(currentMemory < 512 ? 'good' : currentMemory < 1024 ? 'warning' : 'critical');
  let overallStatus = $derived(cpuStatus === 'good' && memoryStatus === 'good' ? 'optimal' : 
                              cpuStatus === 'critical' || memoryStatus === 'critical' ? 'critical' : 'warning');
  
  // Update chart data when metrics change
  $effect(() => {
    if (Object.keys(metrics).length > 0) {
      updateChartData();
    }
  });
  
  function updateChartData() {
    const now = new Date().toLocaleTimeString();
    
    // Update CPU data
    cpuData = [...cpuData, currentCPU].slice(-maxDataPoints);
    
    // Update Memory data
    memoryData = [...memoryData, currentMemory].slice(-maxDataPoints);
    
    // Update timestamps
    timestamps = [...timestamps, now].slice(-maxDataPoints);
  }
  
  function getStatusColor(status: string) {
    switch (status) {
      case 'good': case 'optimal': return 'text-green-400';
      case 'warning': return 'text-yellow-400';
      case 'critical': return 'text-red-400';
      default: return 'text-gray-400';
    }
  }
  
  function getStatusBg(status: string) {
    switch (status) {
      case 'good': case 'optimal': return 'bg-green-400/20 border-green-400/30';
      case 'warning': return 'bg-yellow-400/20 border-yellow-400/30';
      case 'critical': return 'bg-red-400/20 border-red-400/30';
      default: return 'bg-gray-400/20 border-gray-400/30';
    }
  }
  
  // Mini chart SVG generation
  function generateMiniChart(data: number[], maxValue: number = 100) {
    if (data.length < 2) return '';
    
    const width = 120;
    const height = 40;
    const padding = 2;
    
    const maxY = Math.max(maxValue, Math.max(...data));
    const stepX = (width - padding * 2) / (data.length - 1);
    
    let path = '';
    
    data.forEach((value, index) => {
      const x = padding + index * stepX;
      const y = height - padding - ((value / maxY) * (height - padding * 2));
      
      if (index === 0) {
        path += `M ${x} ${y}`;
      } else {
        path += ` L ${x} ${y}`;
      }
    });
    
    return path;
  }
  
  // Performance recommendations
  let recommendations = $derived(() => {
    const recs = [];
    
    if (currentCPU > 85) {
      recs.push({
        type: 'cpu',
        level: 'critical',
        message: 'CPU usage critical - consider scaling worker threads',
        action: 'Scale workers'
      });
    } else if (currentCPU > 70) {
      recs.push({
        type: 'cpu',
        level: 'warning',
        message: 'CPU usage high - monitor performance',
        action: 'Monitor'
      });
    }
    
    if (currentMemory > 1024) {
      recs.push({
        type: 'memory',
        level: 'critical',
        message: 'Memory usage critical - clear caches',
        action: 'Clear cache'
      });
    } else if (currentMemory > 512) {
      recs.push({
        type: 'memory',
        level: 'warning',
        message: 'Memory usage high - optimize allocations',
        action: 'Optimize'
      });
    }
    
    if (cacheHitRate < 70) {
      recs.push({
        type: 'cache',
        level: 'warning',
        message: 'Low cache hit rate - review caching strategy',
        action: 'Review cache'
      });
    }
    
    return recs;
  });
</script>

<div class="space-y-6">
  <!-- Overall Status Banner -->
  <div class="border rounded-xl p-4 {getStatusBg(overallStatus)}">
    <div class="flex items-center justify-between">
      <div class="flex items-center gap-3">
        <div class="text-3xl">
          {#if overallStatus === 'optimal'}🟢
          {:else if overallStatus === 'warning'}🟡
          {:else}🔴{/if}
        </div>
        <div>
          <h3 class="text-lg font-semibold text-white capitalize">System {overallStatus}</h3>
          <p class="text-sm {getStatusColor(overallStatus)}">
            {overallStatus === 'optimal' ? 'All systems operating efficiently' :
             overallStatus === 'warning' ? 'Performance monitoring required' :
             'Immediate attention needed'}
          </p>
        </div>
      </div>
      <div class="text-right">
        <div class="text-2xl font-bold text-white">{Math.round((cpuData.reduce((a, b) => a + b, 0) / Math.max(cpuData.length, 1)) * 10) / 10}%</div>
        <div class="text-sm text-slate-400">Avg CPU</div>
      </div>
    </div>
  </div>

  <!-- Real-time Metrics Grid -->
  <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
    <!-- CPU Monitor -->
    <div class="bg-slate-700/50 border border-slate-600 rounded-lg p-4">
      <div class="flex items-center justify-between mb-3">
        <div class="flex items-center gap-2">
          <span class="text-xl">⚡</span>
          <span class="text-white font-medium">CPU Usage</span>
        </div>
        <span class="text-2xl font-bold {getStatusColor(cpuStatus)}">{currentCPU}%</span>
      </div>
      
      {#if cpuData.length > 1}
        <svg class="w-full h-10" viewBox="0 0 120 40">
          <path 
            d="{generateMiniChart(cpuData)}" 
            stroke="currentColor" 
            stroke-width="2" 
            fill="none"
            class="{getStatusColor(cpuStatus)}"
          />
        </svg>
      {/if}
      
      <div class="flex justify-between text-xs text-slate-400 mt-2">
        <span>Min: {Math.min(...cpuData) || 0}%</span>
        <span>Max: {Math.max(...cpuData) || 0}%</span>
      </div>
    </div>

    <!-- Memory Monitor -->
    <div class="bg-slate-700/50 border border-slate-600 rounded-lg p-4">
      <div class="flex items-center justify-between mb-3">
        <div class="flex items-center gap-2">
          <span class="text-xl">💾</span>
          <span class="text-white font-medium">Memory</span>
        </div>
        <span class="text-2xl font-bold {getStatusColor(memoryStatus)}">{currentMemory}MB</span>
      </div>
      
      {#if memoryData.length > 1}
        <svg class="w-full h-10" viewBox="0 0 120 40">
          <path 
            d="{generateMiniChart(memoryData, 2048)}" 
            stroke="currentColor" 
            stroke-width="2" 
            fill="none"
            class="{getStatusColor(memoryStatus)}"
          />
        </svg>
      {/if}
      
      <div class="flex justify-between text-xs text-slate-400 mt-2">
        <span>Min: {Math.min(...memoryData) || 0}MB</span>
        <span>Max: {Math.max(...memoryData) || 0}MB</span>
      </div>
    </div>

    <!-- Worker Threads -->
    <div class="bg-slate-700/50 border border-slate-600 rounded-lg p-4">
      <div class="flex items-center justify-between mb-3">
        <div class="flex items-center gap-2">
          <span class="text-xl">👷</span>
          <span class="text-white font-medium">Workers</span>
        </div>
        <span class="text-2xl font-bold text-blue-400">{currentWorkers}</span>
      </div>
      
      <div class="space-y-2">
        <div class="flex justify-between text-sm">
          <span class="text-slate-300">Active:</span>
          <span class="text-blue-400">{currentWorkers}</span>
        </div>
        <div class="flex justify-between text-sm">
          <span class="text-slate-300">Queue:</span>
          <span class="text-slate-300">{health.performance?.queueDepths?.total || 0}</span>
        </div>
      </div>
    </div>

    <!-- Cache Performance -->
    <div class="bg-slate-700/50 border border-slate-600 rounded-lg p-4">
      <div class="flex items-center justify-between mb-3">
        <div class="flex items-center gap-2">
          <span class="text-xl">🗄️</span>
          <span class="text-white font-medium">Cache</span>
        </div>
        <span class="text-2xl font-bold text-green-400">{cacheHitRate}%</span>
      </div>
      
      <div class="space-y-2">
        <div class="w-full bg-slate-600 rounded-full h-2">
          <div 
            class="h-2 rounded-full bg-green-400 transition-all duration-300"
            style="width: {cacheHitRate}%"
          ></div>
        </div>
        <div class="flex justify-between text-xs text-slate-400">
          <span>Hit Rate</span>
          <span>{cacheHitRate}%</span>
        </div>
      </div>
    </div>
  </div>

  <!-- Performance Timeline Chart -->
  {#if cpuData.length > 5}
    <div class="bg-slate-700/50 border border-slate-600 rounded-lg p-4">
      <h3 class="text-lg font-medium text-white mb-4">📈 Performance Timeline</h3>
      
      <div class="relative">
        <svg class="w-full h-40" viewBox="0 0 400 120">
          <!-- CPU Line -->
          <path 
            d="{generateMiniChart(cpuData.slice(-20), 100).replace(/^M /, 'M ').replace(/(120)/g, '400')}" 
            stroke="#10b981" 
            stroke-width="2" 
            fill="none"
            transform="scale(3.33, 1)"
          />
          
          <!-- Memory Line -->
          <path 
            d="{generateMiniChart(memoryData.slice(-20), 2048).replace(/^M /, 'M ').replace(/(120)/g, '400')}" 
            stroke="#3b82f6" 
            stroke-width="2" 
            fill="none"
            transform="scale(3.33, 1) translate(0, 20)"
          />
        </svg>
        
        <!-- Legend -->
        <div class="flex gap-4 mt-2 text-sm">
          <div class="flex items-center gap-2">
            <div class="w-3 h-0.5 bg-green-400"></div>
            <span class="text-slate-300">CPU Usage</span>
          </div>
          <div class="flex items-center gap-2">
            <div class="w-3 h-0.5 bg-blue-400"></div>
            <span class="text-slate-300">Memory Usage</span>
          </div>
        </div>
      </div>
    </div>
  {/if}

  <!-- Performance Recommendations -->
  {#if recommendations.length > 0}
    <div class="bg-slate-700/50 border border-slate-600 rounded-lg p-4">
      <h3 class="text-lg font-medium text-white mb-4">💡 Performance Recommendations</h3>
      
      <div class="space-y-3">
        {#each recommendations as rec}
          <div class="flex items-start gap-3 p-3 bg-slate-600/50 rounded-lg">
            <div class="text-xl">
              {#if rec.level === 'critical'}🔴
              {:else if rec.level === 'warning'}🟡
              {:else}🟢{/if}
            </div>
            <div class="flex-1">
              <p class="text-white text-sm font-medium">{rec.message}</p>
              <div class="flex items-center gap-2 mt-2">
                <button class="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-xs rounded-full transition-colors">
                  {rec.action}
                </button>
                <span class="text-xs text-slate-400 capitalize">{rec.type} optimization</span>
              </div>
            </div>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Detailed Metrics Table -->
  <div class="bg-slate-700/50 border border-slate-600 rounded-lg p-4">
    <h3 class="text-lg font-medium text-white mb-4">📊 Detailed Metrics</h3>
    
    <div class="overflow-x-auto">
      <table class="w-full text-sm">
        <thead>
          <tr class="border-b border-slate-600">
            <th class="text-left text-slate-300 pb-2">Metric</th>
            <th class="text-right text-slate-300 pb-2">Current</th>
            <th class="text-right text-slate-300 pb-2">Average</th>
            <th class="text-right text-slate-300 pb-2">Peak</th>
            <th class="text-right text-slate-300 pb-2">Status</th>
          </tr>
        </thead>
        <tbody class="space-y-2">
          <tr class="border-b border-slate-700">
            <td class="py-2 text-white">CPU Usage</td>
            <td class="text-right text-blue-400">{currentCPU}%</td>
            <td class="text-right text-slate-300">{Math.round((cpuData.reduce((a, b) => a + b, 0) / Math.max(cpuData.length, 1)) * 10) / 10}%</td>
            <td class="text-right text-slate-300">{Math.max(...cpuData) || 0}%</td>
            <td class="text-right">
              <span class="px-2 py-1 rounded-full text-xs {getStatusBg(cpuStatus)} {getStatusColor(cpuStatus)}">
                {cpuStatus}
              </span>
            </td>
          </tr>
          <tr class="border-b border-slate-700">
            <td class="py-2 text-white">Memory Usage</td>
            <td class="text-right text-blue-400">{currentMemory} MB</td>
            <td class="text-right text-slate-300">{Math.round((memoryData.reduce((a, b) => a + b, 0) / Math.max(memoryData.length, 1)) * 10) / 10} MB</td>
            <td class="text-right text-slate-300">{Math.max(...memoryData) || 0} MB</td>
            <td class="text-right">
              <span class="px-2 py-1 rounded-full text-xs {getStatusBg(memoryStatus)} {getStatusColor(memoryStatus)}">
                {memoryStatus}
              </span>
            </td>
          </tr>
          <tr class="border-b border-slate-700">
            <td class="py-2 text-white">Cache Hit Rate</td>
            <td class="text-right text-blue-400">{cacheHitRate}%</td>
            <td class="text-right text-slate-300">{cacheHitRate}%</td>
            <td class="text-right text-slate-300">{cacheHitRate}%</td>
            <td class="text-right">
              <span class="px-2 py-1 rounded-full text-xs {getStatusBg(cacheHitRate > 80 ? 'good' : 'warning')} {getStatusColor(cacheHitRate > 80 ? 'good' : 'warning')}">
                {cacheHitRate > 80 ? 'good' : 'needs improvement'}
              </span>
            </td>
          </tr>
          <tr>
            <td class="py-2 text-white">Active Workers</td>
            <td class="text-right text-blue-400">{currentWorkers}</td>
            <td class="text-right text-slate-300">{currentWorkers}</td>
            <td class="text-right text-slate-300">{currentWorkers}</td>
            <td class="text-right">
              <span class="px-2 py-1 rounded-full text-xs bg-green-400/20 border-green-400/30 text-green-400">
                optimal
              </span>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</div>