<!-- Neural Performance Dashboard with Real-time Service Monitoring -->
<!-- Integrates with Enhanced Neural Sprite Engine for comprehensive performance tracking -->

<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { writable, derived, type Readable } from 'svelte/store';
  import type { EnhancedNeuralSpriteEngine } from '$lib/engines/neural-sprite-engine-enhanced';
  import type { EnhancedPerformanceMetrics } from '$lib/engines/neural-sprite-engine-enhanced';

  // Props
  export let neuralEngine: EnhancedNeuralSpriteEngine | null = null;
  export let updateInterval: number = 1000;
  export let maxHistoryPoints: number = 60;
  export let showAdvancedMetrics: boolean = true;

  // Real-time data stores
  const performanceHistory = writable<EnhancedPerformanceMetrics[]>([]);
  const currentMetrics = writable<EnhancedPerformanceMetrics | null>(null);
  const serviceHealth = writable<Record<string, any>>({});
  const connectionStatus = writable<'connected' | 'disconnected' | 'error'>('disconnected');

  // Dashboard state
  let isMonitoring = false;
  let monitoringInterval: NodeJS.Timeout | null = null;
  let lastUpdate = Date.now();

  // Derived performance indicators
  const overallGrade = derived([currentMetrics], ([$metrics]) => {
    if (!$metrics) return 'N/A';
    
    const efficiency = $metrics.neuralEfficiency;
    if (efficiency >= 95) return { grade: 'S+', color: 'text-emerald-400', bg: 'bg-emerald-500/20' };
    if (efficiency >= 90) return { grade: 'S', color: 'text-green-400', bg: 'bg-green-500/20' };
    if (efficiency >= 80) return { grade: 'A', color: 'text-blue-400', bg: 'bg-blue-500/20' };
    if (efficiency >= 70) return { grade: 'B', color: 'text-yellow-400', bg: 'bg-yellow-500/20' };
    if (efficiency >= 60) return { grade: 'C', color: 'text-orange-400', bg: 'bg-orange-500/20' };
    return { grade: 'D', color: 'text-red-400', bg: 'bg-red-500/20' };
  });

  const systemStatus = derived([serviceHealth], ([$health]) => {
    const services = Object.values($health);
    const connected = services.filter((s: any) => s.connected).length;
    const total = services.length;
    
    if (total === 0) return { status: 'Initializing', color: 'text-gray-400' };
    if (connected === total) return { status: 'All Systems Operational', color: 'text-green-400' };
    if (connected > total * 0.7) return { status: 'Minor Issues Detected', color: 'text-yellow-400' };
    return { status: 'Service Degradation', color: 'text-red-400' };
  });

  // Start monitoring when component mounts
  onMount(() => {
    if (neuralEngine) {
      startMonitoring();
    }
  });

  // Cleanup on destroy
  onDestroy(() => {
    stopMonitoring();
  });

  // Start real-time monitoring
  function startMonitoring() {
    if (!neuralEngine || isMonitoring) return;
    
    isMonitoring = true;
    connectionStatus.set('connected');
    
    // Subscribe to neural engine stores
    neuralEngine.enhancedPerformance.subscribe(metrics => {
      currentMetrics.set(metrics);
      updateHistory(metrics);
    });

    neuralEngine.serviceStatus.subscribe(status => {
      serviceHealth.set(status);
    });

    neuralEngine.isServicesHealthy.subscribe(healthy => {
      if (!healthy && $connectionStatus === 'connected') {
        connectionStatus.set('error');
      } else if (healthy && $connectionStatus === 'error') {
        connectionStatus.set('connected');
      }
    });

    // Start periodic updates
    monitoringInterval = setInterval(async () => {
      if (neuralEngine) {
        try {
          const health = await neuralEngine.getServiceHealth();
          serviceHealth.set(health.services);
          lastUpdate = Date.now();
        } catch (error) {
          console.error('Dashboard monitoring error:', error);
          connectionStatus.set('error');
        }
      }
    }, updateInterval);
  }

  // Stop monitoring
  function stopMonitoring() {
    isMonitoring = false;
    connectionStatus.set('disconnected');
    
    if (monitoringInterval) {
      clearInterval(monitoringInterval);
      monitoringInterval = null;
    }
  }

  // Update performance history
  function updateHistory(metrics: EnhancedPerformanceMetrics) {
    performanceHistory.update(history => {
      const newHistory = [...history, { ...metrics, timestamp: Date.now() }];
      return newHistory.slice(-maxHistoryPoints);
    });
  }

  // Format large numbers
  function formatNumber(num: number): string {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toString();
  }

  // Format latency
  function formatLatency(ms: number): string {
    if (ms < 1) return '<1ms';
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  }

  // Get service status icon
  function getServiceIcon(connected: boolean): string {
    return connected ? '🟢' : '🔴';
  }

  // Toggle monitoring
  function toggleMonitoring() {
    if (isMonitoring) {
      stopMonitoring();
    } else {
      startMonitoring();
    }
  }
</script>

<!-- Dashboard Header -->
<div class="neural-dashboard bg-gray-900 text-white rounded-lg border border-gray-800 overflow-hidden">
  <div class="dashboard-header bg-gradient-to-r from-blue-600 to-purple-600 px-4 py-3">
    <div class="flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <div class="w-3 h-3 rounded-full {$connectionStatus === 'connected' ? 'bg-green-400' : $connectionStatus === 'error' ? 'bg-red-400' : 'bg-gray-400'}"></div>
        <h2 class="text-lg font-bold">Neural Engine Performance</h2>
      </div>
      <div class="flex items-center space-x-4">
        <div class="text-sm opacity-90">
          Last Update: {new Date(lastUpdate).toLocaleTimeString()}
        </div>
        <button 
          on:click={toggleMonitoring}
          class="px-3 py-1 bg-white/20 hover:bg-white/30 rounded text-sm font-medium transition-colors"
        >
          {isMonitoring ? 'Pause' : 'Resume'}
        </button>
      </div>
    </div>
  </div>

  <!-- Performance Overview -->
  {#if $currentMetrics}
  <div class="p-4">
    <!-- Overall Grade & Status -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
      <div class="performance-card bg-gray-800 rounded p-4">
        <div class="flex items-center justify-between mb-2">
          <span class="text-sm text-gray-400">Overall Grade</span>
          {#if $overallGrade}
          <div class="grade-badge {$overallGrade.bg} {$overallGrade.color} px-2 py-1 rounded text-lg font-bold">
            {$overallGrade.grade}
          </div>
          {/if}
        </div>
        <div class="text-xl font-mono">
          {$currentMetrics.neuralEfficiency.toFixed(1)}% Efficiency
        </div>
      </div>

      <div class="performance-card bg-gray-800 rounded p-4">
        <div class="flex items-center justify-between mb-2">
          <span class="text-sm text-gray-400">System Status</span>
          <div class="status-indicator w-3 h-3 rounded-full {$systemStatus.color}"></div>
        </div>
        <div class="text-sm {$systemStatus.color}">
          {$systemStatus.status}
        </div>
      </div>

      <div class="performance-card bg-gray-800 rounded p-4">
        <div class="flex items-center justify-between mb-2">
          <span class="text-sm text-gray-400">Cache Performance</span>
          <span class="text-sm text-green-400">{($currentMetrics.cacheHitRate * 100).toFixed(1)}%</span>
        </div>
        <div class="w-full bg-gray-700 rounded-full h-2">
          <div 
            class="bg-gradient-to-r from-green-400 to-blue-400 h-2 rounded-full transition-all duration-500"
            style="width: {$currentMetrics.cacheHitRate * 100}%"
          ></div>
        </div>
      </div>
    </div>

    <!-- Detailed Metrics -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Service Connections -->
      <div class="metrics-section">
        <h3 class="text-lg font-semibold mb-3 flex items-center">
          📡 Service Connections
          <span class="ml-2 text-sm text-gray-400">({$currentMetrics.databaseConnections} active)</span>
        </h3>
        <div class="space-y-2">
          {#each Object.entries($serviceHealth) as [serviceName, status]}
          <div class="service-row flex items-center justify-between p-2 bg-gray-800 rounded">
            <div class="flex items-center space-x-2">
              <span>{getServiceIcon(status.connected)}</span>
              <span class="font-medium">{serviceName}</span>
            </div>
            <div class="flex items-center space-x-2 text-sm">
              {#if status.connected}
                <span class="text-green-400">Connected</span>
                {#if status.lastCheck}
                  <span class="text-gray-400">({new Date(status.lastCheck).toLocaleTimeString()})</span>
                {/if}
              {:else}
                <span class="text-red-400">Disconnected</span>
                {#if status.error}
                  <span class="text-red-300 text-xs">({status.error.substring(0, 20)}...)</span>
                {/if}
              {/if}
            </div>
          </div>
          {/each}
        </div>
      </div>

      <!-- Performance Metrics -->
      <div class="metrics-section">
        <h3 class="text-lg font-semibold mb-3">⚡ Performance Metrics</h3>
        <div class="grid grid-cols-2 gap-3">
          <div class="metric-card bg-gray-800 p-3 rounded">
            <div class="text-xs text-gray-400 mb-1">Memory Usage</div>
            <div class="text-lg font-mono text-blue-400">{$currentMetrics.memoryUsage}MB</div>
          </div>
          <div class="metric-card bg-gray-800 p-3 rounded">
            <div class="text-xs text-gray-400 mb-1">Queue Depth</div>
            <div class="text-lg font-mono text-yellow-400">{$currentMetrics.queueDepth}</div>
          </div>
          <div class="metric-card bg-gray-800 p-3 rounded">
            <div class="text-xs text-gray-400 mb-1">Tasks/sec</div>
            <div class="text-lg font-mono text-green-400">{$currentMetrics.distributedTasksPerSecond.toFixed(1)}</div>
          </div>
          <div class="metric-card bg-gray-800 p-3 rounded">
            <div class="text-xs text-gray-400 mb-1">Storage Used</div>
            <div class="text-lg font-mono text-purple-400">{formatNumber($currentMetrics.storageUtilization)}</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Service Latency -->
    {#if showAdvancedMetrics && Object.keys($currentMetrics.serviceLatency).length > 0}
    <div class="mt-6">
      <h3 class="text-lg font-semibold mb-3">🚀 Service Latency</h3>
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {#each Object.entries($currentMetrics.serviceLatency) as [service, latency]}
        <div class="latency-card bg-gray-800 p-3 rounded">
          <div class="flex justify-between items-center">
            <span class="text-sm text-gray-300">{service}</span>
            <span class="font-mono text-sm {latency < 100 ? 'text-green-400' : latency < 500 ? 'text-yellow-400' : 'text-red-400'}">
              {formatLatency(latency)}
            </span>
          </div>
          <div class="w-full bg-gray-700 rounded-full h-1 mt-2">
            <div 
              class="h-1 rounded-full transition-all duration-300 {latency < 100 ? 'bg-green-400' : latency < 500 ? 'bg-yellow-400' : 'bg-red-400'}"
              style="width: {Math.min(100, (latency / 1000) * 100)}%"
            ></div>
          </div>
        </div>
        {/each}
      </div>
    </div>
    {/if}

    <!-- Performance History Graph (Simplified) -->
    {#if $performanceHistory.length > 5}
    <div class="mt-6">
      <h3 class="text-lg font-semibold mb-3">📈 Performance Trend</h3>
      <div class="bg-gray-800 p-4 rounded">
        <div class="flex justify-between text-xs text-gray-400 mb-2">
          <span>Cache Hit Rate</span>
          <span>Neural Efficiency</span>
        </div>
        <div class="performance-graph h-20 relative bg-gray-900 rounded overflow-hidden">
          <!-- Simple sparkline representation -->
          <div class="absolute inset-0 flex items-end justify-between p-2">
            {#each $performanceHistory.slice(-20) as point, index}
            <div class="flex flex-col items-center justify-end h-full">
              <div 
                class="bg-blue-400 w-1 transition-all duration-300"
                style="height: {point.cacheHitRate * 100}%"
              ></div>
              <div 
                class="bg-green-400 w-1 transition-all duration-300 mt-0.5"
                style="height: {point.neuralEfficiency}%"
              ></div>
            </div>
            {/each}
          </div>
        </div>
        <div class="flex justify-between text-xs text-gray-400 mt-2">
          <span class="flex items-center"><div class="w-2 h-2 bg-blue-400 rounded mr-1"></div> Cache</span>
          <span class="flex items-center"><div class="w-2 h-2 bg-green-400 rounded mr-1"></div> Efficiency</span>
        </div>
      </div>
    </div>
    {/if}
  </div>
  {:else}
  <!-- Loading State -->
  <div class="p-8 text-center">
    <div class="animate-spin w-8 h-8 border-2 border-blue-400 border-t-transparent rounded-full mx-auto mb-4"></div>
    <div class="text-gray-400">Initializing Neural Engine Monitoring...</div>
  </div>
  {/if}
</div>

<style>
  .neural-dashboard {
    font-family: 'JetBrains Mono', monospace;
    max-height: 80vh;
    overflow-y: auto
  }

  .performance-card {
    transition: all 0.3s ease;
  }

  .performance-card:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  }

  .service-row {
    transition: background-color 0.2s ease;
  }

  .service-row:hover {
    background-color: rgb(55, 65, 81);
  }

  .metric-card {
    border-left: 3px solid transparent;
    transition: all 0.2s ease;
  }

  .metric-card:hover {
    border-left-color: rgb(59, 130, 246);
  }

  .performance-graph {
    background-image: 
      linear-gradient(to right, rgba(59, 130, 246, 0.1) 1px, transparent 1px),
      linear-gradient(to bottom, rgba(59, 130, 246, 0.1) 1px, transparent 1px);
    background-size: 10px 10px;
  }

  .grade-badge {
    min-width: 2.5rem;
    text-align: center
    font-size: 1.1em;
    letter-spacing: 1px;
  }

  /* Dark mode scrollbar */
  .neural-dashboard::-webkit-scrollbar {
    width: 6px;
  }

  .neural-dashboard::-webkit-scrollbar-track {
    background: rgb(31, 41, 55);
  }

  .neural-dashboard::-webkit-scrollbar-thumb {
    background: rgb(75, 85, 99);
    border-radius: 3px;
  }

  .neural-dashboard::-webkit-scrollbar-thumb:hover {
    background: rgb(107, 114, 128);
  }

  /* Animation for metrics updates */
  @keyframes pulse {
    0% { opacity: 0.7; }
    50% { opacity: 1; }
    100% { opacity: 0.7; }
  }

  .status-indicator {
    animation: pulse 2s infinite;
  }
</style>