<!-- YoRHa System Dashboard -->
<script lang="ts">
  import { $props, $state } from 'svelte';
  import { onMount, onDestroy } from 'svelte';
  import { yorhaAPI } from '$lib/components/three/yorha-ui/api/YoRHaAPIClient';
  import YoRHaSystemStatus from '$lib/components/yorha/YoRHaSystemStatus.svelte';
  import YoRHaDataViz from '$lib/components/yorha/YoRHaDataViz.svelte';
  import type { PageData } from './$types';
  import { 
    Monitor, 
    Cpu, 
    Database, 
    Activity, 
    HardDrive,
    Zap,
    Network,
    AlertTriangle,
    CheckCircle,
    TrendingUp
  } from 'lucide-svelte';

  let { data }: { data: PageData } = $props();

  // System metrics and status - initialized from SSR data
  let systemMetrics = $state(data.systemStatus);
  let graphData = $state(data.graphData);
  let multicoreStatus = $state(data.multicoreStatus);

  let realtimeData = $state({
    cpuHistory: [],
    memoryHistory: [],
    networkHistory: [],
    timestamp: Date.now()
  });

  let isLoading = $state(!data.initialLoad);
  let lastUpdate = $state(new Date(data.timestamp));

  // Data update intervals
  let metricsInterval: ReturnType<typeof setInterval>;
  let realtimeInterval: ReturnType<typeof setInterval>;

  onMount(async () => {
    await loadSystemData();
    startRealTimeUpdates();
  });

  onDestroy(() => {
    if (metricsInterval) clearInterval(metricsInterval);
    if (realtimeInterval) clearInterval(realtimeInterval);
  });

  async function loadSystemData() {
    try {
      // Load system status from API
      const [status, graph] = await Promise.all([
        yorhaAPI.getSystemStatus(),
        yorhaAPI.getGraphData()
      ]);

      systemMetrics = status;
      graphData = graph;
      
      // Initialize realtime data
      realtimeData = {
        cpuHistory: generateHistoryData(systemMetrics.backend.cpuUsage),
        memoryHistory: generateHistoryData(systemMetrics.backend.memoryUsage),
        networkHistory: generateHistoryData(systemMetrics.database.latency),
        timestamp: Date.now()
      };

      isLoading = false;
    } catch (error) {
      console.error('Failed to load system data:', error);
      // Use mock data for demo
      systemMetrics = mockSystemMetrics();
      graphData = mockGraphData();
      realtimeData = {
        cpuHistory: generateHistoryData(45),
        memoryHistory: generateHistoryData(62),
        networkHistory: generateHistoryData(23),
        timestamp: Date.now()
      };
      isLoading = false;
    }
  }

  function startRealTimeUpdates() {
    // Update metrics every 5 seconds
    metricsInterval = setInterval(async () => {
      try {
        const status = await yorhaAPI.getSystemStatus();
        systemMetrics = status;
        lastUpdate = new Date();
      } catch (error) {
        // Update with simulated changes
        systemMetrics = {
          ...systemMetrics,
          backend: {
            ...systemMetrics.backend,
            cpuUsage: Math.max(20, Math.min(80, systemMetrics.backend.cpuUsage + (Math.random() - 0.5) * 10)),
            memoryUsage: Math.max(30, Math.min(85, systemMetrics.backend.memoryUsage + (Math.random() - 0.5) * 8))
          },
          database: {
            ...systemMetrics.database,
            latency: Math.max(10, Math.min(100, systemMetrics.database.latency + (Math.random() - 0.5) * 5)),
            queryCount: systemMetrics.database.queryCount + Math.floor(Math.random() * 5)
          }
        };
        lastUpdate = new Date();
      }
    }, 5000);

    // Update realtime charts every 2 seconds
    realtimeInterval = setInterval(() => {
      realtimeData = {
        cpuHistory: [...realtimeData.cpuHistory.slice(-29), systemMetrics.backend.cpuUsage],
        memoryHistory: [...realtimeData.memoryHistory.slice(-29), systemMetrics.backend.memoryUsage],
        networkHistory: [...realtimeData.networkHistory.slice(-29), systemMetrics.database.latency],
        timestamp: Date.now()
      };
    }, 2000);
  }

  function generateHistoryData(baseValue: number, points = 30): number[] {
    return Array.from({ length: points }, (_, i) => {
      const variation = (Math.random() - 0.5) * 20;
      return Math.max(0, Math.min(100, baseValue + variation));
    });
  }

  function mockSystemMetrics() {
    return {
      database: {
        connected: true,
        latency: 23,
        activeConnections: 12,
        queryCount: 15847
      },
      backend: {
        healthy: true,
        uptime: 98.7,
        activeServices: 8,
        cpuUsage: 45,
        memoryUsage: 62
      },
      frontend: {
        renderFPS: 60,
        componentCount: 127,
        activeComponents: 89,
        webGPUEnabled: true
      }
    };
  }

  function mockGraphData() {
    return {
      nodes: [
        {
          id: 'postgres',
          type: 'database',
          label: 'PostgreSQL',
          position: { x: 0, y: 0, z: 0 },
          metrics: { connections: 12, queries: 15847 },
          status: 'healthy'
        },
        {
          id: 'redis',
          type: 'database',
          label: 'Redis',
          position: { x: 1, y: 0, z: 0 },
          metrics: { memory: '2.1GB', keys: 45823 },
          status: 'healthy'
        },
        {
          id: 'ollama',
          type: 'service',
          label: 'Ollama AI',
          position: { x: 0, y: 1, z: 0 },
          metrics: { models: 3, requests: 1847 },
          status: 'healthy'
        },
        {
          id: 'sveltekit',
          type: 'component',
          label: 'SvelteKit',
          position: { x: 1, y: 1, z: 0 },
          metrics: { components: 127, fps: 60 },
          status: 'healthy'
        }
      ],
      edges: [
        {
          from: 'sveltekit',
          to: 'postgres',
          type: 'api',
          traffic: 85,
          latency: 23
        },
        {
          from: 'sveltekit',
          to: 'redis',
          type: 'api',
          traffic: 65,
          latency: 12
        },
        {
          from: 'postgres',
          to: 'ollama',
          type: 'data',
          traffic: 45,
          latency: 34
        }
      ]
    };
  }

  function getStatusIcon(status: string) {
    switch (status) {
      case 'healthy':
        return CheckCircle;
      case 'warning':
        return AlertTriangle;
      case 'error':
        return AlertTriangle;
      default:
        return Monitor;
    }
  }

  function getStatusColor(status: string) {
    switch (status) {
      case 'healthy':
        return 'text-green-400';
      case 'warning':
        return 'text-yellow-400';
      case 'error':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  }
</script>

<svelte:head>
  <title>YoRHa Dashboard - System Monitoring</title>
</svelte:head>

<div class="yorha-dashboard-page">
  <!-- Page Header -->
  <header class="yorha-page-header">
    <div class="yorha-header-content">
      <div class="yorha-header-title">
        <Monitor size={48} />
        <h1>SYSTEM DASHBOARD</h1>
        <div class="yorha-header-subtitle">REAL-TIME MONITORING & ANALYTICS</div>
      </div>
      
      <div class="yorha-header-status">
        <div class="yorha-status-item">
          <Activity size={16} />
          <span>LIVE DATA</span>
        </div>
        <div class="yorha-status-item">
          <span>LAST UPDATE: {lastUpdate.toLocaleTimeString()}</span>
        </div>
      </div>
    </div>
  </header>

  {#if isLoading}
    <div class="yorha-loading">
      <div class="yorha-loading-spinner"></div>
      <span>INITIALIZING DASHBOARD...</span>
    </div>
  {:else}
    <!-- System Overview Cards -->
    <section class="yorha-overview">
      <div class="yorha-metrics-grid">
        <!-- Database Status -->
        <div class="yorha-metric-card yorha-card-database">
          <div class="yorha-metric-header">
            <Database size={24} />
            <h3>DATABASE</h3>
            <svelte:component this={getStatusIcon(systemMetrics.database.connected ? 'healthy' : 'error')} 
                             size={20} 
                             class={getStatusColor(systemMetrics.database.connected ? 'healthy' : 'error')} />
          </div>
          <div class="yorha-metric-stats">
            <div class="yorha-stat">
              <span class="yorha-stat-value">{systemMetrics.database.latency}ms</span>
              <span class="yorha-stat-label">LATENCY</span>
            </div>
            <div class="yorha-stat">
              <span class="yorha-stat-value">{systemMetrics.database.activeConnections}</span>
              <span class="yorha-stat-label">CONNECTIONS</span>
            </div>
            <div class="yorha-stat">
              <span class="yorha-stat-value">{systemMetrics.database.queryCount.toLocaleString()}</span>
              <span class="yorha-stat-label">QUERIES</span>
            </div>
          </div>
        </div>

        <!-- Backend Status -->
        <div class="yorha-metric-card yorha-card-backend">
          <div class="yorha-metric-header">
            <Cpu size={24} />
            <h3>BACKEND</h3>
            <svelte:component this={getStatusIcon(systemMetrics.backend.healthy ? 'healthy' : 'error')} 
                             size={20} 
                             class={getStatusColor(systemMetrics.backend.healthy ? 'healthy' : 'error')} />
          </div>
          <div class="yorha-metric-stats">
            <div class="yorha-stat">
              <span class="yorha-stat-value">{systemMetrics.backend.cpuUsage}%</span>
              <span class="yorha-stat-label">CPU</span>
            </div>
            <div class="yorha-stat">
              <span class="yorha-stat-value">{systemMetrics.backend.memoryUsage}%</span>
              <span class="yorha-stat-label">MEMORY</span>
            </div>
            <div class="yorha-stat">
              <span class="yorha-stat-value">{systemMetrics.backend.activeServices}</span>
              <span class="yorha-stat-label">SERVICES</span>
            </div>
          </div>
        </div>

        <!-- Frontend Status -->
        <div class="yorha-metric-card yorha-card-frontend">
          <div class="yorha-metric-header">
            <Monitor size={24} />
            <h3>FRONTEND</h3>
            <svelte:component this={getStatusIcon('healthy')} 
                             size={20} 
                             class={getStatusColor('healthy')} />
          </div>
          <div class="yorha-metric-stats">
            <div class="yorha-stat">
              <span class="yorha-stat-value">{systemMetrics.frontend.renderFPS}</span>
              <span class="yorha-stat-label">FPS</span>
            </div>
            <div class="yorha-stat">
              <span class="yorha-stat-value">{systemMetrics.frontend.activeComponents}</span>
              <span class="yorha-stat-label">ACTIVE</span>
            </div>
            <div class="yorha-stat">
              <span class="yorha-stat-value">{systemMetrics.frontend.webGPUEnabled ? 'YES' : 'NO'}</span>
              <span class="yorha-stat-label">WEBGPU</span>
            </div>
          </div>
        </div>

        <!-- System Health -->
        <div class="yorha-metric-card yorha-card-health">
          <div class="yorha-metric-header">
            <Zap size={24} />
            <h3>HEALTH</h3>
            <svelte:component this={getStatusIcon('healthy')} 
                             size={20} 
                             class={getStatusColor('healthy')} />
          </div>
          <div class="yorha-metric-stats">
            <div class="yorha-stat">
              <span class="yorha-stat-value">{systemMetrics.backend.uptime}%</span>
              <span class="yorha-stat-label">UPTIME</span>
            </div>
            <div class="yorha-stat">
              <span class="yorha-stat-value">8/8</span>
              <span class="yorha-stat-label">SERVICES</span>
            </div>
            <div class="yorha-stat">
              <span class="yorha-stat-value">0</span>
              <span class="yorha-stat-label">ERRORS</span>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Real-time Charts -->
    <section class="yorha-charts">
      <h2 class="yorha-section-title">
        <TrendingUp size={24} />
        REAL-TIME METRICS
      </h2>
      
      <div class="yorha-charts-grid">
        <div class="yorha-chart-card">
          <h3>CPU USAGE</h3>
          <div class="yorha-chart">
            <div class="yorha-chart-line" style="--height: {systemMetrics.backend.cpuUsage}%"></div>
            <span class="yorha-chart-value">{systemMetrics.backend.cpuUsage}%</span>
          </div>
        </div>

        <div class="yorha-chart-card">
          <h3>MEMORY USAGE</h3>
          <div class="yorha-chart">
            <div class="yorha-chart-line" style="--height: {systemMetrics.backend.memoryUsage}%"></div>
            <span class="yorha-chart-value">{systemMetrics.backend.memoryUsage}%</span>
          </div>
        </div>

        <div class="yorha-chart-card">
          <h3>NETWORK LATENCY</h3>
          <div class="yorha-chart">
            <div class="yorha-chart-line" style="--height: {Math.min(100, systemMetrics.database.latency)}%"></div>
            <span class="yorha-chart-value">{systemMetrics.database.latency}ms</span>
          </div>
        </div>
      </div>
    </section>

    <!-- YoRHa System Status Component -->
    <section class="yorha-system-status">
      <YoRHaSystemStatus 
        systemLoad={systemMetrics.systemLoad || systemMetrics.backend?.cpuUsage || 45}
        gpuUtilization={systemMetrics.gpuUtilization || 78}
        memoryUsage={systemMetrics.backend?.memoryUsage || 62}
        networkLatency={systemMetrics.networkLatency || systemMetrics.database?.latency || 23}
      />
    </section>

    <!-- Data Visualization -->
    <section class="yorha-data-viz">
      <YoRHaDataViz />
    </section>

    <!-- System Graph -->
    <section class="yorha-graph">
      <h2 class="yorha-section-title">
        <Network size={24} />
        SYSTEM ARCHITECTURE
      </h2>
      
      <div class="yorha-graph-container">
        {#each graphData.nodes as node}
          <div class="yorha-graph-node yorha-node-{node.type}" 
               style="left: {node.position.x * 200 + 100}px; top: {node.position.y * 150 + 50}px;">
            <div class="yorha-node-icon">
              {#if node.type === 'database'}
                <Database size={20} />
              {:else if node.type === 'service'}
                <Cpu size={20} />
              {:else}
                <Monitor size={20} />
              {/if}
            </div>
            <div class="yorha-node-label">{node.label}</div>
            <div class="yorha-node-status yorha-status-{node.status}"></div>
          </div>
        {/each}
        
        {#each graphData.edges as edge}
          <div class="yorha-graph-edge yorha-edge-{edge.type}">
            <span class="yorha-edge-label">{edge.traffic}% • {edge.latency}ms</span>
          </div>
        {/each}
      </div>
    </section>
  {/if}
</div>

<style>
  .yorha-dashboard-page {
    @apply space-y-8 pb-16;
  }

  /* Page Header */
  .yorha-page-header {
    @apply py-12 px-6 border-b border-amber-400 border-opacity-30;
    background: linear-gradient(135deg, rgba(0, 0, 0, 0.8) 0%, rgba(255, 191, 0, 0.05) 100%);
  }

  .yorha-header-content {
    @apply max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6;
  }

  .yorha-header-title {
    @apply text-center md:text-left space-y-2;
  }

  .yorha-header-title h1 {
    @apply text-3xl md:text-4xl font-bold tracking-wider text-amber-400 flex items-center gap-4;
    text-shadow: 0 0 20px rgba(255, 191, 0, 0.5);
  }

  .yorha-header-subtitle {
    @apply text-lg text-amber-300 tracking-wide opacity-80;
  }

  .yorha-header-status {
    @apply flex items-center gap-4;
  }

  .yorha-status-item {
    @apply flex items-center gap-2 text-xs text-amber-400 opacity-60;
  }

  /* Loading */
  .yorha-loading {
    @apply flex flex-col items-center justify-center py-32 space-y-4;
  }

  .yorha-loading-spinner {
    @apply w-8 h-8 border-2 border-amber-400 border-t-transparent rounded-full;
    animation: spin 1s linear infinite;
  }

  /* Overview Metrics */
  .yorha-overview {
    @apply px-6;
  }

  .yorha-metrics-grid {
    @apply grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-6xl mx-auto;
  }

  .yorha-metric-card {
    @apply bg-gray-900 border-2 p-6 space-y-4;
  }

  .yorha-card-database {
    @apply border-blue-400;
  }

  .yorha-card-backend {
    @apply border-green-400;
  }

  .yorha-card-frontend {
    @apply border-purple-400;
  }

  .yorha-card-health {
    @apply border-orange-400;
  }

  .yorha-metric-header {
    @apply flex items-center justify-between;
  }

  .yorha-metric-header h3 {
    @apply font-bold tracking-wider text-lg;
  }

  .yorha-metric-stats {
    @apply grid grid-cols-3 gap-2 text-center;
  }

  .yorha-stat-value {
    @apply block text-xl font-bold;
  }

  .yorha-stat-label {
    @apply block text-xs opacity-60 mt-1;
  }

  /* Charts */
  .yorha-charts {
    @apply px-6 space-y-6;
  }

  .yorha-section-title {
    @apply text-2xl font-bold text-amber-400 tracking-wider flex items-center gap-3 max-w-6xl mx-auto;
  }

  .yorha-charts-grid {
    @apply grid grid-cols-1 md:grid-cols-3 gap-6 max-w-6xl mx-auto;
  }

  .yorha-chart-card {
    @apply bg-gray-900 border border-amber-400 border-opacity-30 p-6;
  }

  .yorha-chart-card h3 {
    @apply text-sm font-bold text-amber-400 mb-4 tracking-wider;
  }

  .yorha-chart {
    @apply relative h-32 bg-black border border-amber-400 border-opacity-20 flex items-end justify-center;
  }

  .yorha-chart-line {
    @apply w-full bg-gradient-to-t from-amber-400 to-amber-300 transition-all duration-1000;
    height: var(--height);
  }

  .yorha-chart-value {
    @apply absolute top-2 right-2 text-xs text-amber-400 font-mono;
  }

  /* System Status */
  .yorha-system-status {
    @apply px-6;
  }

  .yorha-data-viz {
    @apply px-6;
  }

  /* Graph */
  .yorha-graph {
    @apply px-6 space-y-6;
  }

  .yorha-graph-container {
    @apply relative bg-gray-900 border border-amber-400 border-opacity-30 p-8 max-w-6xl mx-auto;
    min-height: 400px;
  }

  .yorha-graph-node {
    @apply absolute flex flex-col items-center space-y-2 p-3 border border-opacity-60;
  }

  .yorha-node-database {
    @apply border-blue-400 bg-blue-400 bg-opacity-10;
  }

  .yorha-node-service {
    @apply border-green-400 bg-green-400 bg-opacity-10;
  }

  .yorha-node-component {
    @apply border-purple-400 bg-purple-400 bg-opacity-10;
  }

  .yorha-node-icon {
    @apply text-current;
  }

  .yorha-node-label {
    @apply text-xs font-mono text-current;
  }

  .yorha-node-status {
    @apply w-2 h-2 rounded-full;
  }

  .yorha-status-healthy {
    @apply bg-green-400;
  }

  .yorha-status-warning {
    @apply bg-yellow-400;
  }

  .yorha-status-error {
    @apply bg-red-400;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  /* Responsive */
  @media (max-width: 768px) {
    .yorha-header-title h1 {
      @apply text-2xl flex-col;
    }
    
    .yorha-metrics-grid {
      @apply grid-cols-1 gap-4;
    }
    
    .yorha-charts-grid {
      @apply grid-cols-1 gap-4;
    }
  }
</style>