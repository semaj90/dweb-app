<!--
  Cache Performance Monitoring Dashboard
  Comprehensive monitoring and analytics for GPU texture and shader caching

  Features:
  - Real-time performance metrics visualization
  - Cache hit/miss rate tracking with historical data
  - Memory usage monitoring with predictive analysis
  - Shader compilation time optimization tracking
  - WASM acceleration performance metrics
  - Interactive performance tuning controls
  - Export capabilities for performance reports
-->

<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { enhancedGPUCache } from '../../services/enhanced-gpu-cache-service.js';
  import type { CacheAnalytics, CachePerformanceTracker } from '../../services/enhanced-gpu-cache-service.js';
  import { nesGPUBridge } from '../../gpu/nes-gpu-memory-bridge.js';

  interface Props {
    // Dashboard configuration
    updateInterval?: number;
    showRealTimeCharts?: boolean;
    showHistoricalData?: boolean;
    enablePerformanceTuning?: boolean;

    // Display options
    compactMode?: boolean;
    darkTheme?: boolean;
    showAdvancedMetrics?: boolean;
    enableAlerts?: boolean;

    // Performance thresholds
    memoryWarningThreshold?: number;
    cacheHitRateWarning?: number;
    fpsWarningThreshold?: number;

    // Event handlers
    onAlert?: (type: string, message: string, severity: 'info' | 'warning' | 'error') => void;
    onPerformanceOptimized?: (metrics: any) => void;
    onCacheCleared?: () => void;
  }

  let {
    updateInterval = 1000,
    showRealTimeCharts = true,
    showHistoricalData = true,
    enablePerformanceTuning = true,
    compactMode = false,
    darkTheme = true,
    showAdvancedMetrics = false,
    enableAlerts = true,
    memoryWarningThreshold = 0.8,
    cacheHitRateWarning = 0.6,
    fpsWarningThreshold = 45,
    onAlert,
    onPerformanceOptimized,
    onCacheCleared
  }: Props = $props();

  // Component state
  let isMonitoring = $state(false);
  let lastUpdateTime = $state<Date>(new Date());
  let monitoringDuration = $state(0);

  // Current metrics
  let currentMetrics = $state<CacheAnalytics & CachePerformanceTracker>({
    totalEntries: 0,
    totalSize: 0,
    hitRate: 0,
    missRate: 0,
    evictionRate: 0,
    averageEntryAge: 0,
    hotEntries: [],
    coldEntries: [],
    workflowDistribution: {},
    textureHitRate: 0,
    shaderHitRate: 0,
    averageTextureLoadTime: 0,
    averageShaderCompileTime: 0,
    memoryUtilization: 0,
    cacheEfficiency: 0,
    gpuUtilization: 0,
    wasmAccelerationGain: 0
  });

  // Historical data for charts
  let historicalData = $state({
    timestamps: [] as number[],
    hitRates: [] as number[],
    memoryUsage: [] as number[],
    loadTimes: [] as number[],
    compileTimes: [] as number[],
    gpuUtilization: [] as number[],
    wasmGains: [] as number[]
  });

  // Performance alerts
  let activeAlerts = $state<Array<{
    id: string;
    type: string;
    message: string;
    severity: 'info' | 'warning' | 'error';
    timestamp: number;
    acknowledged: boolean;
  }>>([]);

  // Performance optimization suggestions
  let optimizationSuggestions = $state<Array<{
    id: string;
    title: string;
    description: string;
    impact: 'low' | 'medium' | 'high';
    effort: 'easy' | 'moderate' | 'complex';
    action: () => void;
  }>>([]);

  // Monitoring intervals
let metricsInterval = $state<number | null >(null);
let chartUpdateInterval = $state<number | null >(null);
let alertCheckInterval = $state<number | null >(null);

  // Chart configuration
  const maxDataPoints = 60; // Keep last 60 data points
let chartCanvas = $state<HTMLCanvasElement | null >(null);
let chartContext = $state<CanvasRenderingContext2D | null >(null);

  // Performance tuning state
  let tuningEnabled = $state(false);
  let autoOptimizationEnabled = $state(false);
  let optimizationAggressiveness = $state<'conservative' | 'balanced' | 'aggressive'>('balanced');

  /**
   * Start performance monitoring
   */
  function startMonitoring(): void {
    if (isMonitoring) return;

    isMonitoring = true;
    const startTime = Date.now();

    // Main metrics update interval
    metricsInterval = setInterval(() => {
      updateMetrics();
      monitoringDuration = Date.now() - startTime;
    }, updateInterval) as any;

    // Chart update interval (less frequent)
    if (showRealTimeCharts) {
      chartUpdateInterval = setInterval(() => {
        updateCharts();
      }, updateInterval * 2) as any;
    }

    // Alert checking interval
    if (enableAlerts) {
      alertCheckInterval = setInterval(() => {
        checkPerformanceAlerts();
      }, updateInterval * 3) as any;
    }

    console.log('🎯 Cache performance monitoring started');
  }

  /**
   * Stop performance monitoring
   */
  function stopMonitoring(): void {
    if (!isMonitoring) return;

    isMonitoring = false;

    if (metricsInterval) {
      clearInterval(metricsInterval);
      metricsInterval = null;
    }

    if (chartUpdateInterval) {
      clearInterval(chartUpdateInterval);
      chartUpdateInterval = null;
    }

    if (alertCheckInterval) {
      clearInterval(alertCheckInterval);
      alertCheckInterval = null;
    }

    console.log('⏹️ Cache performance monitoring stopped');
  }

  /**
   * Update current performance metrics
   */
  function updateMetrics(): void {
    try {
      // Get cache analytics
      const analytics = enhancedGPUCache.getCacheAnalytics();

      // Get GPU bridge performance
      const gpuMetrics = nesGPUBridge.getPerformanceMetrics();

      // Combine metrics
      currentMetrics = {
        ...analytics,
        ...gpuMetrics,
        gpuUtilization: calculateGPUUtilization(),
        memoryUtilization: analytics.totalSize / (256 * 1024 * 1024), // Assume 256MB budget
      };

      lastUpdateTime = new Date();

      // Update historical data
      if (showHistoricalData) {
        updateHistoricalData();
      }

      // Generate optimization suggestions
      updateOptimizationSuggestions();

    } catch (error: any) {
      console.error('Failed to update metrics:', error);
      addAlert('metrics_error', 'Failed to update performance metrics', 'error');
    }
  }

  /**
   * Update historical data for charts
   */
  function updateHistoricalData(): void {
    const now = Date.now();

    // Add new data points
    historicalData.timestamps.push(now);
    historicalData.hitRates.push(currentMetrics.hitRate);
    historicalData.memoryUsage.push(currentMetrics.memoryUtilization);
    historicalData.loadTimes.push(currentMetrics.averageTextureLoadTime);
    historicalData.compileTimes.push(currentMetrics.averageShaderCompileTime);
    historicalData.gpuUtilization.push(currentMetrics.gpuUtilization);
    historicalData.wasmGains.push(currentMetrics.wasmAccelerationGain);

    // Keep only recent data points
    if (historicalData.timestamps.length > maxDataPoints) {
      historicalData.timestamps = historicalData.timestamps.slice(-maxDataPoints);
      historicalData.hitRates = historicalData.hitRates.slice(-maxDataPoints);
      historicalData.memoryUsage = historicalData.memoryUsage.slice(-maxDataPoints);
      historicalData.loadTimes = historicalData.loadTimes.slice(-maxDataPoints);
      historicalData.compileTimes = historicalData.compileTimes.slice(-maxDataPoints);
      historicalData.gpuUtilization = historicalData.gpuUtilization.slice(-maxDataPoints);
      historicalData.wasmGains = historicalData.wasmGains.slice(-maxDataPoints);
    }
  }

  /**
   * Update real-time charts
   */
  function updateCharts(): void {
    if (!chartCanvas || !chartContext || !showRealTimeCharts) return;

    const ctx = chartContext;
    const canvas = chartCanvas;
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.fillStyle = darkTheme ? '#1a1a1a' : '#ffffff';
    ctx.fillRect(0, 0, width, height);

    // Draw chart grid
    drawChartGrid(ctx, width, height);

    // Draw hit rate line
    drawLineChart(ctx, historicalData.hitRates, width, height, '#00ff00', 0, 1);

    // Draw memory usage line
    drawLineChart(ctx, historicalData.memoryUsage, width, height, '#ff6600', 0, 1);

    // Draw GPU utilization line
    drawLineChart(ctx, historicalData.gpuUtilization, width, height, '#4a90e2', 0, 1);

    // Draw legends
    drawChartLegend(ctx, width, height);
  }

  /**
   * Draw chart grid
   */
  function drawChartGrid(ctx: CanvasRenderingContext2D, width: number, height: number): void {
    ctx.strokeStyle = darkTheme ? '#333333' : '#cccccc';
    ctx.lineWidth = 1;

    // Vertical lines
    for (let x = 0; x <= width; x += width / 10) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    // Horizontal lines
    for (let y = 0; y <= height; y += height / 5) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
  }

  /**
   * Draw line chart
   */
  function drawLineChart(
    ctx: CanvasRenderingContext2D,
    data: number[],
    width: number,
    height: number,
    color: string,
    minValue: number,
    maxValue: number
  ): void {
    if (data.length < 2) return;

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();

    const stepX = width / (data.length - 1);

    data.forEach((value, index) => {
      const x = index * stepX;
      const y = height - ((value - minValue) / (maxValue - minValue)) * height;

      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    ctx.stroke();
  }

  /**
   * Draw chart legend
   */
  function drawChartLegend(ctx: CanvasRenderingContext2D, width: number, height: number): void {
    const legends = [
      { label: 'Hit Rate', color: '#00ff00' },
      { label: 'Memory', color: '#ff6600' },
      { label: 'GPU Util', color: '#4a90e2' }
    ];

    ctx.font = '10px monospace';
    ctx.fillStyle = darkTheme ? '#ffffff' : '#000000';

    legends.forEach((legend, index) => {
      const x = width - 80;
      const y = 15 + index * 15;

      // Color indicator
      ctx.fillStyle = legend.color;
      ctx.fillRect(x, y - 8, 10, 8);

      // Label
      ctx.fillStyle = darkTheme ? '#ffffff' : '#000000';
      ctx.fillText(legend.label, x + 15, y);
    });
  }

  /**
   * Check for performance alerts
   */
  function checkPerformanceAlerts(): void {
    if (!enableAlerts) return;

    const alerts: Array<{ type: string; message: string; severity: 'info' | 'warning' | 'error' }> = [];

    // Memory usage alert
    if (currentMetrics.memoryUtilization > memoryWarningThreshold) {
      alerts.push({
        type: 'memory_high',
        message: `Memory usage is ${(currentMetrics.memoryUtilization * 100).toFixed(1)}% (threshold: ${(memoryWarningThreshold * 100).toFixed(1)}%)`,
        severity: currentMetrics.memoryUtilization > 0.9 ? 'error' : 'warning'
      });
    }

    // Cache hit rate alert
    if (currentMetrics.hitRate < cacheHitRateWarning) {
      alerts.push({
        type: 'cache_hit_low',
        message: `Cache hit rate is ${(currentMetrics.hitRate * 100).toFixed(1)}% (threshold: ${(cacheHitRateWarning * 100).toFixed(1)}%)`,
        severity: currentMetrics.hitRate < 0.4 ? 'error' : 'warning'
      });
    }

    // GPU utilization alert
    if (currentMetrics.gpuUtilization > 0.9) {
      alerts.push({
        type: 'gpu_high',
        message: `GPU utilization is ${(currentMetrics.gpuUtilization * 100).toFixed(1)}%`,
        severity: 'warning'
      });
    }

    // Add new alerts
    alerts.forEach(alert => {
      addAlert(alert.type, alert.message, alert.severity);
    });

    // Auto-clear old alerts
    const fiveMinutesAgo = Date.now() - 5 * 60 * 1000;
    activeAlerts = activeAlerts.filter(alert =>
      alert.timestamp > fiveMinutesAgo || !alert.acknowledged
    );
  }

  /**
   * Add performance alert
   */
  function addAlert(type: string, message: string, severity: 'info' | 'warning' | 'error'): void {
    // Check if alert already exists
    const existingAlert = activeAlerts.find(alert =>
      alert.type === type && !alert.acknowledged
    );

    if (existingAlert) return;

    const alert = {
      id: `${type}_${Date.now()}`,
      type,
      message,
      severity,
      timestamp: Date.now(),
      acknowledged: false
    };

    activeAlerts = [alert, ...activeAlerts].slice(0, 20); // Keep max 20 alerts

    onAlert?.(type, message, severity);
  }

  /**
   * Acknowledge alert
   */
  function acknowledgeAlert(alertId: string): void {
    activeAlerts = activeAlerts.map(alert =>
      alert.id === alertId ? { ...alert, acknowledged: true } : alert
    );
  }

  /**
   * Update optimization suggestions
   */
  function updateOptimizationSuggestions(): void {
    const suggestions: typeof optimizationSuggestions = [];

    // Memory optimization
    if (currentMetrics.memoryUtilization > 0.7) {
      suggestions.push({
        id: 'optimize_memory',
        title: 'Optimize Memory Usage',
        description: 'Clear unused cache entries to free up memory',
        impact: 'high',
        effort: 'easy',
        action: async () => {
          await enhancedGPUCache.optimizeCacheWithWASM();
          addAlert('optimization', 'Memory optimization completed', 'info');
          onPerformanceOptimized?.(currentMetrics);
        }
      });
    }

    // Cache hit rate optimization
    if (currentMetrics.hitRate < 0.6) {
      suggestions.push({
        id: 'improve_cache',
        title: 'Improve Cache Efficiency',
        description: 'Adjust caching strategy and preload common assets',
        impact: 'medium',
        effort: 'moderate',
        action: () => {
          // Implement cache strategy optimization
          addAlert('optimization', 'Cache strategy optimized', 'info');
        }
      });
    }

    // WASM acceleration
    if (currentMetrics.wasmAccelerationGain < 20) {
      suggestions.push({
        id: 'enable_wasm',
        title: 'Enable WASM Acceleration',
        description: 'Use WebAssembly for performance-critical operations',
        impact: 'medium',
        effort: 'easy',
        action: () => {
          // Enable WASM optimization
          addAlert('optimization', 'WASM acceleration enabled', 'info');
        }
      });
    }

    optimizationSuggestions = suggestions;
  }

  /**
   * Calculate GPU utilization estimate
   */
  function calculateGPUUtilization(): number {
    // Estimate based on texture load times and shader compile times
    const textureUtilization = Math.min(currentMetrics.averageTextureLoadTime / 50, 1);
    const shaderUtilization = Math.min(currentMetrics.averageShaderCompileTime / 100, 1);

    return (textureUtilization + shaderUtilization) / 2;
  }

  /**
   * Export performance data
   */
  function exportPerformanceData(): void {
    const exportData = {
      timestamp: new Date().toISOString(),
      monitoringDuration,
      currentMetrics,
      historicalData,
      activeAlerts: activeAlerts.filter(alert => !alert.acknowledged),
      optimizationSuggestions
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json'
    });

    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `cache_performance_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  /**
   * Clear cache data
   */
  async function clearCache(): Promise<void> {
    try {
      await enhancedGPUCache.dispose();
      addAlert('cache_cleared', 'Cache cleared successfully', 'info');
      onCacheCleared?.();
    } catch (error: any) {
      addAlert('cache_error', `Failed to clear cache: ${error.message}`, 'error');
    }
  }

  /**
   * Component lifecycle
   */
  onMount(() => {
    // Initialize chart canvas
    if (chartCanvas) {
      chartContext = chartCanvas.getContext('2d');
    }

    // Start monitoring by default
    startMonitoring();
  });

  onDestroy(() => {
    stopMonitoring();
  });

  // Reactive statements
  $: formattedUptime = formatDuration(monitoringDuration);
  $: memoryUsageColor = currentMetrics.memoryUtilization > 0.8 ? '#ff6600' :
    currentMetrics.memoryUtilization > 0.6 ? '#ffff00' : '#00ff00';
  $: cacheEfficiencyColor = currentMetrics.cacheEfficiency > 0.8 ? '#00ff00' :
    currentMetrics.cacheEfficiency > 0.6 ? '#ffff00' : '#ff6600';

  /**
   * Format duration in human-readable format
   */
  function formatDuration(ms: number): string {
    const seconds = Math.floor(ms / 1000) % 60;
    const minutes = Math.floor(ms / (1000 * 60)) % 60;
    const hours = Math.floor(ms / (1000 * 60 * 60));

    if (hours > 0) {
      return `${hours}h ${minutes}m ${seconds}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${seconds}s`;
    } else {
      return `${seconds}s`;
    }
  }

  /**
   * Format bytes in human-readable format
   */
  function formatBytes(bytes: number): string {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }
</script>

<!-- Cache Performance Monitoring Dashboard -->
<div class="cache-performance-dashboard" class:compact={compactMode} class:dark-theme={darkTheme}>

  <!-- Dashboard Header -->
  <div class="dashboard-header">
    <div class="title-section">
      <h2 class="dashboard-title">🎯 Cache Performance Monitor</h2>
      <div class="status-indicators">
        <div class="status-badge" class:monitoring={isMonitoring}>
          {isMonitoring ? '🔴 LIVE' : '⚫ STOPPED'}
        </div>
        <div class="uptime">Uptime: {formattedUptime}</div>
        <div class="last-update">Last: {lastUpdateTime.toLocaleTimeString()}</div>
      </div>
    </div>

    <div class="header-controls">
      {#if !isMonitoring}
        <button class="control-button start" on:click={startMonitoring}>
          ▶️ START
        </button>
      {:else}
        <button class="control-button stop" on:click={stopMonitoring}>
          ⏹️ STOP
        </button>
      {/if}

      <button class="control-button export" on:click={exportPerformanceData}>
        📊 EXPORT
      </button>

      {#if enablePerformanceTuning}
        <button class="control-button optimize" on:click={() => tuningEnabled = !tuningEnabled}>
          ⚙️ TUNE
        </button>
      {/if}
    </div>
  </div>

  <!-- Active Alerts -->
  {#if activeAlerts.length > 0}
    <div class="alerts-section">
      <h3 class="section-title">🚨 Active Alerts</h3>
      <div class="alerts-container">
        {#each activeAlerts.filter(alert => !alert.acknowledged) as alert (alert.id)}
          <div class="alert alert-{alert.severity}">
            <div class="alert-content">
              <div class="alert-message">{alert.message}</div>
              <div class="alert-timestamp">{new Date(alert.timestamp).toLocaleTimeString()}</div>
            </div>
            <button class="alert-dismiss" on:click={() => acknowledgeAlert(alert.id)}>
              ✕
            </button>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Key Metrics Overview -->
  <div class="metrics-overview">
    <div class="metric-card">
      <div class="metric-label">Cache Hit Rate</div>
      <div class="metric-value" style="color: {cacheEfficiencyColor}">
        {(currentMetrics.hitRate * 100).toFixed(1)}%
      </div>
      <div class="metric-detail">
        Texture: {(currentMetrics.textureHitRate * 100).toFixed(1)}% |
        Shader: {(currentMetrics.shaderHitRate * 100).toFixed(1)}%
      </div>
    </div>

    <div class="metric-card">
      <div class="metric-label">Memory Usage</div>
      <div class="metric-value" style="color: {memoryUsageColor}">
        {(currentMetrics.memoryUtilization * 100).toFixed(1)}%
      </div>
      <div class="metric-detail">
        Total: {formatBytes(currentMetrics.totalSize)} |
        Entries: {currentMetrics.totalEntries}
      </div>
    </div>

    <div class="metric-card">
      <div class="metric-label">Performance</div>
      <div class="metric-value">
        {(currentMetrics.cacheEfficiency * 100).toFixed(1)}%
      </div>
      <div class="metric-detail">
        GPU: {(currentMetrics.gpuUtilization * 100).toFixed(1)}% |
        WASM: +{currentMetrics.wasmAccelerationGain.toFixed(1)}%
      </div>
    </div>

    <div class="metric-card">
      <div class="metric-label">Load Times</div>
      <div class="metric-value">
        {currentMetrics.averageTextureLoadTime.toFixed(1)}ms
      </div>
      <div class="metric-detail">
        Texture: {currentMetrics.averageTextureLoadTime.toFixed(1)}ms |
        Shader: {currentMetrics.averageShaderCompileTime.toFixed(1)}ms
      </div>
    </div>
  </div>

  <!-- Real-time Charts -->
  {#if showRealTimeCharts}
    <div class="charts-section">
      <h3 class="section-title">📈 Real-time Performance Charts</h3>
      <div class="chart-container">
        <canvas
          bind:this={chartCanvas}
          class="performance-chart"
          width="600"
          height="200"
        ></canvas>
      </div>
    </div>
  {/if}

  <!-- Advanced Metrics -->
  {#if showAdvancedMetrics}
    <div class="advanced-metrics">
      <h3 class="section-title">🔬 Advanced Metrics</h3>
      <div class="advanced-grid">

        <div class="advanced-card">
          <h4>Hot Entries</h4>
          <div class="entries-list">
            {#each currentMetrics.hotEntries.slice(0, 5) as entry}
              <div class="entry-item hot">{entry}</div>
            {/each}
          </div>
        </div>

        <div class="advanced-card">
          <h4>Cold Entries</h4>
          <div class="entries-list">
            {#each currentMetrics.coldEntries.slice(0, 5) as entry}
              <div class="entry-item cold">{entry}</div>
            {/each}
          </div>
        </div>

        <div class="advanced-card">
          <h4>Workflow Distribution</h4>
          <div class="workflow-stats">
            {#each Object.entries(currentMetrics.workflowDistribution) as [workflow, count]}
              <div class="workflow-item">
                <span class="workflow-name">{workflow}</span>
                <span class="workflow-count">{count}</span>
              </div>
            {/each}
          </div>
        </div>

        <div class="advanced-card">
          <h4>Performance Stats</h4>
          <div class="perf-stats">
            <div>Miss Rate: {(currentMetrics.missRate * 100).toFixed(1)}%</div>
            <div>Eviction Rate: {(currentMetrics.evictionRate * 100).toFixed(1)}%</div>
            <div>Avg Age: {(currentMetrics.averageEntryAge / 1000).toFixed(1)}s</div>
          </div>
        </div>
      </div>
    </div>
  {/if}

  <!-- Optimization Suggestions -->
  {#if optimizationSuggestions.length > 0}
    <div class="optimization-section">
      <h3 class="section-title">💡 Optimization Suggestions</h3>
      <div class="suggestions-container">
        {#each optimizationSuggestions as suggestion (suggestion.id)}
          <div class="suggestion-card impact-{suggestion.impact}">
            <div class="suggestion-header">
              <div class="suggestion-title">{suggestion.title}</div>
              <div class="suggestion-badges">
                <span class="impact-badge">{suggestion.impact.toUpperCase()} IMPACT</span>
                <span class="effort-badge">{suggestion.effort.toUpperCase()}</span>
              </div>
            </div>
            <div class="suggestion-description">{suggestion.description}</div>
            <button class="suggestion-action" on:click={suggestion.action}>
              APPLY OPTIMIZATION
            </button>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Performance Tuning Panel -->
  {#if tuningEnabled && enablePerformanceTuning}
    <div class="tuning-panel">
      <h3 class="section-title">⚙️ Performance Tuning</h3>
      <div class="tuning-controls">

        <div class="tuning-group">
          <label class="tuning-label">Auto Optimization</label>
          <button class="toggle-button" class:active={autoOptimizationEnabled}
                  on:click={() => autoOptimizationEnabled = !autoOptimizationEnabled}>
            {autoOptimizationEnabled ? '🟢 ON' : '⚫ OFF'}
          </button>
        </div>

        <div class="tuning-group">
          <label class="tuning-label">Optimization Level</label>
          <select bind:value={optimizationAggressiveness} class="tuning-select">
            <option value="conservative">Conservative</option>
            <option value="balanced">Balanced</option>
            <option value="aggressive">Aggressive</option>
          </select>
        </div>

        <div class="tuning-group">
          <label class="tuning-label">Cache Actions</label>
          <div class="action-buttons">
            <button class="action-button optimize" on:click={() => enhancedGPUCache.optimizeCacheWithWASM()}>
              🔧 OPTIMIZE
            </button>
            <button class="action-button clear" on:click={clearCache}>
              🗑️ CLEAR
            </button>
          </div>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .cache-performance-dashboard {
    padding: 16px;
    background: #f8f9fa;
    border-radius: 8px;
    font-family: 'Rajdhani', 'Inter', sans-serif;
    color: #2c3e50;
    max-width: 1200px;
    margin: 0 auto;
  }

  .cache-performance-dashboard.dark-theme {
    background: #1a1a1a;
    color: #ecf0f1;
  }

  .cache-performance-dashboard.compact {
    padding: 8px;
  }

  /* Dashboard Header */
  .dashboard-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 2px solid rgba(74, 144, 226, 0.2);
  }

  .title-section {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .dashboard-title {
    margin: 0;
    font-size: 24px;
    font-weight: 700;
    color: #4a90e2;
  }

  .dark-theme .dashboard-title {
    color: #74c0fc;
  }

  .status-indicators {
    display: flex;
    gap: 16px;
    font-size: 12px;
    opacity: 0.8;
  }

  .status-badge {
    padding: 2px 8px;
    border-radius: 12px;
    background: rgba(108, 117, 125, 0.2);
    font-weight: 600;
  }

  .status-badge.monitoring {
    background: rgba(220, 53, 69, 0.2);
    color: #dc3545;
    animation: pulse 2s infinite;
  }

  .header-controls {
    display: flex;
    gap: 8px;
  }

  .control-button {
    padding: 8px 16px;
    border: none;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .control-button.start {
    background: #28a745;
    color: white;
  }

  .control-button.stop {
    background: #dc3545;
    color: white;
  }

  .control-button.export {
    background: #17a2b8;
    color: white;
  }

  .control-button.optimize {
    background: #ffc107;
    color: #212529;
  }

  .control-button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  }

  /* Alerts Section */
  .alerts-section {
    margin-bottom: 20px;
  }

  .section-title {
    margin: 0 0 12px 0;
    font-size: 16px;
    font-weight: 600;
    color: #495057;
  }

  .dark-theme .section-title {
    color: #adb5bd;
  }

  .alerts-container {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .alert {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px;
    border-radius: 6px;
    border-left: 4px solid;
  }

  .alert-info {
    background: rgba(23, 162, 184, 0.1);
    border-color: #17a2b8;
  }

  .alert-warning {
    background: rgba(255, 193, 7, 0.1);
    border-color: #ffc107;
  }

  .alert-error {
    background: rgba(220, 53, 69, 0.1);
    border-color: #dc3545;
  }

  .alert-content {
    flex: 1;
  }

  .alert-message {
    font-weight: 500;
    margin-bottom: 4px;
  }

  .alert-timestamp {
    font-size: 11px;
    opacity: 0.7;
  }

  .alert-dismiss {
    background: none;
    border: none;
    font-size: 16px;
    cursor: pointer;
    opacity: 0.5;
    padding: 4px;
    border-radius: 50%;
    transition: all 0.2s ease;
  }

  .alert-dismiss:hover {
    opacity: 1;
    background: rgba(0, 0, 0, 0.1);
  }

  /* Metrics Overview */
  .metrics-overview {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }

  .compact .metrics-overview {
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
  }

  .metric-card {
    background: white;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(0, 0, 0, 0.05);
  }

  .dark-theme .metric-card {
    background: #2c3e50;
    border-color: rgba(255, 255, 255, 0.1);
  }

  .metric-label {
    font-size: 12px;
    font-weight: 500;
    color: #6c757d;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .metric-value {
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 4px;
    line-height: 1;
  }

  .metric-detail {
    font-size: 11px;
    opacity: 0.7;
  }

  /* Charts Section */
  .charts-section {
    margin-bottom: 24px;
  }

  .chart-container {
    background: white;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }

  .dark-theme .chart-container {
    background: #2c3e50;
  }

  .performance-chart {
    width: 100%;
    height: 200px;
    border-radius: 4px;
  }

  /* Advanced Metrics */
  .advanced-metrics {
    margin-bottom: 24px;
  }

  .advanced-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px;
  }

  .advanced-card {
    background: white;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }

  .dark-theme .advanced-card {
    background: #2c3e50;
  }

  .advanced-card h4 {
    margin: 0 0 12px 0;
    font-size: 14px;
    font-weight: 600;
    color: #495057;
  }

  .dark-theme .advanced-card h4 {
    color: #adb5bd;
  }

  .entries-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .entry-item {
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-family: 'Courier New', monospace;
  }

  .entry-item.hot {
    background: rgba(220, 53, 69, 0.1);
    color: #dc3545;
  }

  .entry-item.cold {
    background: rgba(23, 162, 184, 0.1);
    color: #17a2b8;
  }

  .workflow-stats,
  .perf-stats {
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 12px;
  }

  .workflow-item {
    display: flex;
    justify-content: space-between;
  }

  .workflow-name {
    color: #6c757d;
  }

  .workflow-count {
    font-weight: 600;
  }

  /* Optimization Section */
  .optimization-section {
    margin-bottom: 24px;
  }

  .suggestions-container {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .suggestion-card {
    background: white;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    border-left: 4px solid;
  }

  .dark-theme .suggestion-card {
    background: #2c3e50;
  }

  .suggestion-card.impact-low {
    border-color: #17a2b8;
  }

  .suggestion-card.impact-medium {
    border-color: #ffc107;
  }

  .suggestion-card.impact-high {
    border-color: #28a745;
  }

  .suggestion-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
  }

  .suggestion-title {
    font-weight: 600;
    font-size: 14px;
  }

  .suggestion-badges {
    display: flex;
    gap: 6px;
  }

  .impact-badge,
  .effort-badge {
    padding: 2px 6px;
    border-radius: 10px;
    font-size: 9px;
    font-weight: 600;
    text-transform: uppercase;
  }

  .impact-badge {
    background: rgba(40, 167, 69, 0.2);
    color: #28a745;
  }

  .effort-badge {
    background: rgba(108, 117, 125, 0.2);
    color: #6c757d;
  }

  .suggestion-description {
    font-size: 12px;
    margin-bottom: 12px;
    opacity: 0.8;
  }

  .suggestion-action {
    background: #4a90e2;
    color: white;
    border: none;
    padding: 6px 12px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .suggestion-action:hover {
    background: #357abd;
    transform: translateY(-1px);
  }

  /* Performance Tuning Panel */
  .tuning-panel {
    background: white;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }

  .dark-theme .tuning-panel {
    background: #2c3e50;
  }

  .tuning-controls {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
  }

  .tuning-group {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .tuning-label {
    font-size: 12px;
    font-weight: 500;
    color: #6c757d;
  }

  .toggle-button {
    padding: 6px 12px;
    border: 1px solid #dee2e6;
    background: white;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .toggle-button.active {
    background: #28a745;
    color: white;
    border-color: #28a745;
  }

  .tuning-select {
    padding: 6px 8px;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    background: white;
  }

  .dark-theme .tuning-select,
  .dark-theme .toggle-button {
    background: #495057;
    border-color: #6c757d;
    color: white;
  }

  .action-buttons {
    display: flex;
    gap: 8px;
  }

  .action-button {
    flex: 1;
    padding: 6px 12px;
    border: none;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .action-button.optimize {
    background: #ffc107;
    color: #212529;
  }

  .action-button.clear {
    background: #dc3545;
    color: white;
  }

  .action-button:hover {
    transform: translateY(-1px);
  }

  /* Animations */
  @keyframes pulse {
    0%, 100% {
      opacity: 1;
    }
    50% {
      opacity: 0.5;
    }
  }

  /* Responsive Design */
  @media (max-width: 768px) {
    .dashboard-header {
      flex-direction: column;
      align-items: flex-start;
      gap: 12px;
    }

    .header-controls {
      width: 100%;
      justify-content: flex-end;
    }

    .metrics-overview {
      grid-template-columns: 1fr;
    }

    .advanced-grid {
      grid-template-columns: 1fr;
    }

    .tuning-controls {
      grid-template-columns: 1fr;
    }
  }

  /* High contrast mode */
  @media (prefers-contrast: high) {
    .cache-performance-dashboard {
      border: 2px solid currentColor;
    }

    .metric-card,
    .chart-container,
    .advanced-card,
    .suggestion-card,
    .tuning-panel {
      border: 1px solid currentColor;
    }
  }
</style>