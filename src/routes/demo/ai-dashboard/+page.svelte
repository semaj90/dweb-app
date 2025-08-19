<!--
  AI Dashboard Demo
  Real-time AI performance monitoring and analytics
-->

<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { writable } from 'svelte/store';

  // Mock AI metrics
  const aiMetrics = writable({
    models: {
      total: 5,
      active: 3,
      idle: 1,
      offline: 1
    },
    performance: {
      averageLatency: 245,
      throughput: 156,
      accuracy: 94.7,
      uptime: 99.2
    },
    usage: {
      requests: 12847,
      successful: 12153,
      failed: 694,
      cached: 3421
    },
    resources: {
      cpuUsage: 67,
      memoryUsage: 78,
      gpuUsage: 84,
      diskUsage: 45
    }
  });

  // Live activity feed
  let activities = [
    { id: 1, type: 'query', model: 'Gemma 3 Legal', status: 'completed', time: new Date(Date.now() - 30000), latency: 234 },
    { id: 2, type: 'embedding', model: 'Nomic Embed', status: 'completed', time: new Date(Date.now() - 45000), latency: 89 },
    { id: 3, type: 'analysis', model: 'Legal Classifier', status: 'processing', time: new Date(Date.now() - 60000), latency: null },
    { id: 4, type: 'query', model: 'Gemma 3 Legal', status: 'completed', time: new Date(Date.now() - 75000), latency: 267 },
    { id: 5, type: 'vectorization', model: 'Document Processor', status: 'failed', time: new Date(Date.now() - 90000), latency: null }
  ];

  // Model status
  let modelStatuses = [
    {
      name: 'Gemma 3 Legal',
      status: 'active',
      load: 85,
      requests: 3421,
      avgLatency: 245,
      accuracy: 94.2,
      memory: '2.1 GB',
      gpu: true
    },
    {
      name: 'Nomic Embed Text',
      status: 'active',
      load: 67,
      requests: 8934,
      avgLatency: 89,
      accuracy: 98.7,
      memory: '512 MB',
      gpu: true
    },
    {
      name: 'Legal Document Classifier',
      status: 'idle',
      load: 12,
      requests: 234,
      avgLatency: 156,
      accuracy: 91.4,
      memory: '256 MB',
      gpu: false
    },
    {
      name: 'Evidence Analyzer',
      status: 'active',
      load: 43,
      requests: 567,
      avgLatency: 334,
      accuracy: 89.8,
      memory: '1.8 GB',
      gpu: true
    },
    {
      name: 'Case Summary Generator',
      status: 'offline',
      load: 0,
      requests: 0,
      avgLatency: 0,
      accuracy: 0,
      memory: '0 MB',
      gpu: false
    }
  ];

  // Performance history for charts
  let performanceHistory = Array.from({ length: 50 }, (_, i) => ({
    time: Date.now() - (49 - i) * 60000,
    latency: 200 + Math.sin(i * 0.1) * 50 + Math.random() * 30,
    throughput: 150 + Math.cos(i * 0.15) * 40 + Math.random() * 20,
    accuracy: 90 + Math.sin(i * 0.08) * 5 + Math.random() * 3,
    errors: Math.max(0, 5 + Math.sin(i * 0.2) * 3 + Math.random() * 2)
  }));

  let updateInterval: number;
  let selectedTimeRange = '1h';
  let autoRefresh = true;

  // Utility functions
  function getStatusColor(status: string): string {
    switch (status) {
      case 'active': return 'var(--yorha-success, #00ff41)';
      case 'idle': return 'var(--yorha-accent-warm, #ffbf00)';
      case 'offline': return 'var(--yorha-text-secondary, #666)';
      case 'processing': return 'var(--yorha-accent-blue, #00bcd4)';
      case 'completed': return 'var(--yorha-success, #00ff41)';
      case 'failed': return 'var(--yorha-danger, #ff4757)';
      default: return 'var(--yorha-text-secondary, #a0a0a0)';
    }
  }

  function getTypeIcon(type: string): string {
    switch (type) {
      case 'query': return '💬';
      case 'embedding': return '🔤';
      case 'analysis': return '🔍';
      case 'vectorization': return '📊';
      default: return '⚙️';
    }
  }

  function formatTime(date: Date): string {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    
    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
  }

  function updateMetrics() {
    aiMetrics.update(current => ({
      ...current,
      performance: {
        ...current.performance,
        averageLatency: current.performance.averageLatency + (Math.random() - 0.5) * 20,
        throughput: Math.max(50, current.performance.throughput + (Math.random() - 0.5) * 30),
        accuracy: Math.min(99, Math.max(85, current.performance.accuracy + (Math.random() - 0.5) * 2))
      },
      resources: {
        cpuUsage: Math.max(20, Math.min(95, current.resources.cpuUsage + (Math.random() - 0.5) * 10)),
        memoryUsage: Math.max(30, Math.min(90, current.resources.memoryUsage + (Math.random() - 0.5) * 8)),
        gpuUsage: Math.max(40, Math.min(100, current.resources.gpuUsage + (Math.random() - 0.5) * 15)),
        diskUsage: Math.max(20, Math.min(80, current.resources.diskUsage + (Math.random() - 0.5) * 5))
      }
    }));

    // Add new performance data point
    const now = Date.now();
    performanceHistory = [
      ...performanceHistory.slice(1),
      {
        time: now,
        latency: 200 + Math.sin(now * 0.0001) * 50 + Math.random() * 30,
        throughput: 150 + Math.cos(now * 0.00015) * 40 + Math.random() * 20,
        accuracy: 90 + Math.sin(now * 0.00008) * 5 + Math.random() * 3,
        errors: Math.max(0, 5 + Math.sin(now * 0.0002) * 3 + Math.random() * 2)
      }
    ];

    // Occasionally add new activity
    if (Math.random() < 0.3) {
      const newActivity = {
        id: Date.now(),
        type: ['query', 'embedding', 'analysis', 'vectorization'][Math.floor(Math.random() * 4)],
        model: ['Gemma 3 Legal', 'Nomic Embed', 'Legal Classifier', 'Document Processor'][Math.floor(Math.random() * 4)],
        status: ['completed', 'processing', 'failed'][Math.floor(Math.random() * 3)],
        time: new Date(),
        latency: Math.random() < 0.8 ? Math.floor(Math.random() * 400 + 50) : null
      };
      
      activities = [newActivity, ...activities.slice(0, 9)];
    }
  }

  function toggleAutoRefresh() {
    autoRefresh = !autoRefresh;
    if (autoRefresh) {
      updateInterval = setInterval(updateMetrics, 3000);
    } else if (updateInterval) {
      clearInterval(updateInterval);
    }
  }

  onMount(() => {
    if (autoRefresh) {
      updateInterval = setInterval(updateMetrics, 3000);
    }
  });

  onDestroy(() => {
    if (updateInterval) {
      clearInterval(updateInterval);
    }
  });

  // Reactive statements
  $: successRate = ($aiMetrics.usage.successful / $aiMetrics.usage.requests * 100).toFixed(1);
  $: errorRate = ($aiMetrics.usage.failed / $aiMetrics.usage.requests * 100).toFixed(1);
</script>

<div class="ai-dashboard">
  <div class="dashboard-header">
    <h1 class="page-title">AI Performance Dashboard</h1>
    <div class="dashboard-controls">
      <select bind:value={selectedTimeRange} class="time-select">
        <option value="15m">Last 15 minutes</option>
        <option value="1h">Last hour</option>
        <option value="6h">Last 6 hours</option>
        <option value="24h">Last 24 hours</option>
      </select>
      <button 
        class="control-btn" 
        class:active={autoRefresh} 
        onclick={toggleAutoRefresh}
      >
        {autoRefresh ? '⏸️' : '▶️'} Auto Refresh
      </button>
    </div>
  </div>

  <!-- Key Metrics Grid -->
  <div class="metrics-grid">
    <div class="metric-card primary">
      <div class="metric-header">
        <h3>Active Models</h3>
        <span class="metric-icon">🧠</span>
      </div>
      <div class="metric-value">{$aiMetrics.models.active}</div>
      <div class="metric-detail">of {$aiMetrics.models.total} total models</div>
      <div class="metric-trend positive">↗ +2 since yesterday</div>
    </div>

    <div class="metric-card secondary">
      <div class="metric-header">
        <h3>Avg Latency</h3>
        <span class="metric-icon">⚡</span>
      </div>
      <div class="metric-value">{Math.round($aiMetrics.performance.averageLatency)}ms</div>
      <div class="metric-detail">response time</div>
      <div class="metric-trend negative">↘ +15ms from last hour</div>
    </div>

    <div class="metric-card success">
      <div class="metric-header">
        <h3>Accuracy</h3>
        <span class="metric-icon">🎯</span>
      </div>
      <div class="metric-value">{$aiMetrics.performance.accuracy.toFixed(1)}%</div>
      <div class="metric-detail">average across all models</div>
      <div class="metric-trend positive">↗ +0.3% improvement</div>
    </div>

    <div class="metric-card warning">
      <div class="metric-header">
        <h3>Total Requests</h3>
        <span class="metric-icon">📊</span>
      </div>
      <div class="metric-value">{$aiMetrics.usage.requests.toLocaleString()}</div>
      <div class="metric-detail">{successRate}% success rate</div>
      <div class="metric-trend positive">↗ +1.2k today</div>
    </div>
  </div>

  <!-- Performance Charts -->
  <div class="charts-section">
    <div class="chart-container">
      <div class="chart-header">
        <h2>Performance Over Time</h2>
        <div class="chart-legend">
          <div class="legend-item">
            <div class="legend-color latency"></div>
            <span>Latency (ms)</span>
          </div>
          <div class="legend-item">
            <div class="legend-color throughput"></div>
            <span>Throughput (req/min)</span>
          </div>
          <div class="legend-item">
            <div class="legend-color accuracy"></div>
            <span>Accuracy (%)</span>
          </div>
        </div>
      </div>
      
      <div class="chart-area">
        <svg class="performance-chart" viewBox="0 0 1000 300">
          <!-- Chart background -->
          <rect width="1000" height="300" fill="var(--yorha-bg-primary, #0a0a0a)" />
          
          <!-- Grid lines -->
          {#each Array(6) as _, i}
            <line 
              x1="0" 
              y1={50 * i} 
              x2="1000" 
              y2={50 * i} 
              stroke="var(--yorha-border, #333)" 
              stroke-width="0.5"
              opacity="0.5"
            />
          {/each}
          
          <!-- Latency line -->
          <polyline
            points={performanceHistory.map((point, i) => `${i * 20},${300 - point.latency * 0.8}`).join(' ')}
            fill="none"
            stroke="var(--yorha-accent-warm, #ffbf00)"
            stroke-width="2"
          />
          
          <!-- Throughput line -->
          <polyline
            points={performanceHistory.map((point, i) => `${i * 20},${300 - point.throughput * 1.2}`).join(' ')}
            fill="none"
            stroke="var(--yorha-success, #00ff41)"
            stroke-width="2"
          />
          
          <!-- Accuracy line (scaled) -->
          <polyline
            points={performanceHistory.map((point, i) => `${i * 20},${300 - point.accuracy * 3}`).join(' ')}
            fill="none"
            stroke="var(--yorha-accent-blue, #00bcd4)"
            stroke-width="2"
          />
        </svg>
      </div>
    </div>

    <!-- Resource Usage -->
    <div class="resources-grid">
      <div class="resource-card">
        <h3>CPU Usage</h3>
        <div class="resource-bar">
          <div 
            class="resource-fill cpu" 
            style="width: {$aiMetrics.resources.cpuUsage}%"
          ></div>
        </div>
        <div class="resource-value">{$aiMetrics.resources.cpuUsage}%</div>
      </div>

      <div class="resource-card">
        <h3>Memory</h3>
        <div class="resource-bar">
          <div 
            class="resource-fill memory" 
            style="width: {$aiMetrics.resources.memoryUsage}%"
          ></div>
        </div>
        <div class="resource-value">{$aiMetrics.resources.memoryUsage}%</div>
      </div>

      <div class="resource-card">
        <h3>GPU Usage</h3>
        <div class="resource-bar">
          <div 
            class="resource-fill gpu" 
            style="width: {$aiMetrics.resources.gpuUsage}%"
          ></div>
        </div>
        <div class="resource-value">{$aiMetrics.resources.gpuUsage}%</div>
      </div>

      <div class="resource-card">
        <h3>Disk I/O</h3>
        <div class="resource-bar">
          <div 
            class="resource-fill disk" 
            style="width: {$aiMetrics.resources.diskUsage}%"
          ></div>
        </div>
        <div class="resource-value">{$aiMetrics.resources.diskUsage}%</div>
      </div>
    </div>
  </div>

  <!-- Model Status and Live Activity -->
  <div class="content-grid">
    <!-- Model Status -->
    <div class="models-section">
      <div class="section-header">
        <h2>Model Status</h2>
        <span class="model-count">{$aiMetrics.models.active} / {$aiMetrics.models.total} Active</span>
      </div>
      
      <div class="models-list">
        {#each modelStatuses as model}
          <div class="model-item">
            <div class="model-header">
              <div class="model-info">
                <h4 class="model-name">{model.name}</h4>
                <div class="model-meta">
                  <span class="model-memory">{model.memory}</span>
                  <span class="model-gpu">{model.gpu ? '🎮' : '🖥️'} {model.gpu ? 'GPU' : 'CPU'}</span>
                </div>
              </div>
              <div 
                class="model-status" 
                style="color: {getStatusColor(model.status)}"
              >
                {model.status}
              </div>
            </div>
            
            <div class="model-metrics">
              <div class="model-metric">
                <span class="metric-label">Load</span>
                <div class="metric-bar-small">
                  <div 
                    class="metric-fill-small" 
                    style="width: {model.load}%; background: {getStatusColor(model.status)}"
                  ></div>
                </div>
                <span class="metric-value-small">{model.load}%</span>
              </div>
              
              <div class="model-stats">
                <div class="stat-item">
                  <span class="stat-label">Requests</span>
                  <span class="stat-value">{model.requests.toLocaleString()}</span>
                </div>
                <div class="stat-item">
                  <span class="stat-label">Avg Latency</span>
                  <span class="stat-value">{model.avgLatency}ms</span>
                </div>
                <div class="stat-item">
                  <span class="stat-label">Accuracy</span>
                  <span class="stat-value">{model.accuracy}%</span>
                </div>
              </div>
            </div>
          </div>
        {/each}
      </div>
    </div>

    <!-- Live Activity Feed -->
    <div class="activity-section">
      <div class="section-header">
        <h2>Live Activity</h2>
        <div class="activity-indicator">
          <div class="pulse-dot"></div>
          <span>Real-time</span>
        </div>
      </div>
      
      <div class="activity-feed">
        {#each activities as activity}
          <div class="activity-item">
            <div class="activity-icon">{getTypeIcon(activity.type)}</div>
            <div class="activity-content">
              <div class="activity-header">
                <span class="activity-type">{activity.type}</span>
                <span 
                  class="activity-status" 
                  style="color: {getStatusColor(activity.status)}"
                >
                  {activity.status}
                </span>
              </div>
              <div class="activity-details">
                <span class="activity-model">{activity.model}</span>
                {#if activity.latency}
                  <span class="activity-latency">{activity.latency}ms</span>
                {/if}
              </div>
              <div class="activity-time">{formatTime(activity.time)}</div>
            </div>
          </div>
        {/each}
      </div>
    </div>
  </div>
</div>

<style>
  .ai-dashboard {
    max-width: 1600px;
    margin: 0 auto;
    padding: 2rem;
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    color: var(--yorha-text-primary, #ffffff);
  }

  .dashboard-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 3rem;
  }

  .page-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--yorha-accent-warm, #ffbf00);
    margin: 0;
    text-transform: uppercase;
    letter-spacing: 2px;
  }

  .dashboard-controls {
    display: flex;
    gap: 1rem;
    align-items: center;
  }

  .time-select {
    background: var(--yorha-bg-secondary, #1a1a1a);
    border: 1px solid var(--yorha-border, #333);
    color: var(--yorha-text-primary, #ffffff);
    padding: 0.5rem 1rem;
    border-radius: 4px;
    font-family: inherit;
  }

  .time-select:focus {
    outline: none;
    border-color: var(--yorha-accent-warm, #ffbf00);
  }

  .control-btn {
    background: var(--yorha-bg-secondary, #1a1a1a);
    border: 1px solid var(--yorha-border, #333);
    color: var(--yorha-text-primary, #ffffff);
    padding: 0.5rem 1rem;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.2s ease;
    font-family: inherit;
  }

  .control-btn:hover,
  .control-btn.active {
    border-color: var(--yorha-accent-warm, #ffbf00);
    background: rgba(255, 191, 0, 0.1);
  }

  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 2rem;
    margin-bottom: 3rem;
  }

  .metric-card {
    background: var(--yorha-bg-secondary, #1a1a1a);
    border: 1px solid var(--yorha-border, #333);
    border-radius: 8px;
    padding: 2rem;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
  }

  .metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: var(--yorha-accent-warm, #ffbf00);
  }

  .metric-card.primary::before {
    background: var(--yorha-accent-warm, #ffbf00);
  }

  .metric-card.secondary::before {
    background: var(--yorha-accent-blue, #00bcd4);
  }

  .metric-card.success::before {
    background: var(--yorha-success, #00ff41);
  }

  .metric-card.warning::before {
    background: var(--yorha-warning, #ffa726);
  }

  .metric-card:hover {
    border-color: var(--yorha-accent-warm, #ffbf00);
    box-shadow: 0 4px 12px rgba(255, 191, 0, 0.15);
  }

  .metric-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }

  .metric-header h3 {
    margin: 0;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--yorha-text-secondary, #a0a0a0);
  }

  .metric-icon {
    font-size: 1.5rem;
  }

  .metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--yorha-text-primary, #ffffff);
    margin-bottom: 0.5rem;
  }

  .metric-detail {
    font-size: 0.9rem;
    color: var(--yorha-text-secondary, #a0a0a0);
    margin-bottom: 1rem;
  }

  .metric-trend {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
  }

  .metric-trend.positive {
    color: var(--yorha-success, #00ff41);
  }

  .metric-trend.negative {
    color: var(--yorha-danger, #ff4757);
  }

  .charts-section {
    background: var(--yorha-bg-secondary, #1a1a1a);
    border: 1px solid var(--yorha-border, #333);
    border-radius: 8px;
    padding: 2rem;
    margin-bottom: 3rem;
  }

  .chart-container {
    margin-bottom: 3rem;
  }

  .chart-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2rem;
  }

  .chart-header h2 {
    margin: 0;
    font-size: 1.3rem;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .chart-legend {
    display: flex;
    gap: 2rem;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.8rem;
  }

  .legend-color {
    width: 12px;
    height: 12px;
    border-radius: 2px;
  }

  .legend-color.latency {
    background: var(--yorha-accent-warm, #ffbf00);
  }

  .legend-color.throughput {
    background: var(--yorha-success, #00ff41);
  }

  .legend-color.accuracy {
    background: var(--yorha-accent-blue, #00bcd4);
  }

  .chart-area {
    border: 1px solid var(--yorha-border, #333);
    border-radius: 4px;
    overflow: hidden;
  }

  .performance-chart {
    width: 100%;
    height: 300px;
  }

  .resources-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 2rem;
  }

  .resource-card {
    text-align: center;
  }

  .resource-card h3 {
    margin: 0 0 1rem 0;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--yorha-text-secondary, #a0a0a0);
  }

  .resource-bar {
    width: 100%;
    height: 8px;
    background: var(--yorha-bg-primary, #0a0a0a);
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 0.5rem;
  }

  .resource-fill {
    height: 100%;
    transition: width 0.3s ease;
  }

  .resource-fill.cpu {
    background: var(--yorha-accent-warm, #ffbf00);
  }

  .resource-fill.memory {
    background: var(--yorha-accent-blue, #00bcd4);
  }

  .resource-fill.gpu {
    background: var(--yorha-success, #00ff41);
  }

  .resource-fill.disk {
    background: var(--yorha-warning, #ffa726);
  }

  .resource-value {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--yorha-text-primary, #ffffff);
  }

  .content-grid {
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 3rem;
  }

  .models-section,
  .activity-section {
    background: var(--yorha-bg-secondary, #1a1a1a);
    border: 1px solid var(--yorha-border, #333);
    border-radius: 8px;
    padding: 2rem;
  }

  .section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--yorha-border, #333);
  }

  .section-header h2 {
    margin: 0;
    font-size: 1.2rem;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .model-count {
    color: var(--yorha-accent-warm, #ffbf00);
    font-weight: 600;
  }

  .models-list {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }

  .model-item {
    background: var(--yorha-bg-primary, #0a0a0a);
    border: 1px solid var(--yorha-border, #333);
    border-radius: 6px;
    padding: 1.5rem;
    transition: all 0.2s ease;
  }

  .model-item:hover {
    border-color: var(--yorha-accent-warm, #ffbf00);
  }

  .model-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 1rem;
  }

  .model-name {
    margin: 0 0 0.5rem 0;
    font-size: 1rem;
    font-weight: 600;
  }

  .model-meta {
    display: flex;
    gap: 1rem;
    font-size: 0.8rem;
    color: var(--yorha-text-secondary, #a0a0a0);
  }

  .model-status {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
  }

  .model-metrics {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .model-metric {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .metric-label {
    min-width: 50px;
    font-size: 0.8rem;
    color: var(--yorha-text-secondary, #a0a0a0);
  }

  .metric-bar-small {
    flex: 1;
    height: 4px;
    background: var(--yorha-bg-secondary, #1a1a1a);
    border-radius: 2px;
    overflow: hidden;
  }

  .metric-fill-small {
    height: 100%;
    transition: width 0.3s ease;
  }

  .metric-value-small {
    min-width: 35px;
    text-align: right;
    font-size: 0.8rem;
    font-weight: 600;
  }

  .model-stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
  }

  .stat-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
  }

  .stat-label {
    font-size: 0.7rem;
    color: var(--yorha-text-secondary, #a0a0a0);
    text-transform: uppercase;
    margin-bottom: 0.25rem;
  }

  .stat-value {
    font-size: 0.9rem;
    font-weight: 600;
  }

  .activity-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: var(--yorha-success, #00ff41);
    font-size: 0.8rem;
  }

  .pulse-dot {
    width: 8px;
    height: 8px;
    background: var(--yorha-success, #00ff41);
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% {
      opacity: 1;
      transform: scale(1);
    }
    50% {
      opacity: 0.5;
      transform: scale(1.2);
    }
  }

  .activity-feed {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    max-height: 600px;
    overflow-y: auto;
  }

  .activity-item {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 1rem;
    background: var(--yorha-bg-primary, #0a0a0a);
    border: 1px solid var(--yorha-border, #333);
    border-radius: 6px;
    transition: all 0.2s ease;
  }

  .activity-item:hover {
    border-color: var(--yorha-accent-warm, #ffbf00);
  }

  .activity-icon {
    font-size: 1.2rem;
    margin-top: 0.2rem;
  }

  .activity-content {
    flex: 1;
  }

  .activity-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
  }

  .activity-type {
    font-size: 0.9rem;
    font-weight: 600;
    text-transform: capitalize;
  }

  .activity-status {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
  }

  .activity-details {
    display: flex;
    gap: 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.8rem;
    color: var(--yorha-text-secondary, #a0a0a0);
  }

  .activity-time {
    font-size: 0.7rem;
    color: var(--yorha-text-secondary, #666);
  }

  @media (max-width: 1200px) {
    .content-grid {
      grid-template-columns: 1fr;
    }

    .chart-legend {
      flex-direction: column;
      gap: 0.5rem;
    }
  }

  @media (max-width: 768px) {
    .ai-dashboard {
      padding: 1rem;
    }

    .dashboard-header {
      flex-direction: column;
      gap: 1rem;
      align-items: flex-start;
    }

    .page-title {
      font-size: 2rem;
    }

    .metrics-grid {
      grid-template-columns: 1fr;
    }

    .resources-grid {
      grid-template-columns: repeat(2, 1fr);
    }
  }
</style>