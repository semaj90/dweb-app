<!--
  AI Memory Dashboard
  Real-time monitoring of AI system memory usage and context management
-->

<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { writable } from 'svelte/store';

  // Memory metrics store
  const memoryMetrics = writable({
    system: {
      total: 16 * 1024 * 1024 * 1024, // 16GB
      used: 8 * 1024 * 1024 * 1024,   // 8GB
      free: 8 * 1024 * 1024 * 1024    // 8GB
    },
    gpu: {
      total: 8 * 1024 * 1024 * 1024,  // 8GB
      used: 3 * 1024 * 1024 * 1024,   // 3GB
      free: 5 * 1024 * 1024 * 1024    // 5GB
    },
    ai: {
      contextWindows: 4,
      activeModels: 3,
      embeddingCache: 245 * 1024 * 1024, // 245MB
      vectorStore: 1.2 * 1024 * 1024 * 1024 // 1.2GB
    },
    performance: {
      cpuUsage: 45,
      gpuUsage: 67,
      networkUsage: 23,
      diskUsage: 78
    }
  });

  // Context windows data
  let contextWindows = [
    {
      id: 'ctx_1',
      name: 'Legal Analysis Session',
      type: 'legal-chat',
      tokens: 4096,
      maxTokens: 8192,
      lastActive: new Date(Date.now() - 5 * 60 * 1000),
      status: 'active',
      memory: 156 * 1024 * 1024 // 156MB
    },
    {
      id: 'ctx_2', 
      name: 'Document Processing',
      type: 'document-analysis',
      tokens: 2048,
      maxTokens: 4096,
      lastActive: new Date(Date.now() - 15 * 60 * 1000),
      status: 'idle',
      memory: 89 * 1024 * 1024 // 89MB
    },
    {
      id: 'ctx_3',
      name: 'Case Research',
      type: 'research',
      tokens: 6144,
      maxTokens: 8192,
      lastActive: new Date(Date.now() - 2 * 60 * 1000),
      status: 'active',
      memory: 234 * 1024 * 1024 // 234MB
    }
  ];

  // Active AI models
  let aiModels = [
    {
      id: 'gemma3-legal',
      name: 'Gemma 3 Legal',
      type: 'language-model',
      status: 'running',
      memory: 2.1 * 1024 * 1024 * 1024, // 2.1GB
      gpu: true,
      requests: 1247,
      avgResponseTime: 890
    },
    {
      id: 'nomic-embed',
      name: 'Nomic Embeddings',
      type: 'embedding-model',
      status: 'running',
      memory: 512 * 1024 * 1024, // 512MB
      gpu: true,
      requests: 5632,
      avgResponseTime: 120
    },
    {
      id: 'legal-classifier',
      name: 'Legal Document Classifier',
      type: 'classification',
      status: 'standby',
      memory: 256 * 1024 * 1024, // 256MB
      gpu: false,
      requests: 89,
      avgResponseTime: 45
    }
  ];

  // Chart data for memory usage over time
  let memoryHistory = Array.from({ length: 50 }, (_, i) => ({
    time: Date.now() - (49 - i) * 1000,
    system: 45 + Math.sin(i * 0.1) * 10 + Math.random() * 5,
    gpu: 35 + Math.cos(i * 0.15) * 15 + Math.random() * 8,
    ai: 20 + Math.sin(i * 0.08) * 8 + Math.random() * 3
  }));

  let updateInterval: number;
  let selectedTimeRange = '1h';
  let selectedModel = 'all';
  let autoRefresh = true;

  // Utility functions
  function formatBytes(bytes: number): string {
    const sizes = ['B', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 B';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  }

  function formatPercent(used: number, total: number): number {
    return Math.round((used / total) * 100);
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'active': return 'var(--yorha-success, #00ff41)';
      case 'idle': return 'var(--yorha-accent-warm, #ffbf00)';
      case 'running': return 'var(--yorha-success, #00ff41)';
      case 'standby': return 'var(--yorha-text-secondary, #a0a0a0)';
      default: return 'var(--yorha-text-secondary, #a0a0a0)';
    }
  }

  function getTypeIcon(type: string): string {
    switch (type) {
      case 'legal-chat': return '💬';
      case 'document-analysis': return '📄';
      case 'research': return '🔍';
      case 'language-model': return '🧠';
      case 'embedding-model': return '🔤';
      case 'classification': return '🏷️';
      default: return '⚙️';
    }
  }

  // Update memory metrics
  function updateMetrics() {
    memoryMetrics.update(current => {
      // Simulate realistic memory usage changes
      const systemVariation = (Math.random() - 0.5) * 0.1 * 1024 * 1024 * 1024;
      const gpuVariation = (Math.random() - 0.5) * 0.2 * 1024 * 1024 * 1024;
      
      return {
        ...current,
        system: {
          ...current.system,
          used: Math.max(
            4 * 1024 * 1024 * 1024, 
            Math.min(
              14 * 1024 * 1024 * 1024,
              current.system.used + systemVariation
            )
          )
        },
        gpu: {
          ...current.gpu,
          used: Math.max(
            1024 * 1024 * 1024,
            Math.min(
              7 * 1024 * 1024 * 1024,
              current.gpu.used + gpuVariation
            )
          )
        },
        performance: {
          cpuUsage: Math.max(20, Math.min(90, current.performance.cpuUsage + (Math.random() - 0.5) * 10)),
          gpuUsage: Math.max(15, Math.min(95, current.performance.gpuUsage + (Math.random() - 0.5) * 15)),
          networkUsage: Math.max(5, Math.min(80, current.performance.networkUsage + (Math.random() - 0.5) * 20)),
          diskUsage: Math.max(60, Math.min(90, current.performance.diskUsage + (Math.random() - 0.5) * 5))
        }
      };
    });

    // Update memory history
    const now = Date.now();
    memoryHistory = [
      ...memoryHistory.slice(1),
      {
        time: now,
        system: 45 + Math.sin(now * 0.0001) * 10 + Math.random() * 5,
        gpu: 35 + Math.cos(now * 0.00015) * 15 + Math.random() * 8,
        ai: 20 + Math.sin(now * 0.00008) * 8 + Math.random() * 3
      }
    ];
  }

  function clearContext(contextId: string) {
    contextWindows = contextWindows.filter(ctx => ctx.id !== contextId);
  }

  function restartModel(modelId: string) {
    aiModels = aiModels.map(model => 
      model.id === modelId 
        ? { ...model, status: 'running', requests: 0 }
        : model
    );
  }

  function optimizeMemory() {
    // Simulate memory optimization
    memoryMetrics.update(current => ({
      ...current,
      system: {
        ...current.system,
        used: current.system.used * 0.85
      },
      gpu: {
        ...current.gpu,
        used: current.gpu.used * 0.8
      },
      ai: {
        ...current.ai,
        embeddingCache: current.ai.embeddingCache * 0.7
      }
    }));
  }

  onMount(() => {
    if (autoRefresh) {
      updateInterval = setInterval(updateMetrics, 2000);
    }
  });

  onDestroy(() => {
    if (updateInterval) {
      clearInterval(updateInterval);
    }
  });

  // Reactive statements
  $: systemUsagePercent = formatPercent($memoryMetrics.system.used, $memoryMetrics.system.total);
  $: gpuUsagePercent = formatPercent($memoryMetrics.gpu.used, $memoryMetrics.gpu.total);
  $: totalAIMemory = $memoryMetrics.ai.embeddingCache + $memoryMetrics.ai.vectorStore;
</script>

<div class="memory-dashboard">
  <div class="dashboard-header">
    <h1 class="page-title">AI Memory Dashboard</h1>
    <div class="dashboard-controls">
      <button class="control-btn" class:active={autoRefresh} on:click={() => autoRefresh = !autoRefresh}>
        🔄 Auto Refresh
      </button>
      <button class="control-btn optimize" on:click={optimizeMemory}>
        ⚡ Optimize Memory
      </button>
    </div>
  </div>

  <!-- Overview Cards -->
  <div class="overview-grid">
    <div class="metric-card system">
      <div class="card-header">
        <h3>System Memory</h3>
        <span class="metric-icon">🖥️</span>
      </div>
      <div class="metric-value">
        {formatBytes($memoryMetrics.system.used)} / {formatBytes($memoryMetrics.system.total)}
      </div>
      <div class="metric-bar">
        <div class="metric-fill" style="width: {systemUsagePercent}%"></div>
      </div>
      <div class="metric-percent">{systemUsagePercent}% Used</div>
    </div>

    <div class="metric-card gpu">
      <div class="card-header">
        <h3>GPU Memory</h3>
        <span class="metric-icon">🎮</span>
      </div>
      <div class="metric-value">
        {formatBytes($memoryMetrics.gpu.used)} / {formatBytes($memoryMetrics.gpu.total)}
      </div>
      <div class="metric-bar">
        <div class="metric-fill" style="width: {gpuUsagePercent}%"></div>
      </div>
      <div class="metric-percent">{gpuUsagePercent}% Used</div>
    </div>

    <div class="metric-card ai">
      <div class="card-header">
        <h3>AI Context</h3>
        <span class="metric-icon">🧠</span>
      </div>
      <div class="metric-value">
        {$memoryMetrics.ai.contextWindows} Active
      </div>
      <div class="metric-detail">
        {$memoryMetrics.ai.activeModels} Models Running
      </div>
      <div class="metric-detail">
        {formatBytes(totalAIMemory)} Total
      </div>
    </div>

    <div class="metric-card performance">
      <div class="card-header">
        <h3>Performance</h3>
        <span class="metric-icon">📊</span>
      </div>
      <div class="performance-grid">
        <div class="perf-item">
          <span class="perf-label">CPU</span>
          <span class="perf-value">{$memoryMetrics.performance.cpuUsage}%</span>
        </div>
        <div class="perf-item">
          <span class="perf-label">GPU</span>
          <span class="perf-value">{$memoryMetrics.performance.gpuUsage}%</span>
        </div>
      </div>
    </div>
  </div>

  <!-- Memory Usage Chart -->
  <div class="chart-section">
    <div class="section-header">
      <h2>Memory Usage Over Time</h2>
      <div class="chart-controls">
        <select bind:value={selectedTimeRange} class="time-select">
          <option value="5m">Last 5 minutes</option>
          <option value="1h">Last hour</option>
          <option value="24h">Last 24 hours</option>
        </select>
      </div>
    </div>
    
    <div class="memory-chart">
      <div class="chart-container">
        <svg class="chart-svg" viewBox="0 0 800 200">
          <!-- Chart background -->
          <rect width="800" height="200" fill="var(--yorha-bg-primary, #0a0a0a)" />
          
          <!-- Grid lines -->
          {#each Array(5) as _, i}
            <line 
              x1="0" 
              y1={40 * i} 
              x2="800" 
              y2={40 * i} 
              stroke="var(--yorha-border, #333)" 
              stroke-width="0.5"
            />
          {/each}
          
          <!-- Memory usage lines -->
          <polyline
            points={memoryHistory.map((point, i) => `${i * 16},${200 - point.system * 2}`).join(' ')}
            fill="none"
            stroke="var(--yorha-accent-warm, #ffbf00)"
            stroke-width="2"
          />
          <polyline
            points={memoryHistory.map((point, i) => `${i * 16},${200 - point.gpu * 2}`).join(' ')}
            fill="none"
            stroke="var(--yorha-success, #00ff41)"
            stroke-width="2"
          />
          <polyline
            points={memoryHistory.map((point, i) => `${i * 16},${200 - point.ai * 2}`).join(' ')}
            fill="none"
            stroke="var(--yorha-accent-blue, #00bcd4)"
            stroke-width="2"
          />
        </svg>
        
        <div class="chart-legend">
          <div class="legend-item">
            <div class="legend-color system"></div>
            <span>System Memory</span>
          </div>
          <div class="legend-item">
            <div class="legend-color gpu"></div>
            <span>GPU Memory</span>
          </div>
          <div class="legend-item">
            <div class="legend-color ai"></div>
            <span>AI Context</span>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Context Windows -->
  <div class="contexts-section">
    <div class="section-header">
      <h2>Active Context Windows</h2>
      <span class="context-count">{contextWindows.length} Active</span>
    </div>
    
    <div class="contexts-grid">
      {#each contextWindows as context}
        <div class="context-card">
          <div class="context-header">
            <div class="context-info">
              <span class="context-icon">{getTypeIcon(context.type)}</span>
              <div class="context-details">
                <h4 class="context-name">{context.name}</h4>
                <span class="context-type">{context.type}</span>
              </div>
            </div>
            <div class="context-status" style="color: {getStatusColor(context.status)}">
              {context.status}
            </div>
          </div>
          
          <div class="context-metrics">
            <div class="context-metric">
              <span class="metric-label">Tokens</span>
              <span class="metric-value">{context.tokens} / {context.maxTokens}</span>
              <div class="token-bar">
                <div 
                  class="token-fill" 
                  style="width: {(context.tokens / context.maxTokens) * 100}%"
                ></div>
              </div>
            </div>
            
            <div class="context-metric">
              <span class="metric-label">Memory</span>
              <span class="metric-value">{formatBytes(context.memory)}</span>
            </div>
            
            <div class="context-metric">
              <span class="metric-label">Last Active</span>
              <span class="metric-value">
                {Math.floor((Date.now() - context.lastActive.getTime()) / 60000)}m ago
              </span>
            </div>
          </div>
          
          <div class="context-actions">
            <button class="context-btn" on:click={() => clearContext(context.id)}>
              🗑️ Clear
            </button>
          </div>
        </div>
      {/each}
    </div>
  </div>

  <!-- AI Models -->
  <div class="models-section">
    <div class="section-header">
      <h2>AI Models</h2>
      <div class="model-controls">
        <select bind:value={selectedModel} class="model-select">
          <option value="all">All Models</option>
          {#each aiModels as model}
            <option value={model.id}>{model.name}</option>
          {/each}
        </select>
      </div>
    </div>
    
    <div class="models-grid">
      {#each aiModels as model}
        <div class="model-card">
          <div class="model-header">
            <div class="model-info">
              <span class="model-icon">{getTypeIcon(model.type)}</span>
              <div class="model-details">
                <h4 class="model-name">{model.name}</h4>
                <span class="model-type">{model.type}</span>
              </div>
            </div>
            <div class="model-status" style="color: {getStatusColor(model.status)}">
              {model.status}
            </div>
          </div>
          
          <div class="model-metrics">
            <div class="model-metric">
              <span class="metric-label">Memory Usage</span>
              <span class="metric-value">{formatBytes(model.memory)}</span>
            </div>
            
            <div class="model-metric">
              <span class="metric-label">Total Requests</span>
              <span class="metric-value">{model.requests.toLocaleString()}</span>
            </div>
            
            <div class="model-metric">
              <span class="metric-label">Avg Response</span>
              <span class="metric-value">{model.avgResponseTime}ms</span>
            </div>
            
            <div class="model-metric">
              <span class="metric-label">GPU Enabled</span>
              <span class="metric-value">{model.gpu ? '✅' : '❌'}</span>
            </div>
          </div>
          
          <div class="model-actions">
            <button class="model-btn" on:click={() => restartModel(model.id)}>
              🔄 Restart
            </button>
          </div>
        </div>
      {/each}
    </div>
  </div>
</div>

<style>
  .memory-dashboard {
    max-width: 1400px;
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
  }

  .control-btn {
    background: var(--yorha-bg-secondary, #1a1a1a);
    border: 1px solid var(--yorha-border, #333);
    color: var(--yorha-text-primary, #ffffff);
    padding: 0.75rem 1rem;
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

  .control-btn.optimize:hover {
    border-color: var(--yorha-success, #00ff41);
    background: rgba(0, 255, 65, 0.1);
  }

  .overview-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 2rem;
    margin-bottom: 3rem;
  }

  .metric-card {
    background: var(--yorha-bg-secondary, #1a1a1a);
    border: 1px solid var(--yorha-border, #333);
    border-radius: 8px;
    padding: 1.5rem;
    transition: all 0.2s ease;
  }

  .metric-card:hover {
    border-color: var(--yorha-accent-warm, #ffbf00);
    box-shadow: 0 4px 8px rgba(255, 191, 0, 0.1);
  }

  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }

  .card-header h3 {
    margin: 0;
    font-size: 1.1rem;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .metric-icon {
    font-size: 1.5rem;
  }

  .metric-value {
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--yorha-accent-warm, #ffbf00);
    margin-bottom: 0.5rem;
  }

  .metric-detail {
    font-size: 0.9rem;
    color: var(--yorha-text-secondary, #a0a0a0);
    margin: 0.25rem 0;
  }

  .metric-bar {
    width: 100%;
    height: 6px;
    background: var(--yorha-bg-primary, #0a0a0a);
    border-radius: 3px;
    overflow: hidden;
    margin: 0.5rem 0;
  }

  .metric-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--yorha-accent-warm, #ffbf00), var(--yorha-success, #00ff41));
    transition: width 0.3s ease;
  }

  .metric-percent {
    font-size: 0.8rem;
    color: var(--yorha-text-secondary, #a0a0a0);
  }

  .performance-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
    margin-top: 0.5rem;
  }

  .perf-item {
    display: flex;
    justify-content: space-between;
  }

  .perf-label {
    font-size: 0.8rem;
    color: var(--yorha-text-secondary, #a0a0a0);
  }

  .perf-value {
    font-weight: 600;
  }

  .chart-section,
  .contexts-section,
  .models-section {
    background: var(--yorha-bg-secondary, #1a1a1a);
    border: 1px solid var(--yorha-border, #333);
    border-radius: 8px;
    padding: 2rem;
    margin-bottom: 2rem;
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
    font-size: 1.5rem;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .context-count {
    color: var(--yorha-accent-warm, #ffbf00);
    font-weight: 600;
  }

  .chart-container {
    position: relative;
  }

  .chart-svg {
    width: 100%;
    height: 200px;
    border: 1px solid var(--yorha-border, #333);
    border-radius: 4px;
  }

  .chart-legend {
    display: flex;
    gap: 2rem;
    margin-top: 1rem;
    justify-content: center;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.9rem;
  }

  .legend-color {
    width: 12px;
    height: 12px;
    border-radius: 2px;
  }

  .legend-color.system {
    background: var(--yorha-accent-warm, #ffbf00);
  }

  .legend-color.gpu {
    background: var(--yorha-success, #00ff41);
  }

  .legend-color.ai {
    background: var(--yorha-accent-blue, #00bcd4);
  }

  .contexts-grid,
  .models-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
    gap: 1.5rem;
  }

  .context-card,
  .model-card {
    background: var(--yorha-bg-primary, #0a0a0a);
    border: 1px solid var(--yorha-border, #333);
    border-radius: 6px;
    padding: 1.5rem;
    transition: all 0.2s ease;
  }

  .context-card:hover,
  .model-card:hover {
    border-color: var(--yorha-accent-warm, #ffbf00);
  }

  .context-header,
  .model-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }

  .context-info,
  .model-info {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .context-icon,
  .model-icon {
    font-size: 1.5rem;
  }

  .context-name,
  .model-name {
    margin: 0 0 0.25rem 0;
    font-size: 1rem;
    font-weight: 600;
  }

  .context-type,
  .model-type {
    font-size: 0.8rem;
    color: var(--yorha-text-secondary, #a0a0a0);
    text-transform: uppercase;
  }

  .context-metrics,
  .model-metrics {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-bottom: 1rem;
  }

  .context-metric,
  .model-metric {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .metric-label {
    font-size: 0.8rem;
    color: var(--yorha-text-secondary, #a0a0a0);
    text-transform: uppercase;
  }

  .token-bar {
    width: 100%;
    height: 4px;
    background: var(--yorha-bg-secondary, #1a1a1a);
    border-radius: 2px;
    overflow: hidden;
    margin-top: 0.25rem;
  }

  .token-fill {
    height: 100%;
    background: var(--yorha-accent-warm, #ffbf00);
    transition: width 0.3s ease;
  }

  .context-actions,
  .model-actions {
    text-align: right;
  }

  .context-btn,
  .model-btn {
    background: transparent;
    border: 1px solid var(--yorha-border, #333);
    color: var(--yorha-text-primary, #ffffff);
    padding: 0.5rem 1rem;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.2s ease;
    font-family: inherit;
    font-size: 0.8rem;
  }

  .context-btn:hover,
  .model-btn:hover {
    border-color: var(--yorha-accent-warm, #ffbf00);
    background: rgba(255, 191, 0, 0.1);
  }

  .time-select,
  .model-select {
    background: var(--yorha-bg-primary, #0a0a0a);
    border: 1px solid var(--yorha-border, #333);
    color: var(--yorha-text-primary, #ffffff);
    padding: 0.5rem;
    border-radius: 4px;
    font-family: inherit;
  }

  .time-select:focus,
  .model-select:focus {
    outline: none;
    border-color: var(--yorha-accent-warm, #ffbf00);
  }

  @media (max-width: 768px) {
    .memory-dashboard {
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

    .overview-grid {
      grid-template-columns: 1fr;
    }

    .contexts-grid,
    .models-grid {
      grid-template-columns: 1fr;
    }

    .chart-legend {
      flex-direction: column;
      align-items: center;
      gap: 0.5rem;
    }
  }
</style>