<!--
  Unified Cache Demo Page
  Demonstrates GPU cache + Redis + WASM graph engine integration
-->

<script lang="ts">
  import { onMount } from 'svelte';
  import { unifiedServiceRegistry } from '$lib/services/unifiedServiceRegistry';
  import { wasmGraphEngine } from '$lib/wasm/graphEngine';
  import { idleDetectionService } from '$lib/machines/idleDetectionMachine';
  import ModernButton from '$lib/components/ui/button/Button.svelte';

  let serviceStatus = $state(null);
  let wasmStats = $state(null);
  let idleState = $state(null);
  let cacheDemo = $state({
    loading: false,
    results: null,
    queryTime: 0
  });

  onMount(async () => {
    // Get initial system status
    serviceStatus = await unifiedServiceRegistry.getSystemStatus();
    wasmStats = wasmGraphEngine.getStats();
    idleState = idleDetectionService.getContext();

    // Listen for idle state changes
    window.addEventListener('idle-state-change', (event) => {
      idleState = event.detail.context;
    });

    // Refresh stats periodically
    const interval = setInterval(async () => {
      wasmStats = wasmGraphEngine.getStats();
      serviceStatus = await unifiedServiceRegistry.getSystemStatus();
    }, 5000);

    return () => clearInterval(interval);
  });

  async function runCacheDemo() {
    cacheDemo.loading = true;
    const startTime = Date.now();

    try {
      // Test graph query with WASM engine
      const result = await wasmGraphEngine.executeQuery(
        'MATCH (case:Case)-[:HAS_EVIDENCE]->(evidence:Evidence) RETURN case, evidence LIMIT 5'
      );
      
      cacheDemo.results = result;
      cacheDemo.queryTime = Date.now() - startTime;
    } catch (error) {
      console.error('Cache demo failed:', error);
      cacheDemo.results = { error: error.message };
    } finally {
      cacheDemo.loading = false;
    }
  }

  async function clearAllCaches() {
    await unifiedServiceRegistry.clearCaches();
    serviceStatus = await unifiedServiceRegistry.getSystemStatus(false);
    wasmStats = wasmGraphEngine.getStats();
  }

  function triggerIdle() {
    idleDetectionService.triggerIdle();
  }

  function triggerActivity() {
    idleDetectionService.triggerActivity();
  }
</script>

<svelte:head>
  <title>GPU Cache Demo - YoRHa Legal AI</title>
</svelte:head>

<div class="space-y-6">
  <header>
    <h1 class="text-3xl font-bold text-nier-accent-warm mb-2">GPU Cache Demo</h1>
    <p class="text-nier-text-secondary">
      Unified caching system: Redis + WASM Graph Engine + Background Processing
    </p>
  </header>

  <!-- System Status Grid -->
  <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
    <!-- Service Registry Status -->
    <div class="bg-nier-bg-secondary border border-nier-border-primary rounded-lg p-4">
      <h3 class="font-bold text-nier-accent-warm mb-3">Service Registry</h3>
      {#if serviceStatus}
        <div class="space-y-2">
          <div class="flex justify-between">
            <span>Health Score:</span>
            <span class="font-mono {serviceStatus.healthScore >= 80 ? 'text-green-400' : serviceStatus.healthScore >= 60 ? 'text-yellow-400' : 'text-red-400'}">
              {serviceStatus.healthScore}%
            </span>
          </div>
          <div class="text-xs space-y-1">
            {#each Object.entries(serviceStatus.services) as [name, status]}
              <div class="flex justify-between">
                <span>{name}:</span>
                <span class="font-mono {status.online ? 'text-green-400' : 'text-red-400'}">
                  {status.online ? '✅' : '❌'} 
                  {status.responseTime ? `${status.responseTime}ms` : ''}
                </span>
              </div>
            {/each}
          </div>
          <div class="text-xs text-nier-text-muted mt-2">
            {serviceStatus.cached ? '📋 Cached' : '🔄 Fresh'} • 
            Updated: {serviceStatus.lastUpdate.toLocaleTimeString()}
          </div>
        </div>
      {:else}
        <div class="text-nier-text-muted">Loading service status...</div>
      {/if}
    </div>

    <!-- WASM Graph Engine Stats -->
    <div class="bg-nier-bg-secondary border border-nier-border-primary rounded-lg p-4">
      <h3 class="font-bold text-nier-accent-warm mb-3">WASM Graph Engine</h3>
      {#if wasmStats}
        <div class="space-y-2 text-sm">
          <div class="flex justify-between">
            <span>Queries Cached:</span>
            <span class="font-mono text-green-400">{wasmStats.queriesCached}</span>
          </div>
          <div class="flex justify-between">
            <span>Memory Usage:</span>
            <span class="font-mono">{wasmStats.memoryUsage}</span>
          </div>
          <div class="flex justify-between">
            <span>Cache Hit Rate:</span>
            <span class="font-mono text-blue-400">{wasmStats.cacheHitRate.toFixed(1)}%</span>
          </div>
          <div class="flex justify-between">
            <span>Uptime:</span>
            <span class="font-mono">{Math.round(wasmStats.uptime / 1000)}s</span>
          </div>
          {#if wasmStats.lastHydration}
            <div class="text-xs text-nier-text-muted mt-2">
              Last hydration: {wasmStats.lastHydration.toLocaleTimeString()}
            </div>
          {/if}
        </div>
      {:else}
        <div class="text-nier-text-muted">WASM engine not initialized</div>
      {/if}
    </div>

    <!-- Idle Detection Status -->
    <div class="bg-nier-bg-secondary border border-nier-border-primary rounded-lg p-4">
      <h3 class="font-bold text-nier-accent-warm mb-3">Idle Detection</h3>
      {#if idleState}
        <div class="space-y-2 text-sm">
          <div class="flex justify-between">
            <span>Current State:</span>
            <span class="font-mono {idleDetectionService.isIdle() ? 'text-yellow-400' : 'text-green-400'}">
              {idleDetectionService.isIdle() ? '💤 Idle' : '🏃 Active'}
            </span>
          </div>
          <div class="flex justify-between">
            <span>Idle Count:</span>
            <span class="font-mono">{idleState.idleCount}</span>
          </div>
          <div class="flex justify-between">
            <span>Background Tasks:</span>
            <span class="font-mono">{idleState.backgroundTasks?.length || 0}</span>
          </div>
          <div class="flex justify-between">
            <span>Hot Queries:</span>
            <span class="font-mono text-blue-400">{idleState.hotQueriesCached}</span>
          </div>
          <div class="text-xs text-nier-text-muted mt-2">
            Last activity: {idleState.lastActivity.toLocaleTimeString()}
          </div>
        </div>
      {:else}
        <div class="text-nier-text-muted">Idle detection not active</div>
      {/if}
    </div>
  </div>

  <!-- Cache Demo Section -->
  <div class="bg-nier-bg-secondary border border-nier-border-primary rounded-lg p-6">
    <h3 class="font-bold text-nier-accent-warm mb-4">Graph Query Cache Demo</h3>
    
    <div class="flex gap-4 mb-4">
      <ModernButton
        on:on:on:click={runCacheDemo}
        disabled={cacheDemo.loading}
        class="bg-blue-600 hover:bg-blue-700"
      >
        {cacheDemo.loading ? '⚡ Running...' : '🔍 Run Graph Query'}
      </ModernButton>
      
      <ModernButton
        on:on:on:click={triggerIdle}
        variant="outline"
        class="border-yellow-500 text-yellow-400 hover:bg-yellow-500/10"
      >
        💤 Trigger Idle
      </ModernButton>
      
      <ModernButton
        on:on:on:click={triggerActivity}
        variant="outline"
        class="border-green-500 text-green-400 hover:bg-green-500/10"
      >
        🏃 Trigger Activity
      </ModernButton>
      
      <ModernButton
        on:on:on:click={clearAllCaches}
        variant="outline"
        class="border-red-500 text-red-400 hover:bg-red-500/10"
      >
        🗑️ Clear Caches
      </ModernButton>
    </div>

    {#if cacheDemo.results}
      <div class="bg-nier-bg-primary border border-nier-border-muted rounded p-4">
        <div class="flex justify-between items-center mb-3">
          <h4 class="font-bold">Query Results</h4>
          <span class="text-xs font-mono text-nier-text-muted">
            Executed in {cacheDemo.queryTime}ms
          </span>
        </div>
        
        {#if cacheDemo.results.error}
          <div class="text-red-400 font-mono text-sm">
            Error: {cacheDemo.results.error}
          </div>
        {:else}
          <div class="space-y-3">
            <div class="flex justify-between text-sm">
              <span>Source:</span>
              <span class="font-mono px-2 py-1 rounded text-xs
                {cacheDemo.results.metadata.source === 'wasm' ? 'bg-blue-500/20 text-blue-400' : 
                  cacheDemo.results.metadata.source === 'cache' ? 'bg-green-500/20 text-green-400' : 
                  'bg-yellow-500/20 text-yellow-400'}">
                {cacheDemo.results.metadata.source.toUpperCase()}
              </span>
            </div>
            
            <div class="grid grid-cols-2 gap-4 text-sm">
              <div>
                <div class="text-nier-text-secondary mb-1">Nodes: {cacheDemo.results.nodes.length}</div>
                <div class="text-xs space-y-1">
                  {#each cacheDemo.results.nodes.slice(0, 3) as node}
                    <div class="font-mono text-nier-text-muted">
                      {node.type}: {node.label}
                    </div>
                  {/each}
                  {#if cacheDemo.results.nodes.length > 3}
                    <div class="text-xs text-nier-text-muted">
                      ... and {cacheDemo.results.nodes.length - 3} more
                    </div>
                  {/if}
                </div>
              </div>
              
              <div>
                <div class="text-nier-text-secondary mb-1">Edges: {cacheDemo.results.edges.length}</div>
                <div class="text-xs space-y-1">
                  {#each cacheDemo.results.edges.slice(0, 3) as edge}
                    <div class="font-mono text-nier-text-muted">
                      {edge.label}
                    </div>
                  {/each}
                  {#if cacheDemo.results.edges.length > 3}
                    <div class="text-xs text-nier-text-muted">
                      ... and {cacheDemo.results.edges.length - 3} more
                    </div>
                  {/if}
                </div>
              </div>
            </div>
          </div>
        {/if}
      </div>
    {/if}
  </div>

  <!-- Integration Architecture -->
  <div class="bg-nier-bg-secondary border border-nier-border-primary rounded-lg p-6">
    <h3 class="font-bold text-nier-accent-warm mb-4">Integration Architecture</h3>
    
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div>
        <h4 class="font-semibold text-nier-text-primary mb-2">Cache Hierarchy</h4>
        <div class="text-sm space-y-1 text-nier-text-secondary">
          <div>1. 🧠 WASM Memory Cache (instant)</div>
          <div>2. 🔴 Redis Cache (< 5ms)</div>
          <div>3. 🐘 PostgreSQL + pgvector (< 50ms)</div>
          <div>4. 🌐 Remote Neo4j (100ms+)</div>
        </div>
      </div>
      
      <div>
        <h4 class="font-semibold text-nier-text-primary mb-2">Background Processing</h4>
        <div class="text-sm space-y-1 text-nier-text-secondary">
          <div>• XState idle detection (5min threshold)</div>
          <div>• Hot query pre-loading</div>
          <div>• WASM engine hydration</div>
          <div>• Service health monitoring</div>
        </div>
      </div>
    </div>
  </div>
</div>

<style>
  /* Additional YoRHa styling for cache demo */
  :global(.font-mono) {
    font-family: 'JetBrains Mono', 'Roboto Mono', monospace;
  }
  
  /* Animations for loading states */
  @keyframes pulse-blue {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
  
  :global(.animate-pulse-blue) {
    animation: pulse-blue 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
  }
</style>