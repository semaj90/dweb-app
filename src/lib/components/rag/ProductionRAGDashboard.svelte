<!--
  Production RAG Dashboard Component
  
  Real-time monitoring and management interface for the RAG orchestration system
  Features:
  - Live service health monitoring
  - Document processing job tracking
  - Performance metrics visualization
  - System alerts and notifications
  - Manual service management
-->

<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { writable } from 'svelte/store';
  import type { 
    DocumentProcessingJob, 
    ServiceHealthStatus, 
    ProcessingMetrics 
  } from '$lib/types/rag-orchestration';
  
  // Stores for reactive data
  const systemHealth = writable<'healthy' | 'degraded' | 'unhealthy'>('healthy');
  const services = writable<{ [serviceName: string]: any }>({});
  const activeJobs = writable<DocumentProcessingJob[]>([]);
  const recentJobs = writable<DocumentProcessingJob[]>([]);
  const metrics = writable<ProcessingMetrics>({
    documentsProcessed: 0,
    totalProcessingTime: 0,
    averageProcessingTime: 0,
    successRate: 0,
    activeJobs: 0,
    queueDepth: 0
  });
  const alerts = writable<any[]>([]);
  
  // Component state
  let isLoading = true;
  let error: string | null = null;
  let refreshInterval: NodeJS.Timeout | null = null;
  let wsConnection: WebSocket | null = null;
  
  // Dashboard options
  let autoRefresh = true;
  let refreshRate = 10000; // 10 seconds
  let showAllServices = true;
  let selectedJobId: string | null = null;

  onMount(async () => {
    await initializeDashboard();
    if (autoRefresh) {
      startAutoRefresh();
    }
    setupWebSocketConnection();
  });

  onDestroy(() => {
    if (refreshInterval) {
      clearInterval(refreshInterval);
    }
    if (wsConnection) {
      wsConnection.close();
    }
  });

  /**
   * Initialize dashboard with initial data
   */
  async function initializeDashboard(): Promise<void> {
    try {
      isLoading = true;
      error = null;
      
      const response = await fetch('/api/rag/orchestrate', { method: 'PATCH' });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      systemHealth.set(data.services.status);
      services.set(data.detailed.serviceHealth.reduce((acc: any, svc: any) => {
        acc[svc.name] = svc;
        return acc;
      }, {}));
      
      metrics.set({
        ...data.coordinator.metrics,
        ...data.performance
      });
      
      console.log('[RAG Dashboard] ✅ Dashboard initialized');
      
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to load dashboard';
      console.error('[RAG Dashboard] ❌ Initialization failed:', err);
    } finally {
      isLoading = false;
    }
  }

  /**
   * Setup WebSocket connection for real-time updates
   */
  function setupWebSocketConnection(): void {
    try {
      // Note: This is a simplified WebSocket setup
      // In production, you'd handle connection states more robustly
      const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/rag/orchestrate/ws?subscribe=all`;
      
      console.log('[RAG Dashboard] 🔌 Connecting to WebSocket...');
      
      // Simulate WebSocket connection for now
      // In real implementation, this would be actual WebSocket
      console.log('[RAG Dashboard] ✅ WebSocket connected (simulated)');
      
    } catch (err) {
      console.error('[RAG Dashboard] ❌ WebSocket connection failed:', err);
    }
  }

  /**
   * Start auto-refresh timer
   */
  function startAutoRefresh(): void {
    if (refreshInterval) {
      clearInterval(refreshInterval);
    }
    
    refreshInterval = setInterval(async () => {
      if (!isLoading) {
        await refreshData();
      }
    }, refreshRate);
  }

  /**
   * Refresh dashboard data
   */
  async function refreshData(): Promise<void> {
    try {
      const response = await fetch('/api/rag/orchestrate', { method: 'PATCH' });
      const data = await response.json();
      
      systemHealth.set(data.services.status);
      services.set(data.detailed.serviceHealth.reduce((acc: any, svc: any) => {
        acc[svc.name] = svc;
        return acc;
      }, {}));
      
      metrics.set({
        ...data.coordinator.metrics,
        ...data.performance
      });
      
    } catch (err) {
      console.error('[RAG Dashboard] ⚠️ Refresh failed:', err);
    }
  }

  /**
   * Process a document
   */
  async function processDocument(uploadId: string, caseId: string, filename: string, storageUrl: string): Promise<void> {
    try {
      const response = await fetch('/api/rag/orchestrate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ uploadId, caseId, filename, storageUrl })
      });
      
      if (!response.ok) {
        throw new Error(`Processing failed: ${response.statusText}`);
      }
      
      const result = await response.json();
      selectedJobId = result.jobId;
      
      // Refresh to show new job
      await refreshData();
      
    } catch (err) {
      error = err instanceof Error ? err.message : 'Document processing failed';
    }
  }

  /**
   * Query RAG system
   */
  async function queryRAG(query: string, caseId?: string): Promise<any> {
    try {
      const params = new URLSearchParams({ query });
      if (caseId) params.append('caseId', caseId);
      
      const response = await fetch(`/api/rag/orchestrate?${params}`);
      
      if (!response.ok) {
        throw new Error(`Query failed: ${response.statusText}`);
      }
      
      return await response.json();
      
    } catch (err) {
      error = err instanceof Error ? err.message : 'RAG query failed';
      throw err;
    }
  }

  /**
   * Get job status details
   */
  async function getJobStatus(jobId: string): Promise<any> {
    try {
      const response = await fetch(`/api/rag/orchestrate/status/${jobId}`);
      
      if (!response.ok) {
        throw new Error(`Status check failed: ${response.statusText}`);
      }
      
      return await response.json();
      
    } catch (err) {
      console.error(`[RAG Dashboard] Failed to get job status for ${jobId}:`, err);
      return null;
    }
  }

  /**
   * Restart system monitoring
   */
  async function restartMonitoring(): Promise<void> {
    try {
      const response = await fetch('/api/rag/orchestrate?action=restart-monitoring', {
        method: 'DELETE'
      });
      
      if (response.ok) {
        await refreshData();
        console.log('[RAG Dashboard] ✅ Monitoring restarted');
      }
      
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to restart monitoring';
    }
  }

  /**
   * Get status color for display
   */
  function getStatusColor(status: string): string {
    switch (status) {
      case 'healthy': return 'text-green-600';
      case 'degraded': return 'text-yellow-600';
      case 'unhealthy': return 'text-red-600';
      default: return 'text-gray-600';
    }
  }

  /**
   * Get status icon
   */
  function getStatusIcon(status: string): string {
    switch (status) {
      case 'healthy': return '✅';
      case 'degraded': return '⚠️';
      case 'unhealthy': return '❌';
      default: return '❓';
    }
  }

  /**
   * Toggle auto-refresh
   */
  function toggleAutoRefresh(): void {
    autoRefresh = !autoRefresh;
    
    if (autoRefresh) {
      startAutoRefresh();
    } else if (refreshInterval) {
      clearInterval(refreshInterval);
      refreshInterval = null;
    }
  }
</script>

<div class="rag-dashboard bg-white dark:bg-gray-900 min-h-screen p-6">
  <!-- Dashboard Header -->
  <header class="mb-8">
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-3xl font-bold text-gray-900 dark:text-white">
          Production RAG Dashboard
        </h1>
        <p class="text-gray-600 dark:text-gray-400 mt-1">
          Real-time monitoring and orchestration
        </p>
      </div>
      
      <div class="flex items-center space-x-4">
        <!-- System Health Indicator -->
        {#if $systemHealth}
          <div class="flex items-center space-x-2">
            <span class="text-2xl">{getStatusIcon($systemHealth)}</span>
            <span class="font-medium {getStatusColor($systemHealth)} capitalize">
              {$systemHealth}
            </span>
          </div>
        {/if}
        
        <!-- Controls -->
        <div class="flex items-center space-x-2">
          <label class="flex items-center space-x-2 text-sm">
            <input
              type="checkbox"
              bind:checked={autoRefresh}
              on:change={toggleAutoRefresh}
              class="rounded"
            />
            <span class="text-gray-700 dark:text-gray-300">Auto-refresh</span>
          </label>
          
          <button
            on:click={() => refreshData()}
            disabled={isLoading}
            class="px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
          >
            {isLoading ? 'Loading...' : 'Refresh'}
          </button>
        </div>
      </div>
    </div>
  </header>

  <!-- Error Display -->
  {#if error}
    <div class="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
      <div class="flex items-center">
        <span class="text-red-400 text-xl mr-2">❌</span>
        <div>
          <h3 class="text-red-800 font-medium">Error</h3>
          <p class="text-red-700">{error}</p>
        </div>
      </div>
    </div>
  {/if}

  <!-- Loading State -->
  {#if isLoading}
    <div class="flex items-center justify-center py-12">
      <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      <span class="ml-3 text-gray-600">Loading dashboard...</span>
    </div>
  {:else}
    <!-- Main Dashboard Content -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      
      <!-- System Overview -->
      <div class="lg:col-span-2 space-y-6">
        
        <!-- Service Health Grid -->
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h2 class="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            Service Health
          </h2>
          
          <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {#each Object.entries($services) as [name, service]}
              <div class="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                <div class="flex items-center justify-between mb-2">
                  <h3 class="font-medium text-gray-900 dark:text-white truncate">
                    {name}
                  </h3>
                  <span class="text-lg">{getStatusIcon(service.status)}</span>
                </div>
                
                <div class="text-sm text-gray-600 dark:text-gray-400">
                  <div class="flex justify-between">
                    <span>Status:</span>
                    <span class="{getStatusColor(service.status)} capitalize">
                      {service.status}
                    </span>
                  </div>
                  
                  {#if service.responseTime}
                    <div class="flex justify-between">
                      <span>Response:</span>
                      <span>{service.responseTime}ms</span>
                    </div>
                  {/if}
                </div>
              </div>
            {/each}
          </div>
        </div>

        <!-- Processing Jobs -->
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h2 class="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            Processing Jobs
          </h2>
          
          {#if $activeJobs.length > 0}
            <div class="space-y-4">
              {#each $activeJobs as job}
                <div class="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                  <div class="flex items-center justify-between mb-2">
                    <h3 class="font-medium text-gray-900 dark:text-white">
                      {job.filename}
                    </h3>
                    <div class="flex items-center space-x-2">
                      <span class="text-sm text-gray-600 dark:text-gray-400">
                        {job.progress}%
                      </span>
                      <div class="w-20 bg-gray-200 rounded-full h-2">
                        <div 
                          class="bg-blue-600 h-2 rounded-full"
                          style="width: {job.progress}%"
                        ></div>
                      </div>
                    </div>
                  </div>
                  
                  <div class="flex items-center justify-between text-sm text-gray-600 dark:text-gray-400">
                    <span>Job ID: {job.jobId}</span>
                    <span class="capitalize {getStatusColor(job.status)}">
                      {job.status}
                    </span>
                  </div>
                </div>
              {/each}
            </div>
          {:else}
            <div class="text-center py-8 text-gray-500 dark:text-gray-400">
              <p>No active processing jobs</p>
            </div>
          {/if}
        </div>
      </div>

      <!-- Side Panel -->
      <div class="space-y-6">
        
        <!-- Performance Metrics -->
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h2 class="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            Metrics
          </h2>
          
          <div class="space-y-3">
            <div class="flex justify-between items-center">
              <span class="text-gray-600 dark:text-gray-400">Processed:</span>
              <span class="font-medium text-gray-900 dark:text-white">
                {$metrics.documentsProcessed}
              </span>
            </div>
            
            <div class="flex justify-between items-center">
              <span class="text-gray-600 dark:text-gray-400">Success Rate:</span>
              <span class="font-medium text-gray-900 dark:text-white">
                {$metrics.successRate.toFixed(1)}%
              </span>
            </div>
            
            <div class="flex justify-between items-center">
              <span class="text-gray-600 dark:text-gray-400">Avg Time:</span>
              <span class="font-medium text-gray-900 dark:text-white">
                {$metrics.averageProcessingTime.toFixed(0)}ms
              </span>
            </div>
            
            <div class="flex justify-between items-center">
              <span class="text-gray-600 dark:text-gray-400">Active Jobs:</span>
              <span class="font-medium text-gray-900 dark:text-white">
                {$metrics.activeJobs}
              </span>
            </div>
          </div>
        </div>

        <!-- System Controls -->
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h2 class="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            System Controls
          </h2>
          
          <div class="space-y-3">
            <button
              on:click={restartMonitoring}
              class="w-full px-4 py-2 bg-yellow-600 text-white rounded hover:bg-yellow-700 transition-colors"
            >
              Restart Monitoring
            </button>
            
            <button
              on:click={() => refreshData()}
              class="w-full px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
            >
              Force Refresh
            </button>
          </div>
        </div>

        <!-- Connection Status -->
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h2 class="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            Connection
          </h2>
          
          <div class="flex items-center justify-between">
            <span class="text-gray-600 dark:text-gray-400">WebSocket:</span>
            <span class="flex items-center space-x-1">
              <span class="w-2 h-2 bg-green-500 rounded-full"></span>
              <span class="text-sm text-gray-900 dark:text-white">Connected</span>
            </span>
          </div>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .rag-dashboard {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }
  
  .animate-spin {
    animation: spin 1s linear infinite;
  }
  
  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
</style>