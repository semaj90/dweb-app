<!-- Multi-Protocol RAG Pipeline Demo Component -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { fade, fly } from 'svelte/transition';
  import { 
    Database, 
    Zap, 
    Activity, 
    CheckCircle, 
    AlertTriangle, 
    Clock,
    Globe,
    Send,
    Loader2,
    BarChart3
  } from 'lucide-svelte';
  import { ragPipeline } from '$lib/services/enhanced-rag-pipeline';
  import { multiProtocolRouter, protocolHealth, routerMetrics } from '$lib/services/multi-protocol-router';

  // Component state
  let query = $state('');
  let isProcessing = $state(false);
  let responses = $state<any[]>([]);
  let selectedProtocol = $state<'auto' | 'quic' | 'grpc' | 'rest'>('auto');
  
  // Protocol status
  let protocolStatuses = $state<any[]>([]);
  let metrics = $state<any>({});

  // Sample queries for testing
  const sampleQueries = [
    'What are the key elements of a valid contract?',
    'Explain the difference between civil and criminal law',
    'What is the statute of limitations for personal injury cases?',
    'How does evidence discovery work in litigation?',
    'What are the requirements for forming a corporation?'
  ];

  onMount(async () => {
    // Initialize the RAG pipeline
    await ragPipeline.initialize();
    
    // Subscribe to protocol health updates
    protocolHealth.subscribe(health => {
      protocolStatuses = health;
    });

    // Subscribe to router metrics
    routerMetrics.subscribe(m => {
      metrics = m;
    });

    // Perform initial health check
    await performHealthCheck();
  });

  async function performHealthCheck() {
    try {
      const health = await multiProtocolRouter.getMetrics();
      console.log('Protocol health check:', health);
    } catch (error) {
      console.error('Health check failed:', error);
    }
  }

  async function handleQuery() {
    if (!query.trim() || isProcessing) return;

    isProcessing = true;
    const startTime = Date.now();

    try {
      const result = await ragPipeline.query(query, {
        protocol: selectedProtocol === 'auto' ? undefined : selectedProtocol,
        maxTokens: 1000,
        temperature: 0.7,
        topK: 5
      });

      const processingTime = Date.now() - startTime;

      responses = [{
        id: `response_${Date.now()}`,
        query,
        result,
        processingTime,
        timestamp: Date.now(),
        protocol: result.metadata?.protocolUsed || 'unknown'
      }, ...responses.slice(0, 4)]; // Keep last 5 responses

      query = '';
    } catch (error) {
      console.error('Query failed:', error);
      responses = [{
        id: `error_${Date.now()}`,
        query,
        error: error.message,
        processingTime: Date.now() - startTime,
        timestamp: Date.now(),
        protocol: 'error'
      }, ...responses.slice(0, 4)];
    } finally {
      isProcessing = false;
    }
  }

  function selectSampleQuery(sampleQuery: string) {
    query = sampleQuery;
  }

  function getProtocolIcon(protocol: string) {
    switch (protocol) {
      case 'quic': return Zap;
      case 'grpc': return Database;
      case 'rest': return Globe;
      default: return Activity;
    }
  }

  function getStatusColor(status: string) {
    switch (status) {
      case 'healthy': return 'text-green-500';
      case 'degraded': return 'text-yellow-500';
      case 'error': return 'text-red-500';
      default: return 'text-gray-500';
    }
  }

  function getStatusIcon(status: string) {
    switch (status) {
      case 'healthy': return CheckCircle;
      case 'degraded': return AlertTriangle;
      case 'error': return AlertTriangle;
      default: return Activity;
    }
  }

  function formatLatency(latency: number) {
    return latency < 1000 ? `${latency}ms` : `${(latency / 1000).toFixed(1)}s`;
  }
</script>

<div class="multi-protocol-rag-demo bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 min-h-screen p-6">
  <!-- Header -->
  <div class="max-w-7xl mx-auto mb-8">
    <div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
      <div class="flex items-center justify-between">
        <div class="flex items-center space-x-4">
          <div class="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
            <Database class="w-7 h-7 text-white" />
          </div>
          <div>
            <h1 class="text-3xl font-bold text-slate-900 dark:text-slate-100">Multi-Protocol RAG Demo</h1>
            <p class="text-slate-600 dark:text-slate-400">QUIC → gRPC → REST Intelligent Fallback</p>
          </div>
        </div>
        
        <!-- Protocol Status Overview -->
        <div class="flex items-center space-x-4">
          {#each protocolStatuses as status}
            {@const StatusIcon = getStatusIcon(status.status)}
            <div class="flex items-center space-x-2 px-3 py-2 bg-slate-100 dark:bg-slate-700 rounded-lg">
              <StatusIcon class="w-4 h-4 {getStatusColor(status.status)}" />
              <span class="text-sm font-medium text-slate-700 dark:text-slate-300 uppercase">
                {status.protocol}
              </span>
              {#if status.latency > 0}
                <span class="text-xs text-slate-500 dark:text-slate-400">
                  {formatLatency(status.latency)}
                </span>
              {/if}
            </div>
          {/each}
        </div>
      </div>
    </div>
  </div>

  <div class="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-8">
    <!-- Query Interface -->
    <div class="lg:col-span-2 space-y-6">
      <!-- Query Input -->
      <div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
        <h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">Legal AI Query</h3>
        
        <!-- Protocol Selection -->
        <div class="mb-4">
          <label class="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
            Protocol Preference
          </label>
          <div class="flex space-x-2">
            {#each ['auto', 'quic', 'grpc', 'rest'] as protocol}
              <button
                class="px-3 py-2 text-sm font-medium rounded-lg transition-colors {
                  selectedProtocol === protocol
                    ? 'bg-blue-600 text-white'
                    : 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-600'
                }"
                on:click={() => selectedProtocol = protocol}
              >
                {protocol.toUpperCase()}
              </button>
            {/each}
          </div>
        </div>

        <!-- Query Input -->
        <div class="mb-4">
          <div class="flex space-x-2">
            <textarea
              bind:value={query}
              placeholder="Enter your legal question..."
              disabled={isProcessing}
              class="flex-1 resize-none rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 px-3 py-2 text-slate-900 dark:text-slate-100 placeholder-slate-500 dark:placeholder-slate-400 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:opacity-50"
              rows="3"
              on:keydown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleQuery();
                }
              }}
            ></textarea>
            
            <button
              on:click={handleQuery}
              disabled={!query.trim() || isProcessing}
              class="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-400 text-white rounded-lg transition-colors flex items-center justify-center min-w-24"
            >
              {#if isProcessing}
                <Loader2 class="w-4 h-4 animate-spin" />
              {:else}
                <Send class="w-4 h-4" />
              {/if}
            </button>
          </div>
        </div>

        <!-- Sample Queries -->
        <div>
          <p class="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">Sample Queries:</p>
          <div class="flex flex-wrap gap-2">
            {#each sampleQueries as sampleQuery}
              <button
                class="px-3 py-1 text-xs bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-full hover:bg-slate-200 dark:hover:bg-slate-600 transition-colors"
                on:click={() => selectSampleQuery(sampleQuery)}
              >
                {sampleQuery}
              </button>
            {/each}
          </div>
        </div>
      </div>

      <!-- Response History -->
      <div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
        <h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">Response History</h3>
        
        {#if responses.length === 0}
          <div class="text-center py-8">
            <Activity class="w-12 h-12 text-slate-400 mx-auto mb-4" />
            <p class="text-slate-600 dark:text-slate-400">No queries yet. Try asking a legal question!</p>
          </div>
        {:else}
          <div class="space-y-4">
            {#each responses as response (response.id)}
              <div 
                class="border border-slate-200 dark:border-slate-600 rounded-lg p-4"
                transition:fly={{ y: 20, duration: 300 }}
              >
                <div class="flex items-center justify-between mb-2">
                  <div class="flex items-center space-x-2">
                    {@const ProtocolIcon = getProtocolIcon(response.protocol)}
                    <ProtocolIcon class="w-4 h-4 text-blue-600 dark:text-blue-400" />
                    <span class="text-sm font-medium text-slate-700 dark:text-slate-300 uppercase">
                      {response.protocol}
                    </span>
                  </div>
                  <div class="flex items-center space-x-2 text-xs text-slate-500 dark:text-slate-400">
                    <Clock class="w-3 h-3" />
                    <span>{formatLatency(response.processingTime)}</span>
                  </div>
                </div>
                
                <div class="mb-2">
                  <p class="text-sm font-medium text-slate-900 dark:text-slate-100">
                    Q: {response.query}
                  </p>
                </div>
                
                {#if response.error}
                  <div class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-3">
                    <p class="text-sm text-red-700 dark:text-red-300">
                      Error: {response.error}
                    </p>
                  </div>
                {:else if response.result}
                  <div class="bg-slate-50 dark:bg-slate-700 rounded-lg p-3">
                    <p class="text-sm text-slate-700 dark:text-slate-300">
                      {response.result.answer || response.result.response || 'No response content'}
                    </p>
                    
                    {#if response.result.sources?.length > 0}
                      <div class="mt-2 pt-2 border-t border-slate-200 dark:border-slate-600">
                        <p class="text-xs text-slate-500 dark:text-slate-400 mb-1">
                          Sources: {response.result.sources.length}
                        </p>
                        <div class="flex flex-wrap gap-1">
                          {#each response.result.sources.slice(0, 3) as source}
                            <span class="px-2 py-1 text-xs bg-blue-100 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 rounded">
                              {source.title || source.id}
                            </span>
                          {/each}
                        </div>
                      </div>
                    {/if}
                  </div>
                {/if}
              </div>
            {/each}
          </div>
        {/if}
      </div>
    </div>

    <!-- Metrics Sidebar -->
    <div class="space-y-6">
      <!-- Protocol Health -->
      <div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
        <h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">Protocol Health</h3>
        
        <div class="space-y-3">
          {#each protocolStatuses as status}
            {@const StatusIcon = getStatusIcon(status.status)}
            <div class="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-700 rounded-lg">
              <div class="flex items-center space-x-3">
                <StatusIcon class="w-5 h-5 {getStatusColor(status.status)}" />
                <div>
                  <p class="font-medium text-slate-900 dark:text-slate-100 uppercase">
                    {status.protocol}
                  </p>
                  <p class="text-sm text-slate-600 dark:text-slate-400">
                    {status.status}
                  </p>
                </div>
              </div>
              
              {#if status.latency > 0}
                <div class="text-right">
                  <p class="text-sm font-medium text-slate-900 dark:text-slate-100">
                    {formatLatency(status.latency)}
                  </p>
                  <p class="text-xs text-slate-500 dark:text-slate-400">
                    Latency
                  </p>
                </div>
              {/if}
            </div>
          {/each}
        </div>
      </div>

      <!-- Performance Metrics -->
      <div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
        <h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">Performance</h3>
        
        <div class="space-y-4">
          <div class="flex justify-between items-center">
            <span class="text-slate-600 dark:text-slate-400">Total Requests</span>
            <span class="font-semibold text-slate-900 dark:text-slate-100">
              {metrics.totalRequests || 0}
            </span>
          </div>
          
          <div class="flex justify-between items-center">
            <span class="text-slate-600 dark:text-slate-400">Success Rate</span>
            <span class="font-semibold text-green-600">
              {metrics.totalRequests > 0 
                ? Math.round((metrics.successfulRequests / metrics.totalRequests) * 100)
                : 0}%
            </span>
          </div>
          
          <div class="flex justify-between items-center">
            <span class="text-slate-600 dark:text-slate-400">Avg Latency</span>
            <span class="font-semibold text-slate-900 dark:text-slate-100">
              {formatLatency(metrics.averageLatency || 0)}
            </span>
          </div>

          <!-- Protocol Usage -->
          {#if metrics.protocolUsage}
            <div class="pt-4 border-t border-slate-200 dark:border-slate-600">
              <p class="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">Protocol Usage</p>
              {#each Object.entries(metrics.protocolUsage) as [protocol, usage]}
                <div class="flex justify-between items-center text-sm">
                  <span class="text-slate-600 dark:text-slate-400 uppercase">{protocol}</span>
                  <span class="font-medium text-slate-900 dark:text-slate-100">{usage}</span>
                </div>
              {/each}
            </div>
          {/if}
        </div>
      </div>

      <!-- System Info -->
      <div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
        <h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">System Info</h3>
        
        <div class="space-y-2 text-sm">
          <div class="flex justify-between">
            <span class="text-slate-600 dark:text-slate-400">Pipeline</span>
            <span class="font-medium text-slate-900 dark:text-slate-100">Enhanced RAG</span>
          </div>
          <div class="flex justify-between">
            <span class="text-slate-600 dark:text-slate-400">Routing</span>
            <span class="font-medium text-slate-900 dark:text-slate-100">Multi-Protocol</span>
          </div>
          <div class="flex justify-between">
            <span class="text-slate-600 dark:text-slate-400">Fallback</span>
            <span class="font-medium text-green-600">Enabled</span>
          </div>
          <div class="flex justify-between">
            <span class="text-slate-600 dark:text-slate-400">Caching</span>
            <span class="font-medium text-green-600">Active</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<style>
  .multi-protocol-rag-demo {
    font-family: system-ui, -apple-system, sans-serif;
  }
</style>