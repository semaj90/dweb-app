<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { io, type Socket } from 'socket.io-client';
  import { uploadStore } from '$lib/stores/upload-machine';
  import { writable } from 'svelte/store';
  
  // Props
  interface Props {
    caseId?: string;
    uploadId?: string;
    showTensorMetrics?: boolean;
    enableAttentionTracking?: boolean;
  }
  let {
    caseId = '',
    uploadId = '',
    showTensorMetrics = false,
    enableAttentionTracking = true
  } = $props();

  // WebSocket connection
  let socket: Socket | null = null;
  let connectionStatus = writable<'disconnected' | 'connecting' | 'connected'>('disconnected');
  
  // Progress tracking
  let progressData = writable({
    stage: 'idle',
    progress: 0,
    status: 'pending',
    metrics: {},
    error: null as string | null,
  });

  // Tensor processing results
  let tensorResults = writable({
    clusters: [],
    embeddings: [],
    interpolationResults: [],
    metrics: {},
  });

  // AI context suggestions
  let aiSuggestions = writable({
    suggestions: [],
    relevantDocuments: [],
    confidence: 0,
  });

  // Real-time metrics
  let realtimeMetrics = writable({
    uploadSpeed: 0,
    processingTime: 0,
    memoryUsage: 0,
    gpuUtilization: 0,
  });

  onMount(() => {
    initializeWebSocket();
    if (enableAttentionTracking) {
      setupAttentionTracking();
    }
  });

  onDestroy(() => {
    cleanupWebSocket();
    cleanupAttentionTracking();
  });

  function initializeWebSocket() {
    connectionStatus.set('connecting');
    
    // Connect to WebSocket server
    socket = io('/api/ws', {
      transports: ['websocket', 'polling'],
      timeout: 5000,
    });

    socket.on('connect', () => {
      console.log('🔌 WebSocket connected');
      connectionStatus.set('connected');
      
      // Join relevant rooms
      if (caseId) {
        socket?.emit('join-case', caseId);
      }
      if (uploadId) {
        socket?.emit('join-upload', uploadId);
      }
    });

    socket.on('disconnect', () => {
      console.log('🔌 WebSocket disconnected');
      connectionStatus.set('disconnected');
    });

    // Upload progress updates
    socket.on('upload-progress', (data) => {
      console.log('📊 Upload progress:', data);
      progressData.update(current => ({
        ...current,
        stage: data.stage || current.stage,
        progress: data.progress || current.progress,
        status: data.status || current.status,
        metrics: { ...current.metrics, ...data.metrics },
      }));

      // Update XState machine
      if (data.stage && data.progress !== undefined) {
        uploadStore.send({
          type: 'PROCESSING_PROGRESS',
          stage: data.stage,
          progress: data.progress,
        });
      }

      // Update real-time metrics
      if (data.metrics) {
        realtimeMetrics.update(current => ({
          ...current,
          uploadSpeed: data.metrics.uploadSpeed || current.uploadSpeed,
          processingTime: data.metrics.processingTime || current.processingTime,
          memoryUsage: data.metrics.memoryUsage || current.memoryUsage,
        }));
      }
    });

    // Case-wide progress updates
    socket.on('case-progress', (data) => {
      console.log('📂 Case progress:', data);
      // Handle case-level progress updates
    });

    // Tensor processing results
    socket.on('tensor-result', (data) => {
      console.log('🧮 Tensor result:', data);
      
      if (showTensorMetrics) {
        tensorResults.update(current => ({
          ...current,
          ...data.result,
          metrics: { ...current.metrics, ...data.result.metrics },
        }));

        // Update GPU utilization if available
        if (data.result.metrics?.gpuUtilization) {
          realtimeMetrics.update(current => ({
            ...current,
            gpuUtilization: data.result.metrics.gpuUtilization,
          }));
        }
      }

      // Notify XState machine of tensor completion
      uploadStore.send({
        type: 'PROCESSING_COMPLETE',
        stage: 'tensor',
        result: data.result,
      });
    });

    // AI context suggestions
    socket.on('ai-context-suggestion', (data) => {
      console.log('🤖 AI suggestions:', data);
      aiSuggestions.set(data);
    });

    // Error handling
    socket.on('upload-error', (data) => {
      console.error('❌ Upload error:', data);
      progressData.update(current => ({
        ...current,
        status: 'failed',
        error: data.error.message || 'Unknown error',
      }));

      uploadStore.send({
        type: 'PROCESSING_FAILED',
        stage: data.stage || 'unknown',
        error: data.error.message || 'Unknown error',
      });
    });

    // Document collaboration (for future use)
    socket.on('document-change', (data) => {
      console.log('📝 Document change:', data);
      // Handle real-time document collaboration
    });

    // Search result streaming
    socket.on('search-results', (data) => {
      console.log('🔍 Search results:', data);
      // Handle streaming search results
    });
  }

  function cleanupWebSocket() {
    if (socket) {
      socket.disconnect();
      socket = null;
    }
  }

  // Attention tracking setup
  let attentionListeners: Array<() => void> = [];

  function setupAttentionTracking() {
    if (!socket) return;

    const trackEvent = (type: string, metadata?: any) => {
      socket?.emit('user-attention', {
        type,
        timestamp: new Date().toISOString(),
        metadata,
      });
    };

    // Focus/blur tracking
    const focusHandler = () => trackEvent('focus');
    const blurHandler = () => trackEvent('blur');
    
    window.addEventListener('focus', focusHandler);
    window.addEventListener('blur', blurHandler);
    
    attentionListeners.push(
      () => window.removeEventListener('focus', focusHandler),
      () => window.removeEventListener('blur', blurHandler)
    );

    // Scroll tracking (throttled)
    let scrollTimeout: number
    const scrollHandler = () => {
      clearTimeout(scrollTimeout);
      scrollTimeout = setTimeout(() => {
        trackEvent('scroll', { 
          scrollY: window.scrollY,
          scrollX: window.scrollX,
        });
      }, 100);
    };

    window.addEventListener('scroll', scrollHandler, { passive: true });
    attentionListeners.push(() => {
      window.removeEventListener('scroll', scrollHandler);
      clearTimeout(scrollTimeout);
    });

    // Click tracking
    const clickHandler = (e: MouseEvent) => {
      trackEvent('click', {
        x: e.clientX,
        y: e.clientY,
        target: e.target instanceof Element ? e.target.tagName : null,
      });
    };

    document.addEventListener('click', clickHandler);
    attentionListeners.push(() => document.removeEventListener('click', clickHandler));
  }

  function cleanupAttentionTracking() {
    attentionListeners.forEach(cleanup => cleanup());
    attentionListeners = [];
  }

  // Typing event tracking
  export function trackTyping(query: string) {
    if (!socket || !enableAttentionTracking) return;
    
    socket.emit('user-attention', {
      type: 'typing',
      timestamp: new Date().toISOString(),
      metadata: { query },
    });
  }

  // Subscribe to tensor job updates
  export function subscribeTensorJob(jobId: string) {
    if (!socket) return;
    socket.emit('subscribe-tensor', jobId);
  }

  // Subscribe to search results
  export function subscribeSearch(searchId: string) {
    if (!socket) return;
    socket.emit('subscribe-search', searchId);
  }

  // Format bytes for display
  function formatBytes(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  // Format duration for display
  function formatDuration(seconds: number): string {
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds.toFixed(0)}s`;
  }
</script>

<!-- Connection Status -->
<div class="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 mb-6">
  <div class="flex items-center justify-between mb-4">
    <h3 class="text-lg font-semibold text-gray-900 dark:text-white">
      Upload Progress
    </h3>
    <div class="flex items-center gap-2">
      <div class="w-3 h-3 rounded-full {$connectionStatus === 'connected' ? 'bg-green-500' : $connectionStatus === 'connecting' ? 'bg-yellow-500' : 'bg-red-500'}"></div>
      <span class="text-sm text-gray-600 dark:text-gray-400 capitalize">
        {$connectionStatus}
      </span>
    </div>
  </div>

  <!-- Progress Bar -->
  <div class="mb-4">
    <div class="flex justify-between items-center mb-2">
      <span class="text-sm font-medium text-gray-700 dark:text-gray-300 capitalize">
        {$progressData.stage}
      </span>
      <span class="text-sm text-gray-600 dark:text-gray-400">
        {$progressData.progress}%
      </span>
    </div>
    
    <div class="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
      <div 
        class="bg-blue-600 h-2 rounded-full transition-all duration-300 ease-out"
        style="width: {$progressData.progress}%"
      ></div>
    </div>
    
    <div class="flex justify-between items-center mt-2">
      <span class="text-xs text-gray-500 dark:text-gray-400 capitalize">
        Status: {$progressData.status}
      </span>
      {#if $progressData.error}
        <span class="text-xs text-red-500">
          Error: {$progressData.error}
        </span>
      {/if}
    </div>
  </div>

  <!-- Real-time Metrics -->
  <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
    <div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
      <div class="text-xs text-gray-500 dark:text-gray-400 mb-1">Upload Speed</div>
      <div class="text-sm font-semibold text-gray-900 dark:text-white">
        {formatBytes($realtimeMetrics.uploadSpeed)}/s
      </div>
    </div>
    
    <div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
      <div class="text-xs text-gray-500 dark:text-gray-400 mb-1">Processing Time</div>
      <div class="text-sm font-semibold text-gray-900 dark:text-white">
        {formatDuration($realtimeMetrics.processingTime)}
      </div>
    </div>
    
    <div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
      <div class="text-xs text-gray-500 dark:text-gray-400 mb-1">Memory Usage</div>
      <div class="text-sm font-semibold text-gray-900 dark:text-white">
        {formatBytes($realtimeMetrics.memoryUsage)}
      </div>
    </div>
    
    <div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
      <div class="text-xs text-gray-500 dark:text-gray-400 mb-1">GPU Utilization</div>
      <div class="text-sm font-semibold text-gray-900 dark:text-white">
        {$realtimeMetrics.gpuUtilization}%
      </div>
    </div>
  </div>
</div>

<!-- Tensor Processing Results -->
{#if showTensorMetrics && Object.keys($tensorResults.metrics).length > 0}
  <div class="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 mb-6">
    <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">
      Tensor Processing Results
    </h3>
    
    <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
      <div class="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
        <div class="text-sm text-blue-600 dark:text-blue-400 mb-1">SOM Clusters</div>
        <div class="text-2xl font-bold text-blue-700 dark:text-blue-300">
          {$tensorResults.clusters.length || 0}
        </div>
      </div>
      
      <div class="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
        <div class="text-sm text-green-600 dark:text-green-400 mb-1">Embeddings</div>
        <div class="text-2xl font-bold text-green-700 dark:text-green-300">
          {$tensorResults.embeddings.length || 0}
        </div>
      </div>
      
      <div class="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
        <div class="text-sm text-purple-600 dark:text-purple-400 mb-1">Interpolations</div>
        <div class="text-2xl font-bold text-purple-700 dark:text-purple-300">
          {$tensorResults.interpolationResults.length || 0}
        </div>
      </div>
    </div>

    <!-- Detailed Metrics -->
    {#if Object.keys($tensorResults.metrics).length > 0}
      <div class="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
        <h4 class="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Processing Metrics
        </h4>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
          {#each Object.entries($tensorResults.metrics) as [key, value]}
            <div class="bg-gray-50 dark:bg-gray-700 rounded px-2 py-1">
              <span class="text-gray-500 dark:text-gray-400">{key}:</span>
              <span class="text-gray-900 dark:text-white ml-1">{value}</span>
            </div>
          {/each}
        </div>
      </div>
    {/if}
  </div>
{/if}

<!-- AI Context Suggestions -->
{#if $aiSuggestions.suggestions.length > 0}
  <div class="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
    <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">
      AI Context Suggestions
    </h3>
    
    <div class="space-y-3">
      {#each $aiSuggestions.suggestions as suggestion}
        <div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
          <div class="text-sm text-gray-900 dark:text-white">
            {suggestion.text}
          </div>
          <div class="text-xs text-gray-500 dark:text-gray-400 mt-1">
            Confidence: {Math.round(suggestion.confidence * 100)}%
          </div>
        </div>
      {/each}
    </div>

    {#if $aiSuggestions.relevantDocuments.length > 0}
      <div class="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
        <h4 class="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Relevant Documents
        </h4>
        <div class="space-y-2">
          {#each $aiSuggestions.relevantDocuments as doc}
            <div class="text-sm text-blue-600 dark:text-blue-400 hover:underline cursor-pointer">
              {doc.title} ({doc.relevanceScore}% match)
            </div>
          {/each}
        </div>
      </div>
    {/if}
  </div>
{/if}

<style>
  /* Add any custom styles here */
  .transition-all {
    transition-property: all
    transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
    transition-duration: 150ms;
  }
</style>