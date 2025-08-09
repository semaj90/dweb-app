<!--
  GPU Processing Orchestrator - XState-powered document processing interface
  Manages concurrent GPU processing with real-time monitoring
-->

<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { createGPUProcessingActor, type DocumentInput, type ProcessingResult } from '$lib/state/gpu-processing-machine';
  import { Button } from 'bits-ui';
  import { Progress } from 'bits-ui';
  import { Badge } from 'bits-ui';
  import { Card } from 'bits-ui';
  import { Tabs } from 'bits-ui';
  import { fade, fly } from 'svelte/transition';

  // Props
  export let documents: DocumentInput[] = [];
  export let autoStart: boolean = false;
  export let maxConcurrent: number = 5;

  // XState actor
  const gpuActor = createGPUProcessingActor();
  
  // Reactive state
  let state = $state(gpuActor.getSnapshot());
  let isProcessing = $derived(state.matches('processing'));
  let isPaused = $derived(state.matches('paused'));
  let processingQueue = $derived(state.context.processingQueue);
  let activeProcessing = $derived(state.context.activeProcessing);
  let completedDocuments = $derived(state.context.completedDocuments);
  let errorDocuments = $derived(state.context.errorDocuments);
  let serviceHealth = $derived(state.context.serviceHealth);
  let metrics = $derived(state.context.metrics);

  // UI state
  let selectedTab = $state('queue');
  let showDetails = $state(false);
  let newDocumentContent = $state('');
  let newDocumentTitle = $state('');
  let processType = $state('full');
  let priority = $state(5);

  // Start the actor
  onMount(() => {
    gpuActor.start();
    
    // Subscribe to state changes
    const subscription = gpuActor.subscribe((snapshot) => {
      state = snapshot;
    });

    // Auto-start processing if enabled
    if (autoStart && documents.length > 0) {
      gpuActor.send({ type: 'BATCH_PROCESS', documents });
    }

    // Periodic health checks
    const healthCheckInterval = setInterval(() => {
      gpuActor.send({ type: 'SERVICE_HEALTH_CHECK' });
    }, 30000); // Every 30 seconds

    return () => {
      subscription.unsubscribe();
      clearInterval(healthCheckInterval);
    };
  });

  onDestroy(() => {
    gpuActor.stop();
  });

  // Action handlers
  function addDocument() {
    if (!newDocumentContent.trim()) return;

    const document: DocumentInput = {
      documentId: `doc_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      content: newDocumentContent,
      title: newDocumentTitle || undefined,
      options: {
        processType: processType as any,
        priority,
        timeout: 30000,
        retries: 3,
        batchSize: 1
      }
    };

    gpuActor.send({ type: 'PROCESS_DOCUMENT', ...document });
    
    // Clear form
    newDocumentContent = '';
    newDocumentTitle = '';
  }

  function processBatch() {
    if (documents.length === 0) return;
    gpuActor.send({ type: 'BATCH_PROCESS', documents });
  }

  function pauseProcessing() {
    gpuActor.send({ type: 'PAUSE_PROCESSING' });
  }

  function resumeProcessing() {
    gpuActor.send({ type: 'RESUME_PROCESSING' });
  }

  function clearQueue() {
    gpuActor.send({ type: 'CLEAR_QUEUE' });
  }

  function cancelDocument(documentId: string) {
    gpuActor.send({ type: 'CANCEL', documentId });
  }

  function retryDocument(documentId: string) {
    gpuActor.send({ type: 'RETRY', documentId });
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'healthy': return 'bg-green-100 text-green-800';
      case 'unhealthy': return 'bg-red-100 text-red-800';
      case 'unknown': return 'bg-gray-100 text-gray-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  }

  function getProcessingStatusColor(status: string): string {
    switch (status) {
      case 'queued': return 'bg-blue-100 text-blue-800';
      case 'processing': return 'bg-yellow-100 text-yellow-800';
      case 'completed': return 'bg-green-100 text-green-800';
      case 'failed': return 'bg-red-100 text-red-800';
      case 'cancelled': return 'bg-gray-100 text-gray-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  }

  function formatDuration(ms: number): string {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}m`;
  }
</script>

<div class="gpu-processing-orchestrator p-6 max-w-7xl mx-auto">
  <!-- Header -->
  <div class="mb-6">
    <h1 class="text-3xl font-bold text-gray-900 mb-2">
      🎮 GPU Processing Orchestrator
    </h1>
    <p class="text-gray-600">
      XState-powered document processing with Go SIMD and Node.js GPU services
    </p>
  </div>

  <!-- Status Bar -->
  <Card.Root class="mb-6">
    <Card.Header>
      <Card.Title>System Status</Card.Title>
    </Card.Header>
    <Card.Content>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <!-- Service Health -->
        <div class="text-center">
          <div class="text-2xl font-bold text-gray-900">{state.value}</div>
          <div class="text-sm text-gray-600">Current State</div>
        </div>
        
        <div class="text-center">
          <Badge class={getStatusColor(serviceHealth.goSimd)}>
            Go SIMD: {serviceHealth.goSimd}
          </Badge>
          <div class="text-sm text-gray-600 mt-1">Service Health</div>
        </div>
        
        <div class="text-center">
          <Badge class={getStatusColor(serviceHealth.nodeGpu)}>
            Node GPU: {serviceHealth.nodeGpu}
          </Badge>
          <div class="text-sm text-gray-600 mt-1">Service Health</div>
        </div>
        
        <!-- Processing Stats -->
        <div class="text-center">
          <div class="text-2xl font-bold text-blue-600">{activeProcessing.size}</div>
          <div class="text-sm text-gray-600">Active Processing</div>
        </div>
      </div>
      
      <!-- Metrics Row -->
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4 pt-4 border-t">
        <div class="text-center">
          <div class="text-lg font-semibold text-green-600">{completedDocuments.size}</div>
          <div class="text-sm text-gray-600">Completed</div>
        </div>
        
        <div class="text-center">
          <div class="text-lg font-semibold text-red-600">{errorDocuments.size}</div>
          <div class="text-sm text-gray-600">Errors</div>
        </div>
        
        <div class="text-center">
          <div class="text-lg font-semibold text-purple-600">
            {metrics.avgProcessingTime.toFixed(1)}ms
          </div>
          <div class="text-sm text-gray-600">Avg Time</div>
        </div>
        
        <div class="text-center">
          <div class="text-lg font-semibold text-indigo-600">
            {metrics.throughputPerMinute.toFixed(1)}/min
          </div>
          <div class="text-sm text-gray-600">Throughput</div>
        </div>
      </div>
    </Card.Content>
  </Card.Root>

  <!-- Controls -->
  <Card.Root class="mb-6">
    <Card.Header>
      <Card.Title>Processing Controls</Card.Title>
    </Card.Header>
    <Card.Content>
      <div class="flex flex-wrap gap-3">
        {#if !isProcessing}
          <Button.Root
            on:click={processBatch}
            disabled={documents.length === 0}
            class="bg-blue-600 hover:bg-blue-700"
          >
            🚀 Start Processing
          </Button.Root>
        {:else}
          {#if !isPaused}
            <Button.Root on:click={pauseProcessing} class="bg-yellow-600 hover:bg-yellow-700">
              ⏸️ Pause
            </Button.Root>
          {:else}
            <Button.Root on:click={resumeProcessing} class="bg-green-600 hover:bg-green-700">
              ▶️ Resume
            </Button.Root>
          {/if}
        {/if}
        
        <Button.Root
          on:click={clearQueue}
          variant="outline"
          class="border-red-300 text-red-700 hover:bg-red-50"
        >
          🗑️ Clear Queue
        </Button.Root>
        
        <Button.Root
          on:click={() => gpuActor.send({ type: 'SERVICE_HEALTH_CHECK' })}
          variant="outline"
        >
          🔍 Health Check
        </Button.Root>
        
        <Button.Root
          on:click={() => showDetails = !showDetails}
          variant="outline"
        >
          {showDetails ? '👁️ Hide Details' : '📊 Show Details'}
        </Button.Root>
      </div>
    </Card.Content>
  </Card.Root>

  <!-- Add Document Form -->
  <Card.Root class="mb-6">
    <Card.Header>
      <Card.Title>Add Document</Card.Title>
    </Card.Header>
    <Card.Content>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">
            Document Title (optional)
          </label>
          <input
            bind:value={newDocumentTitle}
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Enter document title..."
          />
        </div>
        
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">
            Processing Type
          </label>
          <select
            bind:value={processType}
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="embeddings">Embeddings Only</option>
            <option value="clustering">Clustering Only</option>
            <option value="similarity">Similarity Only</option>
            <option value="boost">Boost Transform</option>
            <option value="full">Full Processing</option>
          </select>
        </div>
      </div>
      
      <div class="mb-4">
        <label class="block text-sm font-medium text-gray-700 mb-2">
          Document Content
        </label>
        <textarea
          bind:value={newDocumentContent}
          class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 h-32"
          placeholder="Enter document content to process..."
        ></textarea>
      </div>
      
      <Button.Root
        on:click={addDocument}
        disabled={!newDocumentContent.trim()}
        class="bg-green-600 hover:bg-green-700"
      >
        ➕ Add Document
      </Button.Root>
    </Card.Content>
  </Card.Root>

  <!-- Processing Queue Tabs -->
  <Tabs.Root bind:value={selectedTab} class="w-full">
    <Tabs.List class="grid w-full grid-cols-4">
      <Tabs.Trigger value="queue">
        Queue ({processingQueue.length})
      </Tabs.Trigger>
      <Tabs.Trigger value="active">
        Active ({activeProcessing.size})
      </Tabs.Trigger>
      <Tabs.Trigger value="completed">
        Completed ({completedDocuments.size})
      </Tabs.Trigger>
      <Tabs.Trigger value="errors">
        Errors ({errorDocuments.size})
      </Tabs.Trigger>
    </Tabs.List>

    <!-- Queue Tab -->
    <Tabs.Content value="queue" class="mt-4">
      <Card.Root>
        <Card.Header>
          <Card.Title>Processing Queue</Card.Title>
        </Card.Header>
        <Card.Content>
          {#if processingQueue.length === 0}
            <div class="text-center py-8 text-gray-500">
              📝 No documents in queue
            </div>
          {:else}
            <div class="space-y-3">
              {#each processingQueue as queuedDoc (queuedDoc.document.documentId)}
                <div class="flex items-center justify-between p-3 border rounded-lg" transition:fade>
                  <div class="flex-1">
                    <div class="font-medium">
                      {queuedDoc.document.title || queuedDoc.document.documentId}
                    </div>
                    <div class="text-sm text-gray-600">
                      {queuedDoc.document.content.substring(0, 100)}...
                    </div>
                    <div class="flex items-center gap-2 mt-2">
                      <Badge class={getProcessingStatusColor(queuedDoc.status)}>
                        {queuedDoc.status}
                      </Badge>
                      {#if queuedDoc.retryCount > 0}
                        <Badge class="bg-orange-100 text-orange-800">
                          Retry {queuedDoc.retryCount}
                        </Badge>
                      {/if}
                    </div>
                  </div>
                  
                  <div class="flex gap-2">
                    <Button.Root
                      size="sm"
                      variant="outline"
                      on:click={() => cancelDocument(queuedDoc.document.documentId)}
                    >
                      Cancel
                    </Button.Root>
                  </div>
                </div>
              {/each}
            </div>
          {/if}
        </Card.Content>
      </Card.Root>
    </Tabs.Content>

    <!-- Active Processing Tab -->
    <Tabs.Content value="active" class="mt-4">
      <Card.Root>
        <Card.Header>
          <Card.Title>Active Processing</Card.Title>
        </Card.Header>
        <Card.Content>
          {#if activeProcessing.size === 0}
            <div class="text-center py-8 text-gray-500">
              ⚡ No active processing
            </div>
          {:else}
            <div class="space-y-3">
              {#each Array.from(activeProcessing.values()) as activeDoc (activeDoc.document.documentId)}
                <div class="flex items-center justify-between p-3 border rounded-lg bg-yellow-50" transition:fly={{ y: -10 }}>
                  <div class="flex-1">
                    <div class="font-medium">
                      {activeDoc.document.title || activeDoc.document.documentId}
                    </div>
                    <div class="text-sm text-gray-600">
                      Processing for {formatDuration(Date.now() - (activeDoc.startTime || 0))}
                    </div>
                    <Progress.Root value={50} class="w-full mt-2">
                      <Progress.Indicator class="bg-blue-600 transition-all" />
                    </Progress.Root>
                  </div>
                  
                  <div class="flex gap-2">
                    <Button.Root
                      size="sm"
                      variant="outline"
                      on:click={() => cancelDocument(activeDoc.document.documentId)}
                    >
                      Cancel
                    </Button.Root>
                  </div>
                </div>
              {/each}
            </div>
          {/if}
        </Card.Content>
      </Card.Root>
    </Tabs.Content>

    <!-- Completed Tab -->
    <Tabs.Content value="completed" class="mt-4">
      <Card.Root>
        <Card.Header>
          <Card.Title>Completed Documents</Card.Title>
        </Card.Header>
        <Card.Content>
          {#if completedDocuments.size === 0}
            <div class="text-center py-8 text-gray-500">
              ✅ No completed documents
            </div>
          {:else}
            <div class="space-y-3">
              {#each Array.from(completedDocuments.entries()) as [docId, result] (docId)}
                <div class="p-3 border rounded-lg bg-green-50" transition:fade>
                  <div class="font-medium">{docId}</div>
                  <div class="text-sm text-gray-600">
                    Processed in {formatDuration(result.processingTime)}
                  </div>
                  {#if showDetails}
                    <div class="mt-2 text-xs text-gray-500">
                      <pre class="bg-gray-100 p-2 rounded overflow-x-auto">
{JSON.stringify(result.metadata, null, 2)}
                      </pre>
                    </div>
                  {/if}
                </div>
              {/each}
            </div>
          {/if}
        </Card.Content>
      </Card.Root>
    </Tabs.Content>

    <!-- Errors Tab -->
    <Tabs.Content value="errors" class="mt-4">
      <Card.Root>
        <Card.Header>
          <Card.Title>Processing Errors</Card.Title>
        </Card.Header>
        <Card.Content>
          {#if errorDocuments.size === 0}
            <div class="text-center py-8 text-gray-500">
              🎉 No errors
            </div>
          {:else}
            <div class="space-y-3">
              {#each Array.from(errorDocuments.entries()) as [docId, errorInfo] (docId)}
                <div class="p-3 border rounded-lg bg-red-50" transition:fade>
                  <div class="font-medium text-red-800">{docId}</div>
                  <div class="text-sm text-red-600 mb-2">
                    {errorInfo.error}
                  </div>
                  <div class="flex gap-2">
                    <Button.Root
                      size="sm"
                      on:click={() => retryDocument(docId)}
                      class="bg-blue-600 hover:bg-blue-700"
                    >
                      🔄 Retry
                    </Button.Root>
                  </div>
                </div>
              {/each}
            </div>
          {/if}
        </Card.Content>
      </Card.Root>
    </Tabs.Content>
  </Tabs.Root>
</div>

<style>
  .gpu-processing-orchestrator {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }
</style>