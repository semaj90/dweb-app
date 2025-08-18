// Enhanced Evidence Processing Component
<script lang="ts">
  import { createActor } from 'xstate';
  import { uploadMachine, getFileProgress, getAllFilesStatus, getOverallProgress } from '$lib/machines/uploadMachine';
  import type { ProgressMsg } from '$lib/types/progress';
  import { onMount, onDestroy } from 'svelte';
  
  // Props
  export let evidenceId: string;
  export let steps: string[] = ['ocr', 'embedding', 'analysis'];
  export let autoStart = false;
  export let onComplete: ((result: any) => void) | undefined = undefined;
  export let onError: ((error: any) => void) | undefined = undefined;

  // Machine actor
  let uploadActor = createActor(uploadMachine);
  
  // Reactive state
  $: machineState = uploadActor.getSnapshot();
  $: currentState = machineState.value;
  $: context = machineState.context;
  $: fileProgress = getFileProgress(context, evidenceId);
  $: allFiles = getAllFilesStatus(context);
  $: overallProgress = getOverallProgress(context);
  $: isConnected = context.wsConnected;
  $: hasError = fileProgress.status === 'error' || context.lastError;
  $: isProcessing = currentState === 'processing';
  $: isComplete = fileProgress.status === 'done';

  // Local state
  let showDetails = false;
  let showLogs = false;
  let processingLogs: Array<{ timestamp: string; message: string; type: 'info' | 'error' | 'success' }> = [];

  onMount(() => {
    uploadActor.start();
    
    // Subscribe to machine state changes
    uploadActor.subscribe((state) => {
      console.log('🎭 Machine state changed:', state.value, state.context);
      
      // Handle completion
      if (state.context.files[evidenceId]?.status === 'done') {
        const result = state.context.files[evidenceId]?.result;
        onComplete?.(result);
        addLog('Processing completed successfully', 'success');
      }
      
      // Handle errors
      if (state.context.lastError || state.context.files[evidenceId]?.error) {
        const error = state.context.lastError || state.context.files[evidenceId]?.error;
        onError?.(error);
        addLog(`Error: ${error}`, 'error');
      }
      
      // Log state changes
      if (state.value !== 'idle') {
        addLog(`State: ${String(state.value)}`, 'info');
      }
    });
    
    // Auto-start if requested
    if (autoStart) {
      startProcessing();
    }
  });

  onDestroy(() => {
    uploadActor.stop();
  });

  function addLog(message: string, type: 'info' | 'error' | 'success' = 'info') {
    processingLogs = [
      ...processingLogs,
      {
        timestamp: new Date().toLocaleTimeString(),
        message,
        type
      }
    ].slice(-50); // Keep last 50 logs
  }

  async function startProcessing() {
    try {
      addLog('Starting evidence processing...', 'info');
      
      // Make API call to start processing
      const response = await fetch('/api/evidence/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          evidenceId,
          steps
        })
      });

      if (!response.ok) {
        throw new Error(`Failed to start processing: ${response.statusText}`);
      }

      const data = await response.json();
      const sessionId = data.sessionId;

      addLog(`Session created: ${sessionId}`, 'info');

      // Start the machine with the session ID
      uploadActor.send({
        type: 'START_PROCESS',
        sessionId,
        fileId: evidenceId
      });

    } catch (error) {
      console.error('❌ Failed to start processing:', error);
      addLog(`Failed to start: ${error.message}`, 'error');
      onError?.(error);
    }
  }

  function cancelProcessing() {
    uploadActor.send({ type: 'CANCEL', fileId: evidenceId });
    addLog('Cancelling processing...', 'info');
  }

  function retry() {
    uploadActor.send({ type: 'RETRY' });
    addLog('Retrying processing...', 'info');
  }

  function reset() {
    uploadActor.send({ type: 'RESET' });
    processingLogs = [];
    addLog('Reset complete', 'info');
  }

  function getStepIcon(step: string): string {
    switch (step) {
      case 'ocr': return '🔍';
      case 'embedding': return '🧠';
      case 'rag':
      case 'analysis': return '📚';
      default: return '⚙️';
    }
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'done': return 'text-green-600';
      case 'error': return 'text-red-600';
      case 'processing': return 'text-blue-600';
      case 'uploading': return 'text-yellow-600';
      default: return 'text-gray-600';
    }
  }

  function formatFragment(fragment: any): string {
    if (!fragment) return '';
    
    if (typeof fragment === 'string') return fragment;
    
    if (fragment.textPreview) return fragment.textPreview;
    if (fragment.snippet) return fragment.snippet;
    if (fragment.summary) return fragment.summary;
    
    return JSON.stringify(fragment, null, 2);
  }
</script>

<div class="evidence-processor border rounded-lg p-6 bg-white shadow-sm">
  <!-- Header -->
  <div class="flex items-center justify-between mb-4">
    <div class="flex items-center space-x-3">
      <div class="flex items-center space-x-2">
        {#if isConnected}
          <div class="w-3 h-3 bg-green-500 rounded-full animate-pulse" title="Connected"></div>
        {:else}
          <div class="w-3 h-3 bg-red-500 rounded-full" title="Disconnected"></div>
        {/if}
        <h3 class="text-lg font-semibold">Evidence Processing</h3>
      </div>
      
      {#if isProcessing}
        <div class="flex items-center space-x-2">
          <div class="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
          <span class="text-sm text-blue-600">Processing...</span>
        </div>
      {/if}
    </div>

    <!-- Action Buttons -->
    <div class="flex items-center space-x-2">
      {#if currentState === 'idle'}
        <button
          on:click={startProcessing}
          class="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
        >
          Start Processing
        </button>
      {/if}

      {#if isProcessing}
        <button
          on:click={cancelProcessing}
          class="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2"
        >
          Cancel
        </button>
      {/if}

      {#if hasError}
        <button
          on:click={retry}
          class="px-4 py-2 bg-yellow-600 text-white rounded-md hover:bg-yellow-700 focus:outline-none focus:ring-2 focus:ring-yellow-500 focus:ring-offset-2"
        >
          Retry
        </button>
      {/if}

      <button
        on:click={() => showDetails = !showDetails}
        class="px-3 py-2 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2"
      >
        {showDetails ? 'Hide' : 'Show'} Details
      </button>
    </div>
  </div>

  <!-- Progress Overview -->
  <div class="mb-6">
    <div class="flex items-center justify-between mb-2">
      <span class="text-sm font-medium text-gray-700">Overall Progress</span>
      <span class="text-sm text-gray-600">{overallProgress}%</span>
    </div>
    
    <div class="w-full bg-gray-200 rounded-full h-2">
      <div
        class="bg-blue-600 h-2 rounded-full transition-all duration-300 ease-out"
        style="width: {overallProgress}%"
      ></div>
    </div>
  </div>

  <!-- Current Step Info -->
  {#if fileProgress.step}
    <div class="mb-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
      <div class="flex items-center space-x-3">
        <span class="text-2xl">{getStepIcon(fileProgress.step)}</span>
        <div class="flex-1">
          <h4 class="font-medium text-blue-900 capitalize">{fileProgress.step}</h4>
          <div class="flex items-center space-x-4 mt-1">
            <div class="flex-1">
              <div class="w-full bg-blue-200 rounded-full h-1.5">
                <div
                  class="bg-blue-600 h-1.5 rounded-full transition-all duration-300"
                  style="width: {fileProgress.stepProgress || 0}%"
                ></div>
              </div>
            </div>
            <span class="text-sm text-blue-700">{fileProgress.stepProgress || 0}%</span>
          </div>
        </div>
      </div>

      <!-- Show fragment data if available -->
      {#if fileProgress.fragment}
        <div class="mt-3 p-3 bg-white rounded border">
          <h5 class="text-sm font-medium text-gray-700 mb-1">Live Update:</h5>
          <div class="text-sm text-gray-600 font-mono">
            {formatFragment(fileProgress.fragment)}
          </div>
        </div>
      {/if}
    </div>
  {/if}

  <!-- Error Display -->
  {#if hasError}
    <div class="mb-4 p-4 bg-red-50 rounded-lg border border-red-200">
      <div class="flex items-center space-x-2">
        <span class="text-red-500">❌</span>
        <h4 class="font-medium text-red-900">Error</h4>
      </div>
      <p class="text-red-700 mt-1">
        {fileProgress.error || context.lastError || 'An unknown error occurred'}
      </p>
    </div>
  {/if}

  <!-- Success Display -->
  {#if isComplete}
    <div class="mb-4 p-4 bg-green-50 rounded-lg border border-green-200">
      <div class="flex items-center space-x-2">
        <span class="text-green-500">✅</span>
        <h4 class="font-medium text-green-900">Processing Complete</h4>
      </div>
      <p class="text-green-700 mt-1">Evidence has been successfully processed through all steps.</p>
      
      {#if fileProgress.result}
        <details class="mt-3">
          <summary class="cursor-pointer text-green-800 font-medium">View Results</summary>
          <pre class="mt-2 p-3 bg-white rounded border text-sm overflow-auto">
{JSON.stringify(fileProgress.result, null, 2)}
          </pre>
        </details>
      {/if}
    </div>
  {/if}

  <!-- Detailed Information Panel -->
  {#if showDetails}
    <div class="border-t pt-4">
      <div class="space-y-4">
        <!-- Processing Steps -->
        <div>
          <h4 class="font-medium text-gray-900 mb-3">Processing Steps</h4>
          <div class="space-y-2">
            {#each steps as step, index}
              {@const isCurrentStep = fileProgress.step === step}
              {@const isCompleted = steps.indexOf(fileProgress.step || '') > index}
              <div class="flex items-center space-x-3 p-3 rounded-lg {isCurrentStep ? 'bg-blue-50 border border-blue-200' : isCompleted ? 'bg-green-50 border border-green-200' : 'bg-gray-50'}">
                <span class="text-xl">{getStepIcon(step)}</span>
                <div class="flex-1">
                  <div class="font-medium capitalize {isCurrentStep ? 'text-blue-900' : isCompleted ? 'text-green-900' : 'text-gray-700'}">
                    {step}
                  </div>
                  {#if isCurrentStep && fileProgress.stepProgress}
                    <div class="text-sm text-blue-600">{fileProgress.stepProgress}% complete</div>
                  {:else if isCompleted}
                    <div class="text-sm text-green-600">Completed</div>
                  {:else}
                    <div class="text-sm text-gray-500">Pending</div>
                  {/if}
                </div>
                <div class="w-6 h-6 rounded-full flex items-center justify-center">
                  {#if isCompleted}
                    <span class="text-green-600">✓</span>
                  {:else if isCurrentStep}
                    <div class="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                  {:else}
                    <div class="w-2 h-2 bg-gray-300 rounded-full"></div>
                  {/if}
                </div>
              </div>
            {/each}
          </div>
        </div>

        <!-- Connection Status -->
        <div>
          <h4 class="font-medium text-gray-900 mb-2">Connection Status</h4>
          <div class="flex items-center space-x-2">
            <div class="w-3 h-3 rounded-full {isConnected ? 'bg-green-500' : 'bg-red-500'}"></div>
            <span class="text-sm {isConnected ? 'text-green-700' : 'text-red-700'}">
              {isConnected ? 'Connected to real-time updates' : 'Disconnected'}
            </span>
          </div>
          <div class="text-sm text-gray-600 mt-1">
            State: {String(currentState)}
            {#if context.sessionId}
              | Session: {context.sessionId.substring(0, 8)}...
            {/if}
            {#if context.retryCount > 0}
              | Retries: {context.retryCount}
            {/if}
          </div>
        </div>

        <!-- Processing Logs -->
        <div>
          <div class="flex items-center justify-between mb-2">
            <h4 class="font-medium text-gray-900">Processing Logs</h4>
            <button
              on:click={() => showLogs = !showLogs}
              class="text-sm text-blue-600 hover:text-blue-800"
            >
              {showLogs ? 'Hide' : 'Show'} Logs
            </button>
          </div>
          
          {#if showLogs}
            <div class="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm max-h-64 overflow-y-auto">
              {#each processingLogs as log}
                <div class="flex items-start space-x-2 mb-1">
                  <span class="text-gray-500">[{log.timestamp}]</span>
                  <span class="flex-1 {log.type === 'error' ? 'text-red-400' : log.type === 'success' ? 'text-green-400' : 'text-gray-300'}">
                    {log.message}
                  </span>
                </div>
              {/each}
              
              {#if processingLogs.length === 0}
                <div class="text-gray-500 italic">No logs yet...</div>
              {/if}
            </div>
          {/if}
        </div>

        <!-- Reset Button -->
        <div class="pt-4 border-t">
          <button
            on:click={reset}
            class="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2"
          >
            Reset Processor
          </button>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .evidence-processor {
    /* Custom styles if needed */
  }
</style>
