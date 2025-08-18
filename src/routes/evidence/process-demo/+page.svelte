<script lang="ts">
  import { createActor } from 'xstate';
  import { evidenceProcessingMachine, startEvidenceProcessing, evidenceProcessingSelectors } from '$lib/machines/evidenceProcessingMachine';
  import { Button } from '$lib/components/ui/Button.svelte';
  import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '$lib/components/ui/Card';
  import { Badge } from '$lib/components/ui/Badge.svelte';
  import { Progress } from '$lib/components/ui/progress/Progress.svelte';

  // XState actor
  const actor = createActor(evidenceProcessingMachine);
  
  let state = $state(actor.getSnapshot());
  let files: FileList | null = $state(null);
  let isProcessing = $state(false);
  let logs: string[] = $state([]);
  
  // Subscribe to state changes
  actor.subscribe((snapshot) => {
    state = snapshot;
    console.log('State changed:', snapshot.value, snapshot.context);
  });
  
  actor.start();
  
  // Reactive selectors
  let currentState = $derived(state.value);
  let context = $derived(state.context);
  let sessionId = $derived(context.sessionId);
  let evidenceId = $derived(context.evidenceId);
  let error = $derived(context.error);
  
  function addLog(message: string) {
    logs = [...logs, `${new Date().toLocaleTimeString()}: ${message}`];
  }
  
  async function handleFileUpload(event: Event) {
    const target = event.target as HTMLInputElement;
    files = target.files;
  }
  
  async function startProcessing() {
    if (!files || files.length === 0) {
      addLog('No files selected');
      return;
    }
    
    isProcessing = true;
    addLog('Starting evidence processing...');
    
    try {
      // Simulate evidence ID (in real app, this would come from file upload)
      const testEvidenceId = `demo_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      addLog(`Generated evidence ID: ${testEvidenceId}`);
      
      const { sessionId, evidenceId, steps } = await startEvidenceProcessing(
        testEvidenceId, 
        ['ocr', 'embedding', 'analysis']
      );
      
      addLog(`Processing session started: ${sessionId}`);
      addLog(`Steps: ${steps.join(', ')}`);
      
      // Send event to XState machine
      actor.send({ 
        type: 'START_PROCESSING', 
        sessionId, 
        evidenceId, 
        steps 
      });
      
    } catch (error) {
      addLog(`Error: ${error instanceof Error ? error.message : String(error)}`);
      isProcessing = false;
    }
  }
  
  function reset() {
    actor.send({ type: 'RESET' });
    files = null;
    isProcessing = false;
    logs = [];
    addLog('System reset');
  }
  
  function retry() {
    if (evidenceProcessingSelectors.canRetry(state)) {
      addLog('Retrying processing...');
      actor.send({ type: 'RETRY_PROCESSING' });
    }
  }
  
  // Get file status for display
  function getFileStatus(fileId: string) {
    return evidenceProcessingSelectors.getFileStatus(state, fileId) || 'unknown';
  }
  
  function getFileProgress(fileId: string) {
    return evidenceProcessingSelectors.getFileProgress(state, fileId) || 0;
  }
  
  function getCurrentStep(fileId: string) {
    return evidenceProcessingSelectors.getCurrentStep(state, fileId) || 'waiting';
  }
  
  function getStatusColor(status: string) {
    switch (status) {
      case 'completed': return 'bg-green-500';
      case 'processing': return 'bg-blue-500';
      case 'failed': return 'bg-red-500';
      case 'queued': return 'bg-yellow-500';
      default: return 'bg-gray-500';
    }
  }
</script>

<svelte:head>
  <title>Evidence Processing Demo - Real-time WebSocket Progress</title>
</svelte:head>

<div class="container mx-auto py-8 px-4">
  <div class="max-w-4xl mx-auto space-y-6">
    <!-- Header -->
    <div class="text-center">
      <h1 class="text-3xl font-bold text-white mb-2">Evidence Processing Demo</h1>
      <p class="text-gray-300">Test the real-time evidence processing pipeline with WebSocket progress updates</p>
    </div>
    
    <!-- Status Overview -->
    <Card class="bg-gray-900 border-gray-700">
      <CardHeader>
        <CardTitle class="text-white flex items-center gap-2">
          System Status
          <Badge variant="outline" class={getStatusColor(String(currentState))}>
            {String(currentState).toUpperCase()}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent class="space-y-2 text-gray-300">
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <div class="text-xs text-gray-500">Session ID</div>
            <div class="font-mono text-xs">{sessionId || 'None'}</div>
          </div>
          <div>
            <div class="text-xs text-gray-500">Evidence ID</div>
            <div class="font-mono text-xs">{evidenceId || 'None'}</div>
          </div>
          <div>
            <div class="text-xs text-gray-500">WebSocket</div>
            <div class="font-mono text-xs">
              {evidenceProcessingSelectors.isProcessing(state) ? 'Connected' : 'Disconnected'}
            </div>
          </div>
          <div>
            <div class="text-xs text-gray-500">Files</div>
            <div class="font-mono text-xs">{Object.keys(context.files).length}</div>
          </div>
        </div>
      </CardContent>
    </Card>
    
    <!-- File Upload -->
    <Card class="bg-gray-900 border-gray-700">
      <CardHeader>
        <CardTitle class="text-white">Upload Evidence Files</CardTitle>
        <CardDescription class="text-gray-400">
          Select PDF files to process through the evidence pipeline
        </CardDescription>
      </CardHeader>
      <CardContent class="space-y-4">
        <input 
          type="file" 
          multiple 
          accept=".pdf"
          on:change={handleFileUpload}
          class="block w-full text-sm text-gray-300
                 file:mr-4 file:py-2 file:px-4
                 file:rounded-full file:border-0
                 file:text-sm file:font-semibold
                 file:bg-blue-500 file:text-white
                 hover:file:bg-blue-600 cursor-pointer"
        />
        
        {#if files && files.length > 0}
          <div class="text-sm text-gray-300">
            Selected: {files.length} file{files.length > 1 ? 's' : ''}
            {#each Array.from(files) as file}
              <div class="ml-4 text-xs text-gray-400">• {file.name} ({Math.round(file.size / 1024)}KB)</div>
            {/each}
          </div>
        {/if}
        
        <div class="flex gap-2">
          <Button 
            on:click={startProcessing} 
            disabled={!files || files.length === 0 || isProcessing}
            class="bg-blue-500 hover:bg-blue-600"
          >
            {#if isProcessing}
              Processing...
            {:else}
              Start Processing
            {/if}
          </Button>
          
          <Button 
            on:click={reset} 
            variant="outline"
            class="border-gray-600 text-gray-300 hover:bg-gray-800"
          >
            Reset
          </Button>
          
          {#if evidenceProcessingSelectors.canRetry(state)}
            <Button 
              on:click={retry}
              variant="outline" 
              class="border-yellow-600 text-yellow-300 hover:bg-yellow-800"
            >
              Retry
            </Button>
          {/if}
        </div>
      </CardContent>
    </Card>
    
    <!-- Processing Progress -->
    {#if Object.keys(context.files).length > 0}
      <Card class="bg-gray-900 border-gray-700">
        <CardHeader>
          <CardTitle class="text-white">Processing Progress</CardTitle>
        </CardHeader>
        <CardContent class="space-y-4">
          {#each Object.entries(context.files) as [fileId, fileData]}
            {@const status = getFileStatus(fileId)}
            {@const progress = getFileProgress(fileId)}
            {@const currentStep = getCurrentStep(fileId)}
            
            <div class="p-4 bg-gray-800 rounded-lg">
              <div class="flex justify-between items-center mb-2">
                <div class="font-medium text-white">{fileId}</div>
                <Badge variant="outline" class={getStatusColor(status)}>
                  {status.toUpperCase()}
                </Badge>
              </div>
              
              <div class="space-y-2">
                <div class="flex justify-between text-sm text-gray-400">
                  <span>Current Step: {currentStep}</span>
                  <span>{progress}%</span>
                </div>
                
                <Progress value={progress} class="h-2" />
                
                {#if fileData.fragment}
                  <div class="text-xs text-gray-500 bg-gray-700 p-2 rounded">
                    Latest update: {JSON.stringify(fileData.fragment, null, 2)}
                  </div>
                {/if}
                
                {#if fileData.result}
                  <div class="text-xs text-green-400 bg-gray-700 p-2 rounded">
                    ✅ Result: {JSON.stringify(fileData.result, null, 2)}
                  </div>
                {/if}
                
                {#if fileData.error}
                  <div class="text-xs text-red-400 bg-gray-700 p-2 rounded">
                    ❌ Error: {fileData.error.message}
                  </div>
                {/if}
              </div>
            </div>
          {/each}
        </CardContent>
      </Card>
    {/if}
    
    <!-- System Logs -->
    <Card class="bg-gray-900 border-gray-700">
      <CardHeader>
        <CardTitle class="text-white">System Logs</CardTitle>
      </CardHeader>
      <CardContent>
        <div class="bg-black p-4 rounded-lg h-64 overflow-y-auto font-mono text-xs">
          {#each logs as log}
            <div class="text-green-400 mb-1">{log}</div>
          {/each}
          {#if logs.length === 0}
            <div class="text-gray-500">No logs yet...</div>
          {/if}
        </div>
      </CardContent>
    </Card>
    
    <!-- Error Display -->
    {#if error}
      <Card class="bg-red-900 border-red-700">
        <CardHeader>
          <CardTitle class="text-red-100">Error</CardTitle>
        </CardHeader>
        <CardContent>
          <pre class="text-red-200 text-sm">{JSON.stringify(error, null, 2)}</pre>
        </CardContent>
      </Card>
    {/if}
    
    <!-- Instructions -->
    <Card class="bg-gray-900 border-gray-700">
      <CardHeader>
        <CardTitle class="text-white">How to Test</CardTitle>
      </CardHeader>
      <CardContent class="text-gray-300 space-y-2">
        <ol class="list-decimal list-inside space-y-1 text-sm">
          <li>Select one or more PDF files using the file input above</li>
          <li>Click "Start Processing" to begin the evidence processing pipeline</li>
          <li>Watch the real-time progress updates via WebSocket</li>
          <li>Monitor the XState machine state transitions</li>
          <li>Check system logs for detailed processing information</li>
          <li>Use the batch upload script to test multiple files: <code class="bg-gray-800 px-2 py-1 rounded">node scripts/batch-upload-lawpdfs.js</code></li>
        </ol>
        
        <div class="mt-4 p-3 bg-blue-900 rounded text-blue-100 text-sm">
          <strong>Note:</strong> This demo connects to the real evidence processing pipeline. 
          Make sure your services are running (RabbitMQ, PostgreSQL, Workers) before testing.
        </div>
      </CardContent>
    </Card>
  </div>
</div>

<style>
  :global(body) {
    background: #111827;
    color: #fff;
  }
</style>