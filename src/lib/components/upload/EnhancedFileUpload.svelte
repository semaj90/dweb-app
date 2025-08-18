<!-- Enhanced File Upload with XState Machine and MCP Integration -->
<script lang="ts">
  import { onMount, onDestroy, createEventDispatcher } from 'svelte';
  import { writable } from 'svelte/store';
  import { Upload, FileText, CheckCircle, AlertTriangle, Loader2, Zap, Database } from 'lucide-svelte';
  import { createActor } from 'xstate';
  import { 
    documentUploadMachine, 
    getUploadMetrics,
    isUploading,
    getUploadProgress,
    getStateValue,
    isInErrorState,
    isInProcessingState
  } from '../../state/documentUploadMachine';
  import type { 
    DocumentUploadContext, 
    DocumentUploadStateValue 
  } from '../../state/documentUploadMachine';

  // Props (converted to Svelte 4 syntax)
  export let maxFiles: number = 10;
  export let maxSize: number = 100; // in MB
  export let acceptedTypes: string[] = ['.pdf', '.txt', '.docx', '.doc', '.json'];
  export let enableGPUAcceleration: boolean = true;
  export let enableRealTimeProcessing: boolean = true;
  export let className: string = '';
  export let caseId: string = '';
  export let userId: string = '';

  // Event dispatcher
  const dispatch = createEventDispatcher<{
    upload: { files: File[] };
    progress: { fileId: string, progress: number };
    complete: { fileId: string, result: any };
    error: { fileId: string, error: string };
  }>();

  // XState actors for each file upload
  let fileActors: Map<string, any> = new Map();
  let fileStates: Map<string, any> = new Map();
  
  // Component state (converted to Svelte 4)
  let fileInput: HTMLInputElement;
  let dragActive: boolean = false;
  let processingStats = {
    totalFiles: 0,
    processed: 0,
    errors: 0,
    avgProcessingTime: 0
  };

  // GPU Acceleration Worker
  let gpuWorker: Worker | null = null;
  let webGPUDevice: GPUDevice | null = null;
  
  // WebSocket for real-time updates
  let websocket: WebSocket | null = null;
  let connectionStatus: 'disconnected' | 'connecting' | 'connected' | 'error' = 'disconnected';

  // Enhanced RAG Pipeline Configuration (MCP-powered)
  const processingPipeline = {
    stages: [
      { name: 'Upload', enabled: true, mcpEndpoint: 'http://localhost:8093/upload', port: 8093 },
      { name: 'Text Extraction', enabled: true, mcpEndpoint: 'http://localhost:8093/extract-text', port: 8093 },
      { name: 'Chunking', enabled: true, mcpEndpoint: 'http://localhost:8094/process', port: 8094 },
      { name: 'Embedding Generation', enabled: true, mcpEndpoint: 'http://localhost:8094/embeddings', port: 8094 },
      { name: 'Vector Storage', enabled: true, mcpEndpoint: 'http://localhost:8094/store', port: 8094 },
      { name: 'Semantic Analysis', enabled: true, mcpEndpoint: 'http://localhost:8094/analyze', port: 8094 }
    ],
    protocol: 'HTTP/WebSocket',
    uploadService: 'http://localhost:8093',
    ragService: 'http://localhost:8094'
  };

  onMount(async () => {
    await initializeGPUAcceleration();
    setupEventListeners();
    await initializeWebSocket();
  });

  onDestroy(() => {
    // Clean up WebSocket connection
    if (websocket) {
      websocket.close();
      websocket = null;
    }
    
    // Stop all file actors
    fileActors.forEach(actor => {
      try {
        actor.stop();
      } catch (error) {
        console.warn('Error stopping actor:', error);
      }
    });
    
    fileActors.clear();
    fileStates.clear();
  });

  async function initializeGPUAcceleration() {
    if (!enableGPUAcceleration) return;

    try {
      // Initialize WebGPU
      if ('gpu' in navigator) {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
          webGPUDevice = await adapter.requestDevice();
          console.log('WebGPU initialized for file processing acceleration');
        }
      }

      // Initialize GPU Worker
      gpuWorker = new Worker('/workers/gpu-file-processor.js');
      gpuWorker.onmessage = handleWorkerMessage;
      
      if (webGPUDevice) {
        gpuWorker.postMessage({
          type: 'INIT_GPU',
          device: webGPUDevice
        });
      }
    } catch (error) {
      console.warn('GPU acceleration initialization failed:', error);
      // Fallback to CPU processing
    }
  }

  async function initializeWebSocket() {
    if (!enableRealTimeProcessing) return;
    
    try {
      connectionStatus = 'connecting';
      const wsUrl = `ws://localhost:8094/ws?userId=${userId}&type=upload_status`;
      websocket = new WebSocket(wsUrl);
      
      websocket.onopen = () => {
        connectionStatus = 'connected';
        console.log('WebSocket connected for real-time upload status');
      };
      
      websocket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          handleWebSocketMessage(message);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };
      
      websocket.onclose = () => {
        connectionStatus = 'disconnected';
        console.log('WebSocket disconnected');
        // Attempt reconnection after 3 seconds
        setTimeout(initializeWebSocket, 3000);
      };
      
      websocket.onerror = (error) => {
        connectionStatus = 'error';
        console.error('WebSocket error:', error);
      };
    } catch (error) {
      connectionStatus = 'error';
      console.error('Failed to initialize WebSocket:', error);
    }
  }

  function handleWebSocketMessage(message: any) {
    const { type, fileId, data } = message;
    
    switch (type) {
      case 'upload_progress':
        updateFileProgress(fileId, data.progress, data.stage);
        break;
      case 'upload_complete':
        handleUploadComplete(fileId, data);
        break;
      case 'upload_error':
        handleUploadError(fileId, data.error);
        break;
      case 'processing_stage':
        updateProcessingStage(fileId, data.stage, data.progress);
        break;
      default:
        console.log('Unknown WebSocket message type:', type);
    }
  }

  function updateFileProgress(fileId: string, progress: number, stage?: string) {
    const actor = getFileActor(fileId);
    if (actor) {
      actor.send({
        type: 'UPDATE_PROGRESS',
        progress,
        stage
      });
    }
  }

  function handleUploadComplete(fileId: string, data: any) {
    const actor = getFileActor(fileId);
    if (actor) {
      actor.send({
        type: 'UPLOAD_COMPLETE',
        result: data
      });
    }
    dispatch('complete', { fileId, result: data });
  }

  function handleUploadError(fileId: string, error: string) {
    const actor = getFileActor(fileId);
    if (actor) {
      actor.send({
        type: 'UPLOAD_ERROR',
        error
      });
    }
    dispatch('error', { fileId, error });
  }

  function updateProcessingStage(fileId: string, stage: string, progress: number) {
    const actor = getFileActor(fileId);
    if (actor) {
      actor.send({
        type: 'UPDATE_STAGE',
        stage,
        progress
      });
    }
  }

  function setupEventListeners() {
    // Drag and drop
    document.addEventListener('dragover', handleDragOver);
    document.addEventListener('drop', handleDrop);
    document.addEventListener('dragleave', handleDragLeave);
    
    // Cleanup WebSocket on component destroy
    return () => {
      if (websocket) {
        websocket.close();
        websocket = null;
      }
    };
  }

  function handleDragOver(e: DragEvent) {
    e.preventDefault();
    dragActive = true;
  }

  function handleDragLeave(e: DragEvent) {
    e.preventDefault();
    dragActive = false;
  }

  function handleDrop(e: DragEvent) {
    e.preventDefault();
    dragActive = false;
    
    const files = Array.from(e.dataTransfer?.files || []);
    if (files.length > 0) {
      processFiles(files);
    }
  }

  function handleFileSelect(e: Event) {
    const target = e.target as HTMLInputElement;
    const files = Array.from(target.files || []);
    if (files.length > 0) {
      processFiles(files);
    }
  }

  async function processFiles(files: File[]) {
    // Validate files
    const validFiles = files.filter(validateFile);
    if (validFiles.length === 0) return;

    // Limit files
    const filesToProcess = validFiles.slice(0, maxFiles);
    
    // Create XState actors for each file
    filesToProcess.forEach(file => {
      const fileId = `${file.name}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      // Create actor instance
      const actor = createActor(documentUploadMachine).start();
      
      // Subscribe to state changes
      actor.subscribe(state => {
        fileStates.set(fileId, state);
        updateProcessingStats();
        
        // Dispatch events based on state
        if (state.matches('uploading') || state.matches('processing')) {
          dispatch('progress', { 
            fileId, 
            progress: state.context.uploadProgress || 0 
          });
        } else if (state.matches('completed')) {
          dispatch('complete', { fileId, result: state.context });
          processingStats.processed++;
        } else if (state.matches('uploadError') || state.matches('uploadFailed')) {
          dispatch('error', { fileId, error: state.context.error || 'Upload failed' });
          processingStats.errors++;
        }
        
        // Trigger Svelte reactivity
        fileStates = new Map(fileStates);
      });
      
      // Store actor
      fileActors.set(fileId, actor);
      
      // Start upload process
      actor.send({
        type: 'SELECT_FILE',
        file,
        caseId,
        userId,
        title: file.name,
        description: `Uploaded via enhanced file upload component`,
        tags: ['uploaded', 'document']
      });
      
      // Notify WebSocket server about new upload
      if (websocket && connectionStatus === 'connected') {
        websocket.send(JSON.stringify({
          type: 'start_upload',
          fileId,
          fileName: file.name,
          fileSize: file.size,
          userId,
          caseId
        }));
      }
    });

    processingStats.totalFiles += filesToProcess.length;
    dispatch('upload', { files: filesToProcess });
  }

  function validateFile(file: File): boolean {
    // Check file size
    if (file.size > maxSize * 1024 * 1024) {
      console.error(`File ${file.name} exceeds maximum size of ${maxSize}MB`);
      return false;
    }

    // Check file type
    const extension = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!acceptedTypes.includes(extension)) {
      console.error(`File ${file.name} type not supported`);
      return false;
    }

    return true;
  }

  function updateProcessingStats() {
    const states = Array.from(fileStates.values());
    const completed = states.filter(s => s.matches('completed'));
    const errors = states.filter(s => s.matches('uploadError') || s.matches('uploadFailed'));
    
    if (completed.length > 0) {
      const totalTime = completed.reduce((sum, state) => {
        const metrics = getUploadMetrics(state);
        return sum + metrics.totalTime;
      }, 0);
      processingStats.avgProcessingTime = Math.round(totalTime / completed.length);
    }
    
    processingStats.processed = completed.length;
    processingStats.errors = errors.length;
  }

  // Helper functions for XState integration
  function getFileState(fileId: string) {
    return fileStates.get(fileId);
  }

  function getFileActor(fileId: string) {
    return fileActors.get(fileId);
  }

  function retryUpload(fileId: string) {
    const actor = getFileActor(fileId);
    if (actor) {
      actor.send({ type: 'RETRY_UPLOAD' });
    }
  }

  function cancelUpload(fileId: string) {
    const actor = getFileActor(fileId);
    if (actor) {
      actor.send({ type: 'CANCEL_UPLOAD' });
    }
  }

  function removeFile(fileId: string) {
    const actor = getFileActor(fileId);
    if (actor) {
      actor.stop();
      fileActors.delete(fileId);
      fileStates.delete(fileId);
      fileStates = new Map(fileStates); // Trigger reactivity
      updateProcessingStats();
    }
  }

  function clearCompleted() {
    const completedFileIds = Array.from(fileStates.entries())
      .filter(([_, state]) => state.matches('completed'))
      .map(([fileId, _]) => fileId);
    
    completedFileIds.forEach(fileId => removeFile(fileId));
  }

  function getStatusIcon(status: string) {
    switch (status) {
      case 'completed': return CheckCircle;
      case 'error': return AlertTriangle;
      case 'processing': return Loader2;
      default: return FileText;
    }
  }

  function getStatusColor(status: string) {
    switch (status) {
      case 'completed': return 'text-green-500';
      case 'error': return 'text-red-500';
      case 'processing': return 'text-blue-500';
      default: return 'text-gray-500';
    }
  }
</script>

<div class="enhanced-file-upload {className}">
  <!-- Upload Zone -->
  <div 
    class="upload-zone border-2 border-dashed rounded-xl p-8 text-center transition-all duration-300 {
      dragActive 
        ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' 
        : 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500'
    }"
    on:click={() => fileInput?.click()}
    role="button"
    tabindex="0"
    on:keydown={(e) => e.key === 'Enter' && fileInput?.click()}
  >
    <input
      bind:this={fileInput}
      type="file"
      multiple
      accept={acceptedTypes.join(',')}
      on:change={handleFileSelect}
      class="hidden"
    />

    <div class="flex flex-col items-center space-y-4">
      <div class="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
        <Upload class="w-8 h-8 text-white" />
      </div>
      
      <div>
        <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
          Upload Legal Documents
        </h3>
        <p class="text-gray-600 dark:text-gray-400 mb-4">
          Drag and drop files or click to browse
        </p>
        <p class="text-sm text-gray-500 dark:text-gray-500">
          Supports: {acceptedTypes.join(', ')} • Max {maxSize}MB per file
        </p>
      </div>

      <div class="flex items-center space-x-3">
        {#if enableGPUAcceleration}
          <div class="flex items-center space-x-2 px-3 py-1 bg-gradient-to-r from-purple-100 to-blue-100 dark:from-purple-900/20 dark:to-blue-900/20 rounded-full">
            <Zap class="w-4 h-4 text-purple-600 dark:text-purple-400" />
            <span class="text-sm font-medium text-purple-700 dark:text-purple-300">GPU Accelerated</span>
          </div>
        {/if}
        
        {#if enableRealTimeProcessing}
          <div class="flex items-center space-x-2 px-3 py-1 bg-gradient-to-r from-green-100 to-emerald-100 dark:from-green-900/20 dark:to-emerald-900/20 rounded-full">
            <div class="w-2 h-2 rounded-full {
              connectionStatus === 'connected' ? 'bg-green-500' :
              connectionStatus === 'connecting' ? 'bg-yellow-500 animate-pulse' :
              connectionStatus === 'error' ? 'bg-red-500' : 'bg-gray-400'
            }"></div>
            <span class="text-sm font-medium text-green-700 dark:text-green-300">
              Real-time {connectionStatus === 'connected' ? 'Connected' : 
                        connectionStatus === 'connecting' ? 'Connecting...' :
                        connectionStatus === 'error' ? 'Error' : 'Disconnected'}
            </span>
          </div>
        {/if}
      </div>
    </div>
  </div>

  <!-- Processing Pipeline Status -->
  {#if fileStates.size > 0}
    <div class="mt-6 space-y-4">
      <!-- Pipeline Overview -->
      <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
        <div class="flex items-center justify-between mb-4">
          <h4 class="text-lg font-semibold text-gray-900 dark:text-gray-100">Processing Pipeline</h4>
          <div class="flex items-center space-x-2">
            <Database class="w-4 h-4 text-blue-600 dark:text-blue-400" />
            <span class="text-sm font-medium text-blue-700 dark:text-blue-300">
              {processingPipeline.protocol} Protocol
            </span>
          </div>
        </div>

        <div class="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
          {#each processingPipeline.stages.filter(s => s.enabled) as stage}
            <div class="flex items-center space-x-2 p-2 bg-gray-50 dark:bg-gray-700 rounded">
              <Database class="w-3 h-3 text-blue-500" />
              <span class="text-gray-700 dark:text-gray-300">{stage.name}</span>
            </div>
          {/each}
        </div>
      </div>

      <!-- Processing Stats -->
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <div class="text-2xl font-bold text-blue-600 dark:text-blue-400">{processingStats.totalFiles}</div>
          <div class="text-sm text-gray-600 dark:text-gray-400">Total Files</div>
        </div>
        <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <div class="text-2xl font-bold text-green-600 dark:text-green-400">{processingStats.processed}</div>
          <div class="text-sm text-gray-600 dark:text-gray-400">Processed</div>
        </div>
        <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <div class="text-2xl font-bold text-red-600 dark:text-red-400">{processingStats.errors}</div>
          <div class="text-sm text-gray-600 dark:text-gray-400">Errors</div>
        </div>
        <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <div class="text-2xl font-bold text-purple-600 dark:text-purple-400">{processingStats.avgProcessingTime}ms</div>
          <div class="text-sm text-gray-600 dark:text-gray-400">Avg Time</div>
        </div>
      </div>

      <!-- File List -->
      <div class="space-y-3">
        {#each Array.from(fileStates.entries()) as [fileId, state] (fileId)}
          {@const context = state.context as DocumentUploadContext}
          {@const currentState = getStateValue(state)}
          {@const progress = getUploadProgress(state)}
          {@const isProcessing = isInProcessingState(state)}
          <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
            <div class="flex items-center justify-between mb-3">
              <div class="flex items-center space-x-3">
                {@const StatusIcon = getStatusIcon(String(currentState))}
                <StatusIcon class="w-5 h-5 {getStatusColor(String(currentState))}" />
                <div>
                  <p class="font-medium text-gray-900 dark:text-gray-100">{context.filename}</p>
                  <p class="text-sm text-gray-600 dark:text-gray-400">
                    {(context.fileSize / 1024 / 1024).toFixed(2)} MB • {String(currentState)}
                  </p>
                </div>
              </div>
              
              <div class="flex items-center space-x-2">
                {#if isProcessing}
                  <div class="text-sm font-medium text-blue-600 dark:text-blue-400">
                    {Math.round(progress)}%
                  </div>
                {:else if state.matches('completed')}
                  {@const metrics = getUploadMetrics(state)}
                  <div class="text-sm text-gray-600 dark:text-gray-400">
                    {metrics.totalTime}ms
                  </div>
                {/if}
                
                <button
                  on:click={() => removeFile(fileId)}
                  class="text-gray-400 hover:text-red-500 transition-colors"
                  aria-label="Remove file"
                >
                  ×
                </button>
              </div>
            </div>

            <!-- Progress Bar -->
            {#if isProcessing}
              <div class="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mb-3">
                <div 
                  class="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style="width: {progress}%"
                ></div>
              </div>
            {/if}

            <!-- Stage Progress (XState-based) -->
            {#if isProcessing}
              <div class="grid grid-cols-2 md:grid-cols-3 gap-2 text-xs">
                {#each processingPipeline.stages.filter(s => s.enabled) as stage}
                  <div class="flex items-center space-x-1 p-1">
                    {#if state.matches('completed')}
                      <CheckCircle class="w-3 h-3 text-green-500" />
                    {:else if isProcessing}
                      <Loader2 class="w-3 h-3 text-blue-500 animate-spin" />
                    {:else}
                      <div class="w-3 h-3 rounded-full bg-gray-300 dark:bg-gray-600"></div>
                    {/if}
                    <span class="text-gray-600 dark:text-gray-400">{stage.name}</span>
                  </div>
                {/each}
              </div>
            {/if}

            <!-- Error Display -->
            {#if isInErrorState(state) && context.error}
              <div class="mt-3 p-2 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded text-sm text-red-700 dark:text-red-300">
                {context.error}
              </div>
            {/if}

            <!-- Validation Errors -->
            {#if context.validationErrors && context.validationErrors.length > 0}
              <div class="mt-3 p-2 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded text-sm text-yellow-700 dark:text-yellow-300">
                {#each context.validationErrors as error}
                  <div>{error}</div>
                {/each}
              </div>
            {/if}
          </div>
        {/each}
      </div>

      <!-- Actions -->
      <div class="flex justify-between items-center">
        <button
          on:click={clearCompleted}
          class="px-4 py-2 text-sm font-medium text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 transition-colors"
        >
          Clear Completed
        </button>
        
        {@const hasProcessingFiles = Array.from(fileStates.values()).some(state => isUploading(state))}
        {#if hasProcessingFiles}
          <div class="flex items-center space-x-2 text-sm text-blue-600 dark:text-blue-400">
            <Loader2 class="w-4 h-4 animate-spin" />
            <span>Processing files...</span>
          </div>
        {/if}
      </div>
    </div>
  {/if}
</div>

<style>
  .upload-zone {
    cursor: pointer;
    user-select: none;
  }

  .upload-zone:hover {
    background-color: rgba(59, 130, 246, 0.05);
  }

  @media (prefers-reduced-motion: reduce) {
    .upload-zone,
    .bg-blue-600,
    * {
      transition: none !important;
      animation: none !important;
    }
  }
</style>