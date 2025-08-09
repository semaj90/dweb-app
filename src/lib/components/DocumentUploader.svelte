<!-- @ts-nocheck -->
<!-- @ts-nocheck -->
<!-- @ts-nocheck -->
  import { Button } from 'bits-ui';
  import { Progress } from 'bits-ui';
  import { Badge } from 'bits-ui';
  import { Card } from 'bits-ui';
  import { createEventDispatcher } from 'svelte';
  import { fly, scale } from 'svelte/transition';
  import { aiPipeline, type DocumentUpload, type ProcessingResult } from '$lib/ai/processing-pipeline.js';
  
  // Props
  let { 
    accept = '.pdf,.docx,.txt,.json',
    maxSize = 10 * 1024 * 1024, // 10MB
    multiple = true,
    disabled = false,
    class: className = ''
  } = $props();

  // State
  let isDragOver = $state(false);
  let isUploading = $state(false);
  let files = $state<File[]>([]);
  let uploadProgress = $state<Map<string, number>>(new Map());
  let processingResults = $state<Map<string, ProcessingResult>>(new Map());
  let errors = $state<string[]>([]);

  // Event dispatcher
  const dispatch = createEventDispatcher<{
    upload: { files: File[] };
    progress: { fileId: string; progress: number };
    complete: { fileId: string; result: ProcessingResult };
    error: { error: string };
  }>();

  // File input reference
  let fileInput: HTMLInputElement;

  /**
   * Handle drag and drop events
   */
  function handleDragOver(event: DragEvent) {
    event.preventDefault();
    event.stopPropagation();
    if (!disabled) {
      isDragOver = true;
    }
  }

  function handleDragLeave(event: DragEvent) {
    event.preventDefault();
    event.stopPropagation();
    isDragOver = false;
  }

  async function handleDrop(event: DragEvent) {
    event.preventDefault();
    event.stopPropagation();
    isDragOver = false;

    if (disabled) return;

    const droppedFiles = Array.from(event.dataTransfer?.files || []);
    await processFiles(droppedFiles);
  }

  /**
   * Handle file input change
   */
  async function handleFileInput(event: Event) {
    const input = event.target as HTMLInputElement;
    const selectedFiles = Array.from(input.files || []);
    await processFiles(selectedFiles);
  }

  /**
   * Process selected files
   */
  async function processFiles(newFiles: File[]) {
    // Validate files
    const validFiles = newFiles.filter(file => validateFile(file));
    
    if (validFiles.length === 0) {
      errors = [...errors, 'No valid files to process'];
      return;
    }

    // Add to files list
    files = multiple ? [...files, ...validFiles] : validFiles;
    isUploading = true;
    
    dispatch('upload', { files: validFiles });

    // Process each file
    for (const file of validFiles) {
      await uploadAndProcessFile(file);
    }

    isUploading = false;
  }

  /**
   * Validate individual file
   */
  function validateFile(file: File): boolean {
    // Check file size
    if (file.size > maxSize) {
      errors = [...errors, `File "${file.name}" exceeds maximum size of ${formatFileSize(maxSize)}`];
      return false;
    }

    // Check file type
    const acceptedTypes = accept.split(',').map(type => type.trim());
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
    
    if (!acceptedTypes.includes(fileExtension) && !acceptedTypes.includes(file.type)) {
      errors = [...errors, `File "${file.name}" is not an accepted file type`];
      return false;
    }

    return true;
  }

  /**
   * Upload and process individual file
   */
  async function uploadAndProcessFile(file: File) {
    const fileId = generateFileId(file);
    uploadProgress.set(fileId, 0);

    try {
      // Create document upload object
      const upload: DocumentUpload = {
        file,
        filename: file.name,
        mimeType: file.type || getMimeTypeFromExtension(file.name),
        metadata: {
          originalName: file.name,
          size: file.size,
          lastModified: file.lastModified
        }
      };

      // Start processing
      const result = await aiPipeline.processDocument(upload, {
        includeEmbeddings: true,
        includeSummary: true,
        includeEntities: true,
        includeRiskAnalysis: true,
        cacheResults: true,
        priority: 'medium'
      });

      // Monitor progress
      const progressInterval = setInterval(() => {
        const currentResult = aiPipeline.getProcessingStatus(result.id);
        if (currentResult) {
          const progress = getProgressFromStage(currentResult.metadata.stage);
          uploadProgress.set(fileId, progress);
          dispatch('progress', { fileId, progress });

          if (currentResult.status === 'completed' || currentResult.status === 'error') {
            clearInterval(progressInterval);
            uploadProgress.set(fileId, 100);
            processingResults.set(fileId, currentResult);
            dispatch('complete', { fileId, result: currentResult });
          }
        }
      }, 500);

      // Set final result
      processingResults.set(fileId, result);

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      errors = [...errors, `Failed to process "${file.name}": ${errorMessage}`];
      dispatch('error', { error: errorMessage });
    }
  }

  /**
   * Remove file from upload list
   */
  function removeFile(file: File) {
    files = files.filter(f => f !== file);
    const fileId = generateFileId(file);
    uploadProgress.delete(fileId);
    processingResults.delete(fileId);
  }

  /**
   * Clear all files and reset state
   */
  function clearAll() {
    files = [];
    uploadProgress.clear();
    processingResults.clear();
    errors = [];
    if (fileInput) {
      fileInput.value = '';
    }
  }

  /**
   * Trigger file input dialog
   */
  function openFileDialog() {
    if (!disabled && fileInput) {
      fileInput.click();
    }
  }

  // Helper functions
  function generateFileId(file: File): string {
    return `${file.name}_${file.size}_${file.lastModified}`;
  }

  function formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  function getMimeTypeFromExtension(filename: string): string {
    const extension = filename.split('.').pop()?.toLowerCase();
    const mimeTypes: Record<string, string> = {
      pdf: 'application/pdf',
      docx: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      txt: 'text/plain',
      json: 'application/json'
    };
    return mimeTypes[extension || ''] || 'application/octet-stream';
  }

  function getProgressFromStage(stage: string): number {
    const stageProgress: Record<string, number> = {
      initialization: 10,
      text_extraction: 25,
      document_analysis: 50,
      embedding_generation: 75,
      database_storage: 90,
      cache_storage: 95,
      completed: 100
    };
    return stageProgress[stage] || 0;
  }

  function getStatusColor(status: ProcessingResult['status']): string {
    switch (status) {
      case 'completed': return 'bg-green-500';
      case 'error': return 'bg-red-500';
      case 'processing': return 'bg-blue-500';
      default: return 'bg-gray-500';
    }
  }

  function dismissError(index: number) {
    errors = errors.filter((_, i) => i !== index);
  }
</script>

<div class={`document-uploader ${className}`}>
  <!-- Hidden file input -->
  <input
    bind:this={fileInput}
    type="file"
    {accept}
    {multiple}
    {disabled}
    class="hidden"
    on:change={handleFileInput}
  />

  <!-- Drop zone -->
  <div
    class={`
      drop-zone relative rounded-lg border-2 border-dashed p-8 text-center transition-all duration-200
      ${isDragOver ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}
      ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
    `}
    on:dragover={handleDragOver}
    on:dragleave={handleDragLeave}
    on:drop={handleDrop}
    on:click={openFileDialog}
    role="button"
    tabindex="0"
    on:keydown={(e) => e.key === 'Enter' && openFileDialog()}
  >
    {#if isDragOver}
      <div transition:scale class="text-blue-600">
        <svg class="mx-auto h-12 w-12 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
        </svg>
        <p class="text-lg font-semibold">Drop files here</p>
      </div>
    {:else}
      <div class="text-gray-500">
        <svg class="mx-auto h-12 w-12 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 12l2 2 4-4" />
        </svg>
        <p class="text-lg font-semibold mb-2">
          Drag and drop files here, or click to select
        </p>
        <p class="text-sm">
          Supports: {accept.replace(/\./g, '').toUpperCase()} • Max {formatFileSize(maxSize)}
        </p>
      </div>
    {/if}

    {#if isUploading}
      <div class="absolute inset-0 bg-white bg-opacity-75 flex items-center justify-center rounded-lg">
        <div class="text-center">
          <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
          <p class="text-sm text-gray-600">Processing files...</p>
        </div>
      </div>
    {/if}
  </div>

  <!-- Error messages -->
  {#if errors.length > 0}
    <div class="mt-4 space-y-2">
      {#each errors as error, index}
        <div transition:fly={{ y: -20 }} class="bg-red-50 border border-red-200 rounded-md p-3 flex items-center justify-between">
          <div class="flex items-center">
            <svg class="h-5 w-5 text-red-400 mr-2" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" />
            </svg>
            <span class="text-sm text-red-700">{error}</span>
          </div>
          <button
            type="button"
            class="text-red-400 hover:text-red-600"
            on:click={() => dismissError(index)}
          >
            <svg class="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd" />
            </svg>
          </button>
        </div>
      {/each}
    </div>
  {/if}

  <!-- File list -->
  {#if files.length > 0}
    <div class="mt-6 space-y-3">
      <div class="flex items-center justify-between">
        <h3 class="text-lg font-semibold text-gray-900">
          Uploaded Files ({files.length})
        </h3>
        <Button.Root variant="outline" size="sm" on:click={clearAll}>
          Clear All
        </Button.Root>
      </div>

      <div class="space-y-3">
        {#each files as file}
          {@const fileId = generateFileId(file)}
          {@const progress = uploadProgress.get(fileId) || 0}
          {@const result = processingResults.get(fileId)}
          
          <Card.Root class="p-4">
            <div class="flex items-center justify-between">
              <div class="flex items-center space-x-3 flex-1">
                <!-- File icon -->
                <div class="flex-shrink-0">
                  <svg class="h-8 w-8 text-gray-400" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" clip-rule="evenodd" />
                  </svg>
                </div>

                <!-- File info -->
                <div class="flex-1 min-w-0">
                  <p class="text-sm font-medium text-gray-900 truncate">
                    {file.name}
                  </p>
                  <p class="text-sm text-gray-500">
                    {formatFileSize(file.size)}
                    {#if result}
                      • Processed in {result.metadata.processingTime}ms
                    {/if}
                  </p>
                </div>

                <!-- Status badge -->
                {#if result}
                  <Badge.Root class={`${getStatusColor(result.status)} text-white`}>
                    {result.status}
                  </Badge.Root>
                {/if}
              </div>

              <!-- Remove button -->
              <Button.Root
                variant="ghost"
                size="sm"
                on:click={() => removeFile(file)}
                class="text-gray-400 hover:text-red-600"
              >
                <svg class="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
                  <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd" />
                </svg>
              </Button.Root>
            </div>

            <!-- Progress bar -->
            {#if progress > 0 && progress < 100}
              <div class="mt-3">
                <Progress.Root value={progress} max={100} class="h-2">
                  <Progress.Indicator class="bg-blue-600 h-full rounded-full transition-all duration-300" />
                </Progress.Root>
                <p class="text-xs text-gray-500 mt-1">
                  {progress}% complete
                  {#if result}
                    • {result.metadata.stage}
                  {/if}
                </p>
              </div>
            {/if}

            <!-- Processing result -->
            {#if result && result.status === 'completed' && result.result}
              <div class="mt-3 p-3 bg-green-50 rounded-md">
                <h4 class="text-sm font-medium text-green-800 mb-2">Analysis Complete</h4>
                <div class="text-sm text-green-700 space-y-1">
                  <p>Document ID: {result.result.documentId}</p>
                  <p>Type: {result.result.analysis.documentType || 'Unknown'}</p>
                  <p>Confidence: {Math.round((result.result.analysis.confidenceScore || 0) * 100)}%</p>
                  {#if result.result.analysis.summary}
                    <p class="mt-2 text-xs">
                      Summary: {result.result.analysis.summary.substring(0, 150)}...
                    </p>
                  {/if}
                </div>
              </div>
            {:else if result && result.status === 'error'}
              <div class="mt-3 p-3 bg-red-50 rounded-md">
                <h4 class="text-sm font-medium text-red-800 mb-1">Processing Failed</h4>
                <p class="text-sm text-red-700">{result.error}</p>
              </div>
            {/if}
          </Card.Root>
        {/each}
      </div>
    </div>
  {/if}
</div>

<style>
  .document-uploader {
    @apply max-w-4xl mx-auto;
  }

  .drop-zone {
    min-height: 200px;
    display: flex;
    align-items: center;
    justify-content: center;
  }
</style>