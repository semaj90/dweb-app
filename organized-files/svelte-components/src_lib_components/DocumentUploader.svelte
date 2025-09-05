<script lang="ts">
  import { Button } from 'bits-ui';
  import { Progress } from 'bits-ui';
  import { Badge } from 'bits-ui';
  import { Card } from 'bits-ui';
  import { createEventDispatcher } from 'svelte';
  import { fly, scale } from 'svelte/transition';
  import { aiPipeline, type DocumentUpload, type ProcessingResult } from '$lib/ai/processing-pipeline';

  // Props
  let {
    accept = '.pdf,.docx,.txt,.json',
    maxSize = 10 * 1024 * 1024, // 10MB
    multiple = true,
    disabled = false,
    export let accept: string = '.pdf,.docx,.txt,.json';
    export let maxSize: number = 10 * 1024 * 1024; // 10MB
    export let multiple: boolean = true;
    export let disabled: boolean = false;
    export let className: string = '';
  let uploadProgress = $state(new Map<string, number>());
  let processingResults = $state(new Map<string, ProcessingResult>());
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




  <input
    bind:this={fileInput}
    type="file"
    {accept}
    {multiple}
    {disabled}
    class="hidden"
    on:change={handleFileInput}
  />


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
  tabindex={disabled ? -1 : 0}
  aria-disabled={disabled}
  aria-label="Document upload area. Press Enter or Space to choose files, or drag and drop."
  on:keydown={(e) => (e.key === 'Enter' || e.key === ' ') && openFileDialog()}
  >
    {#if isDragOver}




        Drop files here

    {:else}





          Drag and drop files here, or click to select


          Supports: {accept.replace(/\./g, '').toUpperCase()} • Max {formatFileSize(maxSize)}


    {/if}

    {#if isUploading}



          Processing files...


    {/if}



  {#if errors.length > 0}

      {#each errors as error, index}





            {error}

          <button
            type="button"
            class="text-red-400 hover:text-red-600"
            on:click={() => dismissError(index)}
          >





      {/each}

  {/if}


  {#if files.length > 0}



          Uploaded Files ({files.length})


          Clear All




        {#each files as file}
          {@const fileId = generateFileId(file)}
          {@const progress = uploadProgress.get(fileId) || 0}
          {@const result = processingResults.get(fileId)}














                    {file.name}


                    {formatFileSize(file.size)}
                    {#if result}
                      • Processed in {result.metadata.processingTime}ms
                    {/if}




                {#if result}

                    {result.status}

                {/if}



              <Button.Root
                variant="ghost"
                size="sm"
                on:click={() => removeFile(file)}
                class="text-gray-400 hover:text-red-600"
              >







            {#if progress > 0 && progress < 100}





                  {progress}% complete
                  {#if result}
                    • {result.metadata.stage}
                  {/if}


            {/if}


            {#if result && result.status === 'completed' && result.result}

                Analysis Complete

                  Document ID: {result.result.documentId}
                  Type: {result.result.analysis.documentType || 'Unknown'}
                  Confidence: {Math.round((result.result.analysis.confidenceScore || 0) * 100)}%
                  {#if result.result.analysis.summary}

                      Summary: {result.result.analysis.summary.substring(0, 150)}...

                  {/if}


            {:else if result && result.status === 'error'}

                Processing Failed
                {result.error}

            {/if}

        {/each}


  {/if}



  <style>
  .document-uploader {
    /* Tailwind @apply cannot be used in component CSS without a preprocessor */
    max-width: 56rem; /* approx max-w-4xl */
    margin-left: auto;
    margin-right: auto;
  }

  .drop-zone {
    min-height: 200px;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  </style>
