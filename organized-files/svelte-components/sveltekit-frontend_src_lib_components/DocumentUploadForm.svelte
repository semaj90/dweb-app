<script lang="ts">
  interface Props {
    formData: {
  }
  let {
    formData
  } = $props();



  import { createEventDispatcher } from 'svelte';
  import { Button } from 'bits-ui';
  import { fade, slide } from 'svelte/transition';
  import { writable } from 'svelte/store';
  import type { OCRResult } from '$lib/services/ocr-processor';

  const dispatch = createEventDispatcher();

      uploaded_files: File[];
    ocr_results: OCRResult[];
    processing_status: 'pending' | 'processing' | 'completed' | 'error';
  };

  let dragActive = false;
  let fileInput: HTMLInputElement
  let uploadProgress = writable<Record<string, number>>({});
  let processingErrors = writable<Record<string, string>>({});

  // Accepted file types
  const acceptedTypes = [
    'application/pdf',
    'image/jpeg',
    'image/jpg',
    'image/png',
    'image/tiff',
    'image/bmp',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
  ];

  function isValidFileType(file: File): boolean {
    return acceptedTypes.includes(file.type);
  }

  function formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  function handleDragOver(event: DragEvent) {
    event.preventDefault();
    dragActive = true;
  }

  function handleDragLeave(event: DragEvent) {
    event.preventDefault();
    if (!(event.currentTarget as Element)?.contains(event.relatedTarget as Node)) {
      dragActive = false;
    }
  }

  function handleDrop(event: DragEvent) {
    event.preventDefault();
    dragActive = false;

    const files = Array.from(event.dataTransfer?.files || []);
    handleFileSelection(files);
  }

  function handleFileInputChange(event: Event) {
    const input = event.target as HTMLInputElement;
    const files = Array.from(input.files || []);
    handleFileSelection(files);
  }

  async function handleFileSelection(files: File[]) {
    const validFiles = files.filter(file => {
      if (!isValidFileType(file)) {
        processingErrors.update(errors => ({
          ...errors,
          [file.name]: `Unsupported file type: ${file.type}`
        }));
        return false;
      }
      if (file.size > 50 * 1024 * 1024) { // 50MB limit
        processingErrors.update(errors => ({
          ...errors,
          [file.name]: 'File size exceeds 50MB limit'
        }));
        return false;
      }
      return true;
    });

    formData.uploaded_files = [...formData.uploaded_files, ...validFiles];

    // Process files immediately if they're documents
    for (const file of validFiles) {
      if (file.type === 'application/pdf' || file.type.startsWith('image/')) {
        await processFile(file);
      }
    }
  }

  async function processFile(file: File) {
    formData.processing_status = 'processing';

    try {
      uploadProgress.update(progress => ({ ...progress, [file.name]: 0 }));

      // Simulate progress updates
      const progressInterval = setInterval(() => {
        uploadProgress.update(progress => {
          const current = progress[file.name] || 0;
          if (current < 90) {
            return { ...progress, [file.name]: current + 10 };
          }
          return progress;
        });
      }, 200);

      // Create mock OCR result for File object (browser environment)
      const ocrResult: OCRResult = {
        text: `Mock OCR text for ${file.name}`,
        confidence: 0.95,
        pages: [],
        metadata: {
          title: file.name,
          creation_date: new Date(),
          page_count: 1,
          file_size: file.size,
          content_type: file.type
        },
        processing_time: 100
      };

      clearInterval(progressInterval);
      uploadProgress.update(progress => ({ ...progress, [file.name]: 100 }));

      formData.ocr_results = [...formData.ocr_results, ocrResult];

      processingErrors.update(errors => {
        const newErrors = { ...errors };
        delete newErrors[file.name];
        return newErrors;
      });

    } catch (error) {
      console.error('OCR processing failed:', error);
      processingErrors.update(errors => ({
        ...errors,
        [file.name]: `Processing failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      }));
    }

    // Check if all files are processed
    const processedCount = formData.ocr_results.length;
    const totalDocuments = formData.uploaded_files.filter(f =>
      f.type === 'application/pdf' || f.type.startsWith('image/')
    ).length;

    if (processedCount === totalDocuments) {
      formData.processing_status = 'completed';
    }
  }

  function removeFile(index: number) {
    const removedFile = formData.uploaded_files[index];
    formData.uploaded_files = formData.uploaded_files.filter((_, i) => i !== index);

    // Remove corresponding OCR result
    formData.ocr_results = formData.ocr_results.filter(result =>
      result.metadata.title !== removedFile.name
    );

    // Clear any errors for this file
    processingErrors.update(errors => {
      const newErrors = { ...errors };
      delete newErrors[removedFile.name];
      return newErrors;
    });
  }

  function handleNext() {
    if (formData.uploaded_files.length === 0) {
      alert('Please upload at least one document before proceeding.');
      return;
    }

    dispatch('next', { step: 'documents', data: formData });
  }

  function handlePrevious() {
    dispatch('previous', { step: 'documents' });
  }

  function handleSaveDraft() {
    dispatch('saveDraft', { step: 'documents', data: formData });
  }

  function getFileIcon(fileType: string): string {
    if (fileType === 'application/pdf') return '📄';
    if (fileType.startsWith('image/')) return '🖼️';
    if (fileType.includes('word')) return '📝';
    return '📎';
  }

  function getProcessingStatusColor(status: string): string {
    switch (status) {
      case 'pending': return 'text-gray-500';
      case 'processing': return 'text-blue-500';
      case 'completed': return 'text-green-500';
      case 'error': return 'text-red-500';
      default: return 'text-gray-500';
    }
  }
</script>

<div class="max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-lg" transition:fade>
  <div class="mb-8">
    <h2 class="text-2xl font-bold text-gray-900 mb-2">Document Upload</h2>
    <p class="text-gray-600">Upload legal documents, contracts, evidence, and other case materials</p>
  </div>

  <!-- File Drop Zone -->
  <div
    class="border-2 border-dashed rounded-lg p-8 text-center transition-colors duration-200"
    class:border-blue-400={dragActive}
    class:bg-blue-50={dragActive}
    class:border-gray-300={!dragActive}
    class:bg-gray-50={!dragActive}
    ondragover={handleDragOver}
    ondragleave={handleDragLeave}
    ondrop={handleDrop}
    role="button"
    tabindex="0"
    onclick={() => fileInput.click()}
    onkeydown={(e) => e.key === 'Enter' && fileInput.click()}
  >
    <div class="space-y-4">
      <div class="text-4xl">📁</div>
      <div>
        <p class="text-lg font-medium text-gray-700">
          Drag and drop files here, or <span class="text-blue-600 underline">browse</span>
        </p>
        <p class="text-sm text-gray-500 mt-2">
          Supports: PDF, DOCX, DOC, JPEG, PNG, TIFF, BMP (max 50MB each)
        </p>
      </div>
    </div>
  </div>

  <input
    bind:this={fileInput}
    type="file"
    multiple
    accept=".pdf,.doc,.docx,.jpg,.jpeg,.png,.tiff,.bmp"
    onchange={handleFileInputChange}
    class="hidden"
  />

  <!-- Uploaded Files List -->
  {#if formData.uploaded_files.length > 0}
    <div class="mt-8" transition:slide>
      <h3 class="text-lg font-medium text-gray-900 mb-4">
        Uploaded Files ({formData.uploaded_files.length})
      </h3>

      <div class="space-y-3">
        {#each formData.uploaded_files as file, index}
          {@const progress = $uploadProgress[file.name] || 0}
          {@const error = $processingErrors[file.name]}

          <div class="border border-gray-200 rounded-lg p-4" transition:fade>
            <div class="flex items-center justify-between">
              <div class="flex items-center space-x-3 flex-1">
                <span class="text-2xl">{getFileIcon(file.type)}</span>
                <div class="flex-1 min-w-0">
                  <p class="text-sm font-medium text-gray-900 truncate">{file.name}</p>
                  <p class="text-sm text-gray-500">{formatFileSize(file.size)}</p>
                </div>
              </div>

              <div class="flex items-center space-x-3">
                <!-- Processing Status -->
                {#if file.type === 'application/pdf' || file.type.startsWith('image/')}
                  <div class="text-sm {getProcessingStatusColor(formData.processing_status)}">
                    {#if formData.processing_status === 'processing'}
                      Processing...
                    {:else if formData.processing_status === 'completed'}
                      ✅ Processed
                    {:else if error}
                      ❌ Error
                    {:else}
                      ⏳ Pending
                    {/if}
                  </div>
                {/if}

                <Button.Root
                  onclick={() => removeFile(index)}
                  class="p-1 text-red-600 hover:text-red-800 focus:outline-none"
                >
                  🗑️
                </Button.Root>
              </div>
            </div>

            <!-- Progress Bar -->
            {#if progress > 0 && progress < 100}
              <div class="mt-3">
                <div class="bg-gray-200 rounded-full h-2">
                  <div
                    class="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style="width: {progress}%"
                  ></div>
                </div>
                <p class="text-xs text-gray-500 mt-1">{progress}% complete</p>
              </div>
            {/if}

            <!-- Error Message -->
            {#if error}
              <div class="mt-3 text-sm text-red-600 bg-red-50 p-2 rounded">
                {error}
              </div>
            {/if}
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- OCR Results Summary -->
  {#if formData.ocr_results.length > 0}
    <div class="mt-8" transition:slide>
      <h3 class="text-lg font-medium text-gray-900 mb-4">
        OCR Processing Results ({formData.ocr_results.length})
      </h3>

      <div class="bg-green-50 border border-green-200 rounded-lg p-4">
        <div class="flex items-center">
          <span class="text-green-600 text-xl mr-3">✅</span>
          <div>
            <p class="text-sm font-medium text-green-800">
              Successfully processed {formData.ocr_results.length} document{formData.ocr_results.length !== 1 ? 's' : ''}
            </p>
            <p class="text-xs text-green-600 mt-1">
              Text extraction completed with an average confidence of {
                Math.round(formData.ocr_results.reduce((acc, result) => acc + result.confidence, 0) / formData.ocr_results.length)
              }%
            </p>
          </div>
        </div>
      </div>
    </div>
  {/if}

  <!-- Processing Status Indicator -->
  {#if formData.processing_status === 'processing'}
    <div class="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4" transition:fade>
      <div class="flex items-center">
        <div class="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600 mr-3"></div>
        <div>
          <p class="text-sm font-medium text-blue-800">Processing documents...</p>
          <p class="text-xs text-blue-600">Please wait while we extract text and analyze your documents</p>
        </div>
      </div>
    </div>
  {/if}

  <!-- Error Summary -->
  {#if Object.keys($processingErrors).length > 0}
    <div class="mt-6 bg-red-50 border border-red-200 rounded-lg p-4" transition:fade>
      <h4 class="text-sm font-medium text-red-800 mb-2">Processing Errors:</h4>
      <ul class="text-xs text-red-600 space-y-1">
        {#each Object.entries($processingErrors) as [filename, error]}
          <li>• {filename}: {error}</li>
        {/each}
      </ul>
    </div>
  {/if}

  <!-- Form Actions -->
  <div class="flex justify-between pt-6 mt-8 border-t border-gray-200">
    <Button.Root
      onclick={handlePrevious}
      class="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
    >
      ← Previous
    </Button.Root>

    <div class="flex space-x-3">
      <Button.Root
        onclick={handleSaveDraft}
        class="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
      >
        Save Draft
      </Button.Root>

      <Button.Root
        onclick={handleNext}
        disabled={formData.uploaded_files.length === 0 || formData.processing_status === 'processing'}
        class="px-6 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        Next: Evidence Analysis →
      </Button.Root>
    </div>
  </div>
</div>
