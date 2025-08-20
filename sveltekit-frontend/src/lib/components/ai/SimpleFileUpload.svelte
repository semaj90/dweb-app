<!-- Simple File Upload Test Component (No XState) -->
<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';

  import { toast } from '$lib/utils/toast';
  import { Upload, Check, X, Loader2 } from 'lucide-svelte';
  import { onMount } from 'svelte';

  // Props interface
  interface Props extends CommonProps {
    onUploadComplete?: (doc: any) => void;
    accept?: string;
    maxSize?: number;
    enableOCR?: boolean;
    enableEmbedding?: boolean;
    enableRAG?: boolean;
    class?: string; // note: renamed locally to avoid reserved identifier usage
  }

  // Svelte 5 props with defaults
  let {
    onUploadComplete = () => {},
    accept = '.pdf,.docx,.txt,.jpg,.png,.tiff',
    maxSize = 50 * 1024 * 1024, // 50MB
    enableOCR = true,
    enableEmbedding = true,
    enableRAG = true,
    class: classNameVar = '',
  }: Props = $props();

  // State variables
  let files = $state<File[]>([]);
  let uploadStates = $state<Map<string, any>>(new Map());
  let isDragOver = $state(false);
  let fileInput: HTMLInputElement;
  let systemStatus = $state<any>({});

  // Check system status on mount
  onMount(async () => {
    try {
      const response = await fetch('/api/rag/status');
      if (response.ok) {
        systemStatus = await response.json();
      }
    } catch (error) {
      console.error('Failed to fetch system status:', error);
    }
  });

  function handleDragOver(event: DragEvent) {
    event.preventDefault();
    isDragOver = true;
  }

  function handleDragLeave() {
    isDragOver = false;
  }

  function handleDrop(event: DragEvent) {
    event.preventDefault();
    isDragOver = false;

    const droppedFiles = Array.from(event.dataTransfer?.files || []);
    handleFiles(droppedFiles);
  }

  function handleFileInput(event: Event) {
    const input = event.target as HTMLInputElement;
    const selectedFiles = Array.from(input.files || []);
    handleFiles(selectedFiles);
  }

  function handleFiles(newFiles: File[]) {
    // Validate files
    const validFiles = newFiles.filter(file => {
      if (file.size > maxSize) {
        toast.error(`File ${file.name} is too large (max ${Math.round(maxSize / 1024 / 1024)}MB)`);
        return false;
      }
      return true;
    });

    files = [...files, ...validFiles];

    // Start upload for each file
    validFiles.forEach(file => {
      uploadFile(file);
    });
  }

  async function uploadFile(file: File) {
    const fileId = `${file.name}-${Date.now()}`;

    // Initialize upload state
    uploadStates.set(fileId, {
      status: 'uploading',
      progress: 0,
      fileName: file.name,
      fileSize: file.size
    });
    uploadStates = new Map(uploadStates);

    try {
      // Create form data
      const formData = new FormData();
      formData.append('files', file);
      formData.append('enableOCR', enableOCR.toString());
      formData.append('enableEmbedding', enableEmbedding.toString());
      formData.append('enableRAG', enableRAG.toString());

      console.log('Uploading file:', file.name, 'to /api/rag/process');

      // Upload to RAG API
      const response = await fetch('/api/rag/process', {
        method: 'POST',
        body: formData
      });

      const result = await response.json();

      if (response.ok) {
        // Update state to success
        uploadStates.set(fileId, {
          ...uploadStates.get(fileId),
          status: 'success',
          progress: 100,
          result
        });
        uploadStates = new Map(uploadStates);

        toast.success(`Successfully processed ${file.name}`);
        onUploadComplete(result);

        console.log('Upload result:', result);
      } else {
        throw new Error(result.error || 'Upload failed');
      }

    } catch (error) {
      console.error('Upload error:', error);

      // Update state to error
      uploadStates.set(fileId, {
        ...uploadStates.get(fileId),
        status: 'error',
        error: error.message
      });
      uploadStates = new Map(uploadStates);

      toast.error(`Failed to upload ${file.name}: ${error.message}`);
    }
  }

  function removeFile(index: number) {
    files = files.filter((_, i) => i !== index);
  }

  function formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }
</script>

<div class={`space-y-4 ${classNameVar}`}>
  <!-- System Status -->
  {#if systemStatus.services}
    <div class="mb-4 p-3 bg-gray-50 rounded-lg">
      <h3 class="text-sm font-medium mb-2">System Status</h3>
      <div class="flex flex-wrap gap-2">
        {#each Object.entries(systemStatus.services) as [service, status]}
          <span class="inline-flex items-center px-2 py-1 rounded-full text-xs {status.healthy ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}">
            {status.healthy ? '✓' : '✗'} {service}
          </span>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Upload Zone -->
  <div
    class="border-2 border-dashed rounded-lg p-8 text-center transition-colors {isDragOver ? 'border-blue-400 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}"
    ondragover={handleDragOver}
    ondragleave={handleDragLeave}
    ondrop={handleDrop}
    role="button"
    tabindex="0"
    onclick={() => fileInput?.click()}
    onkeydown={(e) => e.key === 'Enter' && fileInput?.click()}
  >
    <Upload class="w-12 h-12 text-gray-400 mx-auto mb-4" />
    <p class="text-lg font-medium text-gray-600 mb-2">
      Drop files here or click to upload
    </p>
    <p class="text-sm text-gray-500">
      Supports: {accept} (max {Math.round(maxSize / 1024 / 1024)}MB each)
    </p>

    <input
      bind:this={fileInput}
      type="file"
      multiple
      {accept}
      class="hidden"
      onchange={handleFileInput}
    />
  </div>

  <!-- Upload Progress -->
  {#if uploadStates.size > 0}
    <div class="space-y-3">
      <h3 class="text-lg font-medium">Upload Progress</h3>
      {#each Array.from(uploadStates.entries()) as [fileId, state]}
        <div class="border rounded-lg p-4">
          <div class="flex items-center justify-between mb-2">
            <span class="font-medium truncate">{state.fileName}</span>
            <span class="text-sm text-gray-500">{formatFileSize(state.fileSize)}</span>
          </div>

          <div class="flex items-center gap-2">
            {#if state.status === 'uploading'}
              <Loader2 class="w-4 h-4 animate-spin text-blue-600" />
              <span class="text-sm text-blue-600">Processing...</span>
            {:else if state.status === 'success'}
              <Check class="w-4 h-4 text-green-600" />
              <span class="text-sm text-green-600">Completed</span>
            {:else if state.status === 'error'}
              <X class="w-4 h-4 text-red-600" />
              <span class="text-sm text-red-600">Error: {state.error}</span>
            {/if}
          </div>

          {#if state.result}
            <div class="mt-2 p-2 bg-gray-50 rounded text-xs">
              <pre>{JSON.stringify(state.result, null, 2)}</pre>
            </div>
          {/if}
        </div>
      {/each}
    </div>
  {/if}

  <!-- Settings -->
  <div class="flex flex-wrap gap-4 text-sm">
    <label class="flex items-center gap-2">
      <input type="checkbox" bind:checked={enableOCR} />
      Enable OCR
    </label>
    <label class="flex items-center gap-2">
      <input type="checkbox" bind:checked={enableEmbedding} />
      Enable Embeddings
    </label>
    <label class="flex items-center gap-2">
      <input type="checkbox" bind:checked={enableRAG} />
      Enable RAG
    </label>
  </div>
</div>

<style>
  pre {
    max-height: 200px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-all;
  }
</style>

<!-- Provide a default export to satisfy default import usage -->
<script context="module" lang="ts">
  import Component from './SimpleFileUpload.svelte';
  export default Component;
</script>
