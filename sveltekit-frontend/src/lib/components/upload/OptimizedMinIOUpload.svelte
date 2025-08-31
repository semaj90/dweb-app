<!-- Optimized MinIO Upload with HTML5 Drag-and-Drop and Redis Sync -->
<script lang="ts">
  import { $props, $state, $effect } from 'svelte';
  import { Upload, FileText, Image, CheckCircle, AlertCircle, Loader2 } from 'lucide-svelte';

  interface Props {
    caseId?: string;
    onUploadComplete?: (result: UploadResult) => void;
    onUploadError?: (error: string) => void;
    multiple?: boolean;
    disabled?: boolean;
    accept?: string;
    maxSize?: number;
  }

  let {
    caseId = '',
    onUploadComplete,
    onUploadError,
    multiple = false,
    disabled = false,
    accept = '.pdf,.doc,.docx,.txt,.jpg,.jpeg,.png,.tiff',
    maxSize = 100 * 1024 * 1024 // 100MB
  }: Props = $props();

  interface UploadResult {
    success: boolean;
    id: string;
    fileName: string;
    originalName: string;
    fileSize: number;
    url: string;
    hash: string;
    message: string;
  }

  // Upload state
  let files = $state<File[]>([]);
  let uploading = $state(false);
  let dragOver = $state(false);
  let uploadProgress = $state(0);
  let uploadStatus = $state<'idle' | 'uploading' | 'processing' | 'completed' | 'error'>('idle');
  let errorMessage = $state<string | null>(null);
  let fileInput: HTMLInputElement;

  // Drag and drop handlers
  function handleDragOver(event: DragEvent) {
    event.preventDefault();
    if (disabled || uploading) return;
    dragOver = true;
  }

  function handleDragLeave(event: DragEvent) {
    event.preventDefault();
    if (disabled || uploading) return;
    dragOver = false;
  }

  function handleDrop(event: DragEvent) {
    event.preventDefault();
    if (disabled || uploading) return;
    
    dragOver = false;
    const droppedFiles = Array.from(event.dataTransfer?.files || []);
    processFiles(droppedFiles);
  }

  function handleFileSelect(event: Event) {
    const target = event.target as HTMLInputElement;
    const selectedFiles = Array.from(target.files || []);
    processFiles(selectedFiles);
  }

  function processFiles(newFiles: File[]) {
    errorMessage = null;
    
    // Validate files
    const validFiles = newFiles.filter(file => {
      if (file.size > maxSize) {
        errorMessage = `File ${file.name} exceeds ${formatFileSize(maxSize)} limit`;
        return false;
      }
      return true;
    });

    if (multiple) {
      files = [...files, ...validFiles];
    } else {
      files = validFiles.slice(0, 1);
    }
  }

  function removeFile(index: number) {
    files = files.filter((_, i) => i !== index);
  }

  async function uploadFiles() {
    if (files.length === 0 || uploading) return;

    uploading = true;
    uploadStatus = 'uploading';
    uploadProgress = 0;
    errorMessage = null;

    try {
      const results: UploadResult[] = [];

      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const formData = new FormData();
        
        // Add file and metadata
        formData.append('file', file);
        formData.append('uploadData', JSON.stringify({
          caseId,
          title: file.name,
          description: `Uploaded via drag-and-drop: ${file.name}`,
          evidenceType: getEvidenceType(file),
          enableAiAnalysis: true,
          enableEmbeddings: true,
          enableOcr: file.type.startsWith('image/') || file.type === 'application/pdf'
        }));

        // Update progress
        uploadProgress = Math.round(((i / files.length) * 90) + 5);
        uploadStatus = 'processing';

        // Upload to MinIO via evidence API
        const response = await fetch('/api/evidence/upload', {
          method: 'POST',
          body: formData
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error?.message || 'Upload failed');
        }

        const result = await response.json();
        if (result.success && result.data?.[0]) {
          results.push(result.data[0]);
          
          // Publish Redis event for real-time updates
          await fetch('/api/v1/redis/publish', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              channel: 'evidence_update',
              data: {
                type: 'EVIDENCE_UPLOADED',
                evidenceId: result.data[0].id,
                caseId,
                fileName: file.name,
                timestamp: new Date().toISOString()
              }
            })
          }).catch(err => console.warn('Redis publish failed:', err));
        }
      }

      uploadProgress = 100;
      uploadStatus = 'completed';

      // Clear files and reset
      setTimeout(() => {
        files = [];
        uploadProgress = 0;
        uploadStatus = 'idle';
        if (fileInput) fileInput.value = '';
      }, 2000);

      // Notify success
      results.forEach(result => onUploadComplete?.(result));

    } catch (error) {
      console.error('Upload failed:', error);
      uploadStatus = 'error';
      errorMessage = error instanceof Error ? error.message : 'Upload failed';
      onUploadError?.(errorMessage);
    } finally {
      uploading = false;
    }
  }

  function getEvidenceType(file: File): string {
    if (file.type.startsWith('image/')) return 'IMAGE';
    if (file.type === 'application/pdf') return 'PDF';
    if (file.type.startsWith('text/')) return 'TEXT';
    if (file.type.startsWith('video/')) return 'VIDEO';
    if (file.type.startsWith('audio/')) return 'AUDIO';
    return 'DOCUMENT';
  }

  function formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  function openFileDialog() {
    if (!disabled && !uploading && fileInput) {
      fileInput.click();
    }
  }
</script>

<!-- MinIO Upload Zone -->
<div class="upload-container">
  <!-- Hidden file input -->
  <input
    bind:this={fileInput}
    type="file"
    {accept}
    {multiple}
    disabled={disabled || uploading}
    onchange={handleFileSelect}
    style="display: none"
  />

  <!-- Drag and Drop Zone -->
  <div
    class="drop-zone"
    class:drag-over={dragOver}
    class:has-files={files.length > 0}
    class:uploading
    role="button"
    tabindex="0"
    ondrop={handleDrop}
    ondragover={handleDragOver}
    ondragleave={handleDragLeave}
    onclick={openFileDialog}
    onkeydown={(e) => e.key === 'Enter' && openFileDialog()}
  >
    {#if files.length === 0}
      <div class="upload-prompt">
        <div class="upload-icon">
          {#if dragOver}
            <Upload class="w-12 h-12 text-blue-500" />
          {:else}
            <Upload class="w-12 h-12 text-gray-400" />
          {/if}
        </div>
        <div class="upload-text">
          <h3>{dragOver ? 'Drop files here' : 'Drop files or click to browse'}</h3>
          <p class="text-sm text-gray-500">
            Supports PDF, Word, Text, and Image files up to {formatFileSize(maxSize)}
          </p>
        </div>
      </div>
    {:else}
      <!-- File List -->
      <div class="file-list">
        {#each files as file, index}
          <div class="file-item">
            <div class="file-icon">
              {#if file.type.startsWith('image/')}
                <Image class="w-6 h-6" />
              {:else}
                <FileText class="w-6 h-6" />
              {/if}
            </div>
            <div class="file-info">
              <div class="file-name">{file.name}</div>
              <div class="file-size">{formatFileSize(file.size)}</div>
            </div>
            {#if !uploading}
              <button
                type="button"
                class="remove-file"
                onclick={(e) => { e.stopPropagation(); removeFile(index); }}
              >
                ✕
              </button>
            {/if}
          </div>
        {/each}
      </div>
    {/if}
  </div>

  <!-- Upload Progress -->
  {#if uploadStatus !== 'idle'}
    <div class="upload-progress">
      <div class="progress-bar">
        <div class="progress-fill" style="width: {uploadProgress}%"></div>
      </div>
      <div class="progress-text">
        {#if uploadStatus === 'uploading'}
          <Loader2 class="w-4 h-4 animate-spin" />
          Uploading to MinIO... {uploadProgress}%
        {:else if uploadStatus === 'processing'}
          <Loader2 class="w-4 h-4 animate-spin" />
          Processing with AI... {uploadProgress}%
        {:else if uploadStatus === 'completed'}
          <CheckCircle class="w-4 h-4 text-green-500" />
          Upload completed successfully
        {:else if uploadStatus === 'error'}
          <AlertCircle class="w-4 h-4 text-red-500" />
          {errorMessage || 'Upload failed'}
        {/if}
      </div>
    </div>
  {/if}

  <!-- Upload Actions -->
  <div class="upload-actions">
    <button
      type="button"
      class="upload-button"
      disabled={files.length === 0 || uploading || disabled}
      onclick={uploadFiles}
    >
      {#if uploading}
        <Loader2 class="w-4 h-4 animate-spin" />
        Uploading...
      {:else}
        <Upload class="w-4 h-4" />
        Upload to MinIO
      {/if}
    </button>

    {#if files.length > 0 && !uploading}
      <button
        type="button"
        class="clear-button"
        onclick={() => { files = []; if (fileInput) fileInput.value = ''; }}
      >
        Clear Files
      </button>
    {/if}
  </div>
</div>

<style>
  .upload-container {
    width: 100%;
    max-width: 600px;
    margin: 0 auto;
  }

  .drop-zone {
    border: 2px dashed #d1d5db;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s ease;
    background: #f9fafb;
    min-height: 200px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .drop-zone:hover:not(.uploading) {
    border-color: #3b82f6;
    background: #eff6ff;
  }

  .drop-zone.drag-over {
    border-color: #3b82f6;
    background: #dbeafe;
    transform: scale(1.02);
  }

  .drop-zone.has-files {
    border-style: solid;
    border-color: #10b981;
    background: #ecfdf5;
  }

  .drop-zone.uploading {
    cursor: not-allowed;
    opacity: 0.7;
  }

  .upload-prompt {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
  }

  .upload-icon {
    transition: transform 0.2s ease;
  }

  .drop-zone:hover .upload-icon:not(.uploading) {
    transform: scale(1.1);
  }

  .upload-text h3 {
    margin: 0;
    font-size: 1.125rem;
    font-weight: 600;
    color: #374151;
  }

  .file-list {
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .file-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem;
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    text-align: left;
  }

  .file-icon {
    flex-shrink: 0;
    color: #6b7280;
  }

  .file-info {
    flex: 1;
    min-width: 0;
  }

  .file-name {
    font-weight: 500;
    color: #374151;
    word-break: break-word;
  }

  .file-size {
    font-size: 0.875rem;
    color: #6b7280;
  }

  .remove-file {
    padding: 0.25rem;
    background: none;
    border: none;
    color: #ef4444;
    cursor: pointer;
    border-radius: 4px;
    transition: background-color 0.2s;
  }

  .remove-file:hover {
    background: #fee2e2;
  }

  .upload-progress {
    margin-top: 1rem;
    padding: 1rem;
    background: #f3f4f6;
    border-radius: 8px;
  }

  .progress-bar {
    width: 100%;
    height: 8px;
    background: #e5e7eb;
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 0.5rem;
  }

  .progress-fill {
    height: 100%;
    background: #3b82f6;
    transition: width 0.3s ease;
  }

  .progress-text {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.875rem;
    color: #374151;
  }

  .upload-actions {
    margin-top: 1rem;
    display: flex;
    gap: 0.75rem;
  }

  .upload-button {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    padding: 0.75rem 1.5rem;
    background: #3b82f6;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 500;
    cursor: pointer;
    transition: background-color 0.2s;
  }

  .upload-button:hover:not(:disabled) {
    background: #2563eb;
  }

  .upload-button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .clear-button {
    padding: 0.75rem 1rem;
    background: #f3f4f6;
    color: #374151;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
  }

  .clear-button:hover {
    background: #e5e7eb;
  }

  .error-alert {
    margin-top: 1rem;
    padding: 0.75rem;
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 8px;
    color: #dc2626;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  function formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  function getEvidenceType(file: File): string {
    if (file.type.startsWith('image/')) return 'IMAGE';
    if (file.type === 'application/pdf') return 'PDF';
    if (file.type.startsWith('text/')) return 'TEXT';
    if (file.type.startsWith('video/')) return 'VIDEO';
    if (file.type.startsWith('audio/')) return 'AUDIO';
    return 'DOCUMENT';
  }
</script>

<!-- Error Message -->
{#if errorMessage}
  <div class="error-alert">
    <AlertCircle class="w-4 h-4" />
    {errorMessage}
  </div>
{/if}