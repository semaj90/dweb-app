<!--
Enhanced File Upload Component with localStorage Fallback
Automatically handles server upload with localStorage fallback
-->
<script lang="ts">
  import { enhancedFileUpload, type UploadResponse } from '$lib/services/enhanced-file-upload.js';
  import { localStorageFiles } from '$lib/services/localStorage-file-fallback.js';
  import { createEventDispatcher } from 'svelte';

  // Props
  interface Props {
    caseId?: string;
    description?: string;
    tags?: string[];
    multiple?: boolean;
    accept?: string;
    maxSize?: number; // MB
    forceLocalStorage?: boolean;
  }
  
  let {
    caseId,
    description,
    tags = [],
    multiple = false,
    accept = '*/*',
    maxSize = 10,
    forceLocalStorage = false
  }: Props = $props();

  // State
  let isDragOver = $state(false);
  let isUploading = $state(false);
  let uploadProgress = $state(0);
  let currentFile = $state('');
  let uploadResults = $state<UploadResponse[]>([]);
  let error = $state<string | null>(null);

  // Storage stats
  let storageStats = $state(localStorageFiles.getStorageUsage());

  // Event dispatcher
  const dispatch = createEventDispatcher<{
    upload: { results: UploadResponse[] };
    error: { error: string };
    progress: { completed: number; total: number; file: string };
  }>();

  // File input reference
let fileInput = $state<HTMLInputElement;

  /**
   * Handle file selection
   */
  async function handleFileSelect(files: FileList) {
    if (files.length >(== 0) return);

    // Validate files
    const validFiles: File[] = [];
    for (const file of Array.from(files)) {
      if (file.size > maxSize * 1024 * 1024) {
        error = `File ${file.name} is too large (max ${maxSize}MB)`;
        return;
      }
      validFiles.push(file);
    }

    await uploadFiles(validFiles);
  }

  /**
   * Upload files with progress tracking
   */
  async function uploadFiles(files: File[]) {
    isUploading = true;
    uploadProgress = 0;
    currentFile = '';
    error = null;
    uploadResults = [];

    try {
      const results = await enhancedFileUpload.uploadFiles(
        files,
        { caseId, description, tags, useLocalStorage: forceLocalStorage },
        (completed, total, file) => {
          uploadProgress = (completed / total) * 100;
          currentFile = file;
          dispatch('progress', { completed, total, file });
        }
      );

      uploadResults = results;
      storageStats = localStorageFiles.getStorageUsage();

      const successCount = results.filter(r => r.success).length;
      const errorCount = results.length - successCount;

      if (errorCount > 0) {
        error = `${errorCount} file(s) failed to upload`;
        dispatch('error', { error: error! });
      }

      dispatch('upload', { results });

    } catch (err: any) {
      error = err.message || 'Upload failed';
      dispatch('error', { error: error! });
    } finally {
      isUploading = false;
      uploadProgress = 0;
      currentFile = '';
    }
  }

  /**
   * Handle drag and drop
   */
  function handleDrop(event: DragEvent) {
    event.preventDefault();
    isDragOver = false;

    const files = event.dataTransfer?.files;
    if (files) {
      handleFileSelect(files);
    }
  }

  function handleDragOver(event: DragEvent) {
    event.preventDefault();
    isDragOver = true;
  }

  function handleDragLeave() {
    isDragOver = false;
  }

  /**
   * Open file selector
   */
  function openFileSelector() {
    fileInput.click();
  }

  /**
   * Clear results
   */
  function clearResults() {
    uploadResults = [];
    error = null;
  }
</script>

<div class="file-upload-container">
  <!-- Storage Usage Indicator -->
  {#if forceLocalStorage || storageStats.percentage > 0}
    <div class="storage-indicator">
      <div class="storage-bar">
        <div 
          class="storage-fill" 
          style="width: {storageStats.percentage}%"
          class:warning={storageStats.percentage > 75}
          class:critical={storageStats.percentage > 90}
        ></div>
      </div>
      <span class="storage-text">
        localStorage: {Math.round(storageStats.used / 1024)}KB / {Math.round(storageStats.available / 1024)}KB
      </span>
    </div>
  {/if}

  <!-- Drop Zone -->
  <div 
    class="drop-zone"
    class:drag-over={isDragOver}
    class:uploading={isUploading}
    ondrop={handleDrop}
    ondragover={handleDragOver}
    ondragleave={handleDragLeave}
    role="button"
    tabindex="0"
    on:on:onclick={openFileSelector}
    keydown={(e) => e.key === 'Enter' && openFileSelector()}
  >
    <div class="drop-zone-content">
      {#if isUploading}
        <div class="upload-progress">
          <div class="spinner"></div>
          <div class="progress-text">
            <div>Uploading {currentFile}...</div>
            <div class="progress-bar">
              <div class="progress-fill" style="width: {uploadProgress}%"></div>
            </div>
            <div>{Math.round(uploadProgress)}% complete</div>
          </div>
        </div>
      {:else}
        <div class="upload-icon">📁</div>
        <div class="upload-text">
          <strong>Drop files here or click to upload</strong>
          <div class="upload-subtitle">
            {multiple ? 'Multiple files allowed' : 'Single file only'} • Max {maxSize}MB per file
            {#if forceLocalStorage}
              <br><span class="fallback-notice">⚠️ Using localStorage fallback</span>
            {/if}
          </div>
        </div>
      {/if}
    </div>
  </div>

  <!-- Hidden file input -->
  <input
    bind:this={fileInput}
    type="file"
    {accept}
    {multiple}
    style="display: none"
    onchange={(e) => e.target?.files && handleFileSelect(e.target.files)}
  />

  <!-- Error Display -->
  {#if error}
    <div class="error-message">
      ❌ {error}
    </div>
  {/if}

  <!-- Results Display -->
  {#if uploadResults.length > 0}
    <div class="results-container">
      <div class="results-header">
        <h4>Upload Results</h4>
        <button class="clear-btn" on:on:onclick={clearResults}>Clear</button>
      </div>
      
      <div class="results-list">
        {#each uploadResults as result}
          <div class="result-item" class:success={result.success} class:error={!result.success}>
            <div class="result-icon">
              {result.success ? '✅' : '❌'}
            </div>
            <div class="result-details">
              <div class="result-name">{result.fileName}</div>
              <div class="result-meta">
                {#if result.success}
                  <span class="storage-type">{result.storageType}</span>
                  {#if result.fallbackUsed}
                    <span class="fallback-badge">localStorage fallback</span>
                  {/if}
                  <span class="file-size">{Math.round(result.size / 1024)}KB</span>
                {:else}
                  <span class="error-text">{result.error}</span>
                {/if}
              </div>
            </div>
          </div>
        {/each}
      </div>
    </div>
  {/if}
</div>

<style>
  .file-upload-container {
    width: 100%;
    max-width: 600px;
  }

  .storage-indicator {
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.875rem;
  }

  .storage-bar {
    flex: 1;
    height: 4px;
    background-color: #e5e7eb;
    border-radius: 2px;
    overflow: hidden;
  }

  .storage-fill {
    height: 100%;
    background-color: #3b82f6;
    transition: width 0.3s ease;
  }

  .storage-fill.warning {
    background-color: #f59e0b;
  }

  .storage-fill.critical {
    background-color: #ef4444;
  }

  .storage-text {
    color: #6b7280;
    white-space: nowrap;
  }

  .drop-zone {
    border: 2px dashed #d1d5db;
    border-radius: 8px;
    padding: 2rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s ease;
    background-color: #f9fafb;
  }

  .drop-zone:hover {
    border-color: #3b82f6;
    background-color: #f0f9ff;
  }

  .drop-zone.drag-over {
    border-color: #3b82f6;
    background-color: #dbeafe;
    transform: scale(1.02);
  }

  .drop-zone.uploading {
    border-color: #10b981;
    background-color: #f0fdf4;
  }

  .upload-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
  }

  .upload-text strong {
    display: block;
    color: #374151;
    margin-bottom: 0.5rem;
  }

  .upload-subtitle {
    color: #6b7280;
    font-size: 0.875rem;
  }

  .fallback-notice {
    color: #f59e0b;
    font-weight: 500;
  }

  .upload-progress {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .spinner {
    width: 24px;
    height: 24px;
    border: 2px solid #e5e7eb;
    border-top: 2px solid #3b82f6;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }

  .progress-text {
    flex: 1;
  }

  .progress-bar {
    width: 100%;
    height: 8px;
    background-color: #e5e7eb;
    border-radius: 4px;
    overflow: hidden;
    margin: 0.5rem 0;
  }

  .progress-fill {
    height: 100%;
    background-color: #10b981;
    transition: width 0.3s ease;
  }

  .error-message {
    margin-top: 1rem;
    padding: 1rem;
    background-color: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 6px;
    color: #dc2626;
  }

  .results-container {
    margin-top: 1rem;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    overflow: hidden;
  }

  .results-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem;
    background-color: #f9fafb;
    border-bottom: 1px solid #e5e7eb;
  }

  .results-header h4 {
    margin: 0;
    color: #374151;
  }

  .clear-btn {
    padding: 0.25rem 0.5rem;
    background-color: #6b7280;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.75rem;
  }

  .clear-btn:hover {
    background-color: #4b5563;
  }

  .results-list {
    max-height: 300px;
    overflow-y: auto;
  }

  .result-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 1rem;
    border-bottom: 1px solid #f3f4f6;
  }

  .result-item:last-child {
    border-bottom: none;
  }

  .result-item.success {
    background-color: #f0fdf4;
  }

  .result-item.error {
    background-color: #fef2f2;
  }

  .result-icon {
    font-size: 1.2rem;
  }

  .result-details {
    flex: 1;
  }

  .result-name {
    font-weight: 500;
    color: #374151;
  }

  .result-meta {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 0.25rem;
    font-size: 0.75rem;
  }

  .storage-type {
    padding: 0.125rem 0.375rem;
    background-color: #e0e7ff;
    color: #3730a3;
    border-radius: 12px;
    font-weight: 500;
  }

  .fallback-badge {
    padding: 0.125rem 0.375rem;
    background-color: #fef3c7;
    color: #92400e;
    border-radius: 12px;
    font-weight: 500;
  }

  .file-size {
    color: #6b7280;
  }

  .error-text {
    color: #dc2626;
  }
</style>