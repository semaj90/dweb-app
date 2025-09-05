<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { writable } from 'svelte/store';

  export let caseId: string;
  export let maxFileSize = 50 * 1024 * 1024; // 50MB default

  const dispatch = createEventDispatcher();

  let files: FileList | null = null;
  let dragActive = false;
  let uploading = false;
  let uploadProgress = writable(0);
  let uploadStatus = writable('');

  // File type categories for validation and UI
  const allowedTypes = {
    images: ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/bmp'],
    videos: ['video/mp4', 'video/webm', 'video/avi', 'video/mov', 'video/wmv'],
    documents: ['application/pdf', 'text/plain', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'],
    audio: ['audio/mp3', 'audio/wav', 'audio/m4a', 'audio/aac']
  };

  const allAllowedTypes = Object.values(allowedTypes).flat();

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
    
    const droppedFiles = e.dataTransfer?.files;
    if (droppedFiles) {
      files = droppedFiles;
      handleFileUpload();
    }
  }

  function handleFileSelect(e: Event) {
    const input = e.target as HTMLInputElement;
    files = input.files;
    if (files) {
      handleFileUpload();
    }
  }

  async function handleFileUpload() {
    if (!files || files.length === 0) return;

    uploading = true;
    uploadStatus.set('Preparing upload...');
    uploadProgress.set(0);

    try {
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        
        // Validate file type
        if (!allAllowedTypes.includes(file.type)) {
          uploadStatus.set(`Unsupported file type: ${file.type}`);
          continue;
        }

        // Validate file size
        if (file.size > maxFileSize) {
          uploadStatus.set(`File too large: ${file.name} (${(file.size / 1024 / 1024).toFixed(1)}MB)`);
          continue;
        }

        uploadStatus.set(`Uploading ${file.name}...`);

        const formData = new FormData();
        formData.append('file', file);
        formData.append('caseId', caseId);
        formData.append('title', file.name);
        formData.append('evidenceType', getEvidenceType(file.type));

        const response = await fetch('/api/upload', {
          method: 'POST',
          body: formData
        });

        if (response.ok) {
          const result = await response.json();
          uploadProgress.set(((i + 1) / files.length) * 100);
          
          // Dispatch success event
          dispatch('uploaded', {
            file,
            evidence: result.evidence
          });
        } else {
          const error = await response.json();
          uploadStatus.set(`Upload failed: ${error.error}`);
        }
      }

      uploadStatus.set('Upload complete');
      setTimeout(() => {
        uploadStatus.set('');
        uploadProgress.set(0);
      }, 2000);

    } catch (error) {
      uploadStatus.set(`Upload error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      uploading = false;
      files = null;
    }
  }

  function getEvidenceType(mimeType: string): string {
    if (allowedTypes.images.includes(mimeType)) return 'photograph';
    if (allowedTypes.videos.includes(mimeType)) return 'video';
    if (allowedTypes.documents.includes(mimeType)) return 'document';
    if (allowedTypes.audio.includes(mimeType)) return 'audio';
    return 'physical';
  }

  function getFileIcon(mimeType: string): string {
    if (allowedTypes.images.includes(mimeType)) return '🖼️';
    if (allowedTypes.videos.includes(mimeType)) return '🎥';
    if (allowedTypes.documents.includes(mimeType)) return '📄';
    if (allowedTypes.audio.includes(mimeType)) return '🎵';
    return '📁';
  }

  function formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }
</script>

<div class="mx-auto px-4 max-w-7xl">
  <div 
    class="mx-auto px-4 max-w-7xl"
    class:drag-active={dragActive}
    class:uploading
    on:dragover={handleDragOver}
    on:dragleave={handleDragLeave}
    on:drop={handleDrop}
    role="button"
    tabindex={0}
    onclick={() => document.getElementById('file-input')?.click()}
    onkeydown={(e) => e.key === 'Enter' && document.getElementById('file-input')?.click()}
  >
    <input
      id="file-input"
      type="file"
      multiple
      accept={allAllowedTypes.join(',')}
      style="display: none;"
      onchange={handleFileSelect}
    />

    {#if uploading}
      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">⏳</div>
        <div class="mx-auto px-4 max-w-7xl">
          {$uploadStatus}
        </div>
        {#if $uploadProgress > 0}
          <div class="mx-auto px-4 max-w-7xl">
            <div class="mx-auto px-4 max-w-7xl" style="width: {$uploadProgress}%"></div>
          </div>
        {/if}
      </div>
    {:else}
      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">📤</div>
        <h3>Upload Evidence</h3>
        <p>Drag and drop files here or click to browse</p>
        <div class="mx-auto px-4 max-w-7xl">
          <span class="mx-auto px-4 max-w-7xl">🖼️ Images</span>
          <span class="mx-auto px-4 max-w-7xl">🎥 Videos</span>
          <span class="mx-auto px-4 max-w-7xl">📄 Documents</span>
          <span class="mx-auto px-4 max-w-7xl">🎵 Audio</span>
        </div>
        <div class="mx-auto px-4 max-w-7xl">
          Max file size: {formatFileSize(maxFileSize)}
        </div>
      </div>
    {/if}
  </div>

  <!-- File preview if files selected but not uploaded yet -->
  {#if files && files.length > 0 && !uploading}
    <div class="mx-auto px-4 max-w-7xl">
      <h4>Selected Files ({files.length})</h4>
      {#each Array.from(files) as file}
        <div class="mx-auto px-4 max-w-7xl">
          <span class="mx-auto px-4 max-w-7xl">{getFileIcon(file.type)}</span>
          <div class="mx-auto px-4 max-w-7xl">
            <div class="mx-auto px-4 max-w-7xl">{file.name}</div>
            <div class="mx-auto px-4 max-w-7xl">
              {formatFileSize(file.size)} • {file.type}
            </div>
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .evidence-uploader {
    width: 100%;
  }

  .upload-zone {
    border: 2px dashed #ccc;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
    background: var(--background-alt, #f8f9fa);
  }

  .upload-zone:hover {
    border-color: var(--primary, #007bff);
    background: var(--background-hover, #e9ecef);
  }

  .upload-zone.drag-active {
    border-color: var(--primary, #007bff);
    background: var(--primary-light, #e7f3ff);
    transform: scale(1.02);
  }

  .upload-zone.uploading {
    border-color: var(--warning, #ffc107);
    cursor: not-allowed;
  }

  .upload-prompt .upload-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
  }

  .upload-prompt h3 {
    margin: 0 0 0.5rem 0;
    color: var(--text-primary, #333);
  }

  .upload-prompt p {
    margin: 0 0 1rem 0;
    color: var(--text-secondary, #666);
  }

  .file-types {
    display: flex;
    gap: 1rem;
    justify-content: center;
    flex-wrap: wrap;
    margin-bottom: 1rem;
  }

  .file-type {
    padding: 0.25rem 0.5rem;
    background: var(--surface, #fff);
    border-radius: 4px;
    font-size: 0.875rem;
    color: var(--text-secondary, #666);
  }

  .size-limit {
    font-size: 0.875rem;
    color: var(--text-muted, #999);
  }

  .upload-progress {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
  }

  .upload-spinner {
    font-size: 2rem;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  .upload-message {
    font-weight: 500;
    color: var(--text-primary, #333);
  }

  .progress-bar {
    width: 100%;
    max-width: 300px;
    height: 8px;
    background: var(--surface, #e9ecef);
    border-radius: 4px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    background: var(--primary, #007bff);
    transition: width 0.3s ease;
  }

  .file-preview {
    margin-top: 1rem;
    padding: 1rem;
    background: var(--surface, #fff);
    border-radius: 8px;
    border: 1px solid var(--border, #dee2e6);
  }

  .file-preview h4 {
    margin: 0 0 1rem 0;
    color: var(--text-primary, #333);
  }

  .file-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--border-light, #f1f3f4);
  }

  .file-item:last-child {
    border-bottom: none;
  }

  .file-icon {
    font-size: 1.5rem;
  }

  .file-info {
    flex: 1;
  }

  .file-name {
    font-weight: 500;
    color: var(--text-primary, #333);
  }

  .file-meta {
    font-size: 0.875rem;
    color: var(--text-secondary, #666);
    margin-top: 0.25rem;
  }
</style>
