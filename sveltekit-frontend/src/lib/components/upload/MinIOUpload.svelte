<!-- MinIO Upload Component with SvelteKit 2 + Superforms + PostgreSQL Integration -->
<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';

  import { superForm } from 'sveltekit-superforms/client';
  import { fileUploadSchema, type FileUploadData } from '$lib/schemas/upload';
  import { page } from '$app/state';
  import { invalidateAll } from '$app/navigation';
  import type { PageData } from './$types';
  import { createActor } from 'xstate';
  import { evidenceProcessingMachine } from '$lib/state/evidenceProcessingMachine';
  import { documentApiService } from '$lib/services/documentApi';
  
  // Props
  interface Props extends CommonProps {
    data: PageData;
    caseId?: string;
    onUploadComplete?: (result: UploadResult) => void;
    onUploadError?: (error: string) => void;
    multiple?: boolean;
    disabled?: boolean;
  }
  
  let { 
    data, 
    caseId = '', 
    onUploadComplete,
    onUploadError,
    multiple = false,
    disabled = false 
  }: Props = $props();

  interface UploadResult {
    success: boolean;
    documentId: string;
    url: string;
    objectName: string;
    message: string;
  }

  // Superforms setup
  const { form, errors, enhance, submitting, message } = superForm(data.form, {
    dataType: 'form',
    multipleFiles: true,
    validators: {
      file: (value) => {
        if (!value || !(value instanceof File)) return 'File is required';
        
        const maxSize = 100 * 1024 * 1024; // 100MB
        if (value.size > maxSize) return 'File must be less than 100MB';
        
        const allowedTypes = [
          'application/pdf',
          'application/msword',
          'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
          'text/plain',
          'image/jpeg',
          'image/png',
          'image/tiff'
        ];
        
        if (!allowedTypes.includes(value.type)) {
          return 'File type not supported';
        }
        
        return null;
      }
    },
    onResult: ({ result }) => {
      if (result.type === 'success') {
        const uploadResult = result.data?.uploadResult as UploadResult;
        if (uploadResult?.success) {
          onUploadComplete?.(uploadResult);
          // Reset form
          $form.file = undefined as any;
          $form.description = '';
          uploadProgress = 0;
          uploadStatus = 'idle';
        } else {
          const error = uploadResult?.message || 'Upload failed';
          onUploadError?.(error);
          uploadStatus = 'error';
        }
      } else if (result.type === 'error') {
        onUploadError?.('Upload failed: ' + result.error?.message);
        uploadStatus = 'error';
      }
    }
  });

  // Upload state
  let uploadProgress = $state(0);
  let uploadStatus: 'idle' | 'uploading' | 'processing' | 'completed' | 'error' = $state('idle');
  let fileInput: HTMLInputElement;
  let dragOver = $state(false);
  let previewUrl = $state<string | null>(null);

  // XState evidence processing actor
  let evidenceActor = $state<ReturnType<typeof createActor> | null>(null);
  let processingStage = $state('');
  let processingError = $state<string | null>(null);

  // Set default caseId if provided
  $effect(() => {
    if (caseId && !$form.caseId) {
      $form.caseId = caseId;
    }
  });

  // File handling
  function handleFileSelect(event: Event) {
    const target = event.target as HTMLInputElement;
    const file = target.files?.[0];
    if (file) {
      $form.file = file;
      generatePreview(file);
    }
  }

  function handleDrop(event: DragEvent) {
    event.preventDefault();
    dragOver = false;
    
    const files = event.dataTransfer?.files;
    if (files && files.length > 0) {
      $form.file = files[0];
      generatePreview(files[0]);
    }
  }

  function handleDragOver(event: DragEvent) {
    event.preventDefault();
    dragOver = true;
  }

  function handleDragLeave() {
    dragOver = false;
  }

  function generatePreview(file: File) {
    if (file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        previewUrl = e.target?.result as string;
      };
      reader.readAsDataURL(file);
    } else {
      previewUrl = null;
    }
  }

  function removeFile() {
    $form.file = undefined as any;
    previewUrl = null;
    if (fileInput) {
      fileInput.value = '';
    }
  }

  // Enhanced form submission with XState evidence processing
  function handleSubmit() {
    uploadStatus = 'uploading';
    uploadProgress = 0;
    processingError = null;
    
    return async ({ formData }: { formData: FormData }) => {
      try {
        // Initial upload to MinIO/storage
        uploadProgress = 10;
        
        // First upload the file and get basic document info
        const file = formData.get('file') as File;
        if (!file) {
          throw new Error('No file selected');
        }

        // Read file content for processing
        const fileContent = await file.text();
        const evidenceId = `evidence_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        
        uploadProgress = 20;
        uploadStatus = 'processing';
        
        // Create and start evidence processing actor
        evidenceActor = createActor(evidenceProcessingMachine);
        evidenceActor.start();
        
        // Subscribe to state changes
        evidenceActor.subscribe((state) => {
          processingStage = state.context.stage;
          uploadProgress = state.context.progress;
          
          if (state.context.error) {
            processingError = state.context.error;
            uploadStatus = 'error';
          }
          
          if (state.matches('completed')) {
            uploadStatus = 'completed';
            uploadProgress = 100;
            
            // Trigger success callback
            const enhancedResult = {
              success: true,
              documentId: evidenceId,
              url: '',
              objectName: evidenceId,
              message: 'Evidence processed successfully through XState pipeline',
              processing: {
                extractedText: state.context.extractedText,
                embeddings: state.context.embeddings,
                analysis: state.context.analysis,
                chunks: state.context.chunks
              }
            };
            
            onUploadComplete?.(enhancedResult);
            
            // Reset after delay
            setTimeout(() => {
              uploadProgress = 0;
              uploadStatus = 'idle';
              processingStage = '';
              evidenceActor?.stop();
              evidenceActor = null;
            }, 3000);
          }
          
          if (state.matches('failed')) {
            uploadStatus = 'error';
            processingError = state.context.error || 'Processing failed';
            onUploadError?.(processingError);
            evidenceActor?.stop();
            evidenceActor = null;
          }
        });
        
        // Start the evidence processing workflow
        evidenceActor.send({
          type: 'START_PROCESSING',
          evidenceId,
          caseId: $form.caseId || 'default',
          userId: 'current-user', // TODO: Get from auth
          filename: file.name,
          content: fileContent,
          metadata: {
            documentType: $form.documentType,
            description: $form.description,
            priority: $form.priority,
            tags: $form.tags?.split(',').map(t => t.trim()) || [],
            isConfidential: $form.isConfidential,
            fileSize: file.size,
            mimeType: file.type
          }
        });
        
        // Also process through legal ingest API if it's a legal document
        if ($form.documentType === 'evidence' || $form.documentType === 'contract') {
          try {
            const legalResult = await documentApiService.processLegalDocuments([file], {
              caseId: $form.caseId || 'default',
              jurisdiction: 'federal',
              enhanceRAG: true
            });
            
            console.log('Legal processing result:', legalResult);
          } catch (legalError) {
            console.warn('Legal processing failed (non-critical):', legalError);
          }
        }
        
      } catch (error) {
        console.error('Upload/processing failed:', error);
        uploadStatus = 'error';
        uploadProgress = 0;
        processingError = error instanceof Error ? error.message : 'Upload failed';
        onUploadError?.(processingError);
        evidenceActor?.stop();
        evidenceActor = null;
      }
    };
  }

  // Format file size
  function formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  // Document type options
  const documentTypes = [
    { value: 'contract', label: 'Contract' },
    { value: 'evidence', label: 'Evidence' },
    { value: 'pleading', label: 'Pleading' },
    { value: 'motion', label: 'Motion' },
    { value: 'brief', label: 'Brief' },
    { value: 'correspondence', label: 'Correspondence' },
    { value: 'exhibit', label: 'Exhibit' },
    { value: 'transcript', label: 'Transcript' },
    { value: 'discovery', label: 'Discovery' },
    { value: 'expert_report', label: 'Expert Report' },
    { value: 'forensic_analysis', label: 'Forensic Analysis' },
    { value: 'other', label: 'Other' }
  ];

  const priorityOptions = [
    { value: 'low', label: 'Low' },
    { value: 'medium', label: 'Medium' },
    { value: 'high', label: 'High' },
    { value: 'urgent', label: 'Urgent' }
  ];
</script>

<div class="minio-upload-container">
  <form method="POST" action="?/upload" use:enhance={handleSubmit} enctype="multipart/form-data">
    <!-- Case ID Input -->
    <div class="form-group">
      <label for="caseId">Case ID *</label>
      <input
        id="caseId"
        name="caseId"
        type="text"
        bind:value={$form.caseId}
        placeholder="Enter case ID"
        required
        disabled={disabled || $submitting}
        class="form-input"
        class:error={$errors.caseId}
      />
      {#if $errors.caseId}
        <div class="error-message">{$errors.caseId}</div>
      {/if}
    </div>

    <!-- File Upload Area -->
    <div class="form-group">
      <label>Document Upload *</label>
      <div
        class="file-upload-area"
        class:drag-over={dragOver}
        class:has-file={$form.file}
        role="button"
        tabindex="0"
        ondrop={handleDrop}
        ondragover={handleDragOver}
        ondragleave={handleDragLeave}
        onclick={() => fileInput?.click()}
        onkeydown={(e) => e.key === 'Enter' && fileInput?.click()}
      >
        <input
          bind:this={fileInput}
          type="file"
          name="file"
          accept=".pdf,.doc,.docx,.txt,.jpg,.jpeg,.png,.tiff"
          onchange={handleFileSelect}
          style="display: none"
          disabled={disabled || $submitting}
        />

        {#if $form.file}
          <div class="file-preview">
            {#if previewUrl}
              <img src={previewUrl} alt="Preview" class="image-preview" />
            {:else}
              <div class="file-icon">📄</div>
            {/if}
            <div class="file-info">
              <div class="file-name">{$form.file.name}</div>
              <div class="file-size">{formatFileSize($form.file.size)}</div>
              <button type="button" class="remove-file" onclick={removeFile}>
                ✕ Remove
              </button>
            </div>
          </div>
        {:else}
          <div class="upload-prompt">
            <div class="upload-icon">📤</div>
            <div class="upload-text">
              <div>Drop your document here or click to browse</div>
              <div class="upload-hint">PDF, Word, Text, or Image files up to 100MB</div>
            </div>
          </div>
        {/if}
      </div>
      {#if $errors.file}
        <div class="error-message">{$errors.file}</div>
      {/if}
    </div>

    <!-- Document Type -->
    <div class="form-group">
      <label for="documentType">Document Type *</label>
      <select
        id="documentType"
        name="documentType"
        bind:value={$form.documentType}
        required
        disabled={disabled || $submitting}
        class="form-select"
      >
        {#each documentTypes as option}
          <option value={option.value}>{option.label}</option>
        {/each}
      </select>
    </div>

    <!-- Description -->
    <div class="form-group">
      <label for="description">Description</label>
      <textarea
        id="description"
        name="description"
        bind:value={$form.description}
        placeholder="Optional description of the document"
        rows="3"
        maxlength="1000"
        disabled={disabled || $submitting}
        class="form-textarea"
      ></textarea>
    </div>

    <!-- Priority -->
    <div class="form-group">
      <label for="priority">Priority</label>
      <select
        id="priority"
        name="priority"
        bind:value={$form.priority}
        disabled={disabled || $submitting}
        class="form-select"
      >
        {#each priorityOptions as option}
          <option value={option.value}>{option.label}</option>
        {/each}
      </select>
    </div>

    <!-- Tags -->
    <div class="form-group">
      <label for="tags">Tags (comma-separated)</label>
      <input
        id="tags"
        name="tags"
        type="text"
        placeholder="e.g., contract, confidential, priority"
        disabled={disabled || $submitting}
        class="form-input"
      />
    </div>

    <!-- Confidential Flag -->
    <div class="form-group">
      <label class="checkbox-label">
        <input
          type="checkbox"
          name="isConfidential"
          bind:checked={$form.isConfidential}
          disabled={disabled || $submitting}
        />
        Mark as confidential
      </label>
    </div>

    <!-- Upload Progress -->
    {#if uploadStatus !== 'idle'}
      <div class="upload-progress">
        <div class="progress-bar">
          <div class="progress-fill" style="width: {uploadProgress}%"></div>
        </div>
        <div class="progress-text">
          {#if uploadStatus === 'uploading'}
            Uploading... {Math.round(uploadProgress)}%
          {:else if uploadStatus === 'processing'}
            {#if processingStage}
              Processing: {processingStage} ({Math.round(uploadProgress)}%)
            {:else}
              Processing document... ({Math.round(uploadProgress)}%)
            {/if}
          {:else if uploadStatus === 'completed'}
            Processing completed ✅
          {:else if uploadStatus === 'error'}
            {#if processingError}
              Error: {processingError} ❌
            {:else}
              Upload failed ❌
            {/if}
          {/if}
        </div>
      </div>
    {/if}

    <!-- Submit Button -->
    <div class="form-actions">
      <button
        type="submit"
        disabled={disabled || $submitting || !$form.file || !$form.caseId}
        class="submit-button"
      >
        {#if $submitting}
          Uploading...
        {:else}
          Upload Document
        {/if}
      </button>
      
      <!-- Retry button for failed processing -->
      {#if uploadStatus === 'error' && evidenceActor && processingError}
        <button
          type="button"
          class="retry-button"
          onclick={() => {
            if (evidenceActor) {
              processingError = null;
              uploadStatus = 'processing';
              evidenceActor.send({ type: 'RETRY' });
            }
          }}
        >
          Retry Processing
        </button>
      {/if}
    </div>

    <!-- Messages -->
    {#if $message}
      <div class="form-message" class:error={uploadStatus === 'error'}>
        {$message}
      </div>
    {/if}
  </form>
</div>

<style>
  .minio-upload-container {
    max-width: 600px;
    margin: 0 auto;
    padding: 2rem;
    background: var(--bg-secondary);
    border-radius: 12px;
    border: 1px solid var(--border-color);
  }

  .form-group {
    margin-bottom: 1.5rem;
  }

  .form-group label {
    display: block;
    margin-bottom: 0.5rem;
    font-weight: 600;
    color: var(--text-primary);
  }

  .form-input,
  .form-select,
  .form-textarea {
    width: 100%;
    padding: 0.75rem;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-family: inherit;
    transition: border-color 0.2s;
  }

  .form-input:focus,
  .form-select:focus,
  .form-textarea:focus {
    outline: none;
    border-color: var(--accent-primary);
    box-shadow: 0 0 0 3px var(--accent-primary-20);
  }

  .form-input.error {
    border-color: var(--error-color);
  }

  .file-upload-area {
    border: 2px dashed var(--border-color);
    border-radius: 8px;
    padding: 2rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    background: var(--bg-primary);
  }

  .file-upload-area:hover,
  .file-upload-area.drag-over {
    border-color: var(--accent-primary);
    background: var(--accent-primary-10);
  }

  .file-upload-area.has-file {
    border-style: solid;
    border-color: var(--success-color);
  }

  .upload-prompt {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
  }

  .upload-icon {
    font-size: 3rem;
    opacity: 0.6;
  }

  .upload-text {
    color: var(--text-secondary);
  }

  .upload-hint {
    font-size: 0.875rem;
    opacity: 0.8;
  }

  .file-preview {
    display: flex;
    align-items: center;
    gap: 1rem;
    text-align: left;
  }

  .image-preview {
    width: 80px;
    height: 80px;
    object-fit: cover;
    border-radius: 6px;
  }

  .file-icon {
    width: 80px;
    height: 80px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    background: var(--bg-secondary);
    border-radius: 6px;
  }

  .file-info {
    flex: 1;
  }

  .file-name {
    font-weight: 600;
    margin-bottom: 0.25rem;
  }

  .file-size {
    color: var(--text-secondary);
    font-size: 0.875rem;
  }

  .remove-file {
    margin-top: 0.5rem;
    padding: 0.25rem 0.5rem;
    border: 1px solid var(--error-color);
    background: transparent;
    color: var(--error-color);
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.875rem;
  }

  .remove-file:hover {
    background: var(--error-color);
    color: white;
  }

  .checkbox-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    cursor: pointer;
  }

  .upload-progress {
    margin: 1rem 0;
  }

  .progress-bar {
    width: 100%;
    height: 8px;
    background: var(--bg-tertiary);
    border-radius: 4px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    background: var(--accent-primary);
    transition: width 0.3s ease;
  }

  .progress-text {
    margin-top: 0.5rem;
    text-align: center;
    font-size: 0.875rem;
    color: var(--text-secondary);
  }

  .form-actions {
    margin-top: 2rem;
  }

  .submit-button {
    width: 100%;
    padding: 0.875rem;
    background: var(--accent-primary);
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    cursor: pointer;
    transition: background-color 0.2s;
  }

  .submit-button:hover:not(:disabled) {
    background: var(--accent-primary-dark);
  }

  .submit-button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .retry-button {
    margin-top: 0.5rem;
    padding: 0.5rem 1rem;
    background: var(--warning-color);
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 500;
    cursor: pointer;
    transition: background-color 0.2s;
  }

  .retry-button:hover {
    background: var(--warning-color-dark);
  }

  .error-message {
    color: var(--error-color);
    font-size: 0.875rem;
    margin-top: 0.25rem;
  }

  .form-message {
    margin-top: 1rem;
    padding: 0.75rem;
    border-radius: 6px;
    background: var(--success-color-20);
    color: var(--success-color);
    border: 1px solid var(--success-color);
  }

  .form-message.error {
    background: var(--error-color-20);
    color: var(--error-color);
    border-color: var(--error-color);
  }
</style>