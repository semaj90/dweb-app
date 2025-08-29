<!-- MinIO Upload Component with SvelteKit 2 + Superforms + PostgreSQL Integration -->
<script lang="ts">
  import { $props, $state, $effect } from 'svelte';


  import { superForm } from 'sveltekit-superforms/client';
  import { zod } from 'sveltekit-superforms/adapters';
  import { evidenceUploadSchema, type EvidenceUploadData } from '$lib/schemas/evidence-upload.js';
  import { page } from '$app/state';
  import { invalidateAll } from '$app/navigation';
  import type { PageData } from './$types';
  import { createActor } from 'xstate';
  import { evidenceProcessingMachine } from '$lib/state/evidenceProcessingMachine';
  import { enhancedEvidenceProcessor } from '$lib/services/enhanced-evidence-processor.js';
  
  // Props
  interface Props {
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

  // Superforms setup with unified schema
  const { form, errors, enhance, submitting, message } = superForm(data.form, {
    validators: zod(evidenceUploadSchema),
    dataType: 'form',
    resetForm: false,
    invalidateAll: true,
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

  // Enhanced form submission with unified evidence processing
  function handleSubmit() {
    uploadStatus = 'uploading';
    uploadProgress = 0;
    processingError = null;
    
    return async ({ formData }: { formData: FormData }) => {
      try {
        // Initial upload to MinIO/storage
        uploadProgress = 10;
        
        // Get the uploaded file
        const file = formData.get('file') as File;
        if (!file) {
          throw new Error('No file selected');
        }

        console.log('Starting enhanced evidence processing:', {
          filename: file.name,
          size: file.size,
          type: file.type,
          evidenceType: $form.evidence_type
        });
        
        uploadProgress = 20;
        uploadStatus = 'processing';
        
        // Use Enhanced Evidence Processor service
        const processingResult = await enhancedEvidenceProcessor.processEvidence(
          file,
          $form.evidence_type || 'UNKNOWN',
          {
            enableOcr: $form.enableOcr || false,
            enableAiAnalysis: $form.enableAiAnalysis || false,
            enableEmbeddings: $form.enableEmbeddings || false,
            enableSummarization: $form.enableSummarization || false,
            caseId: $form.case_id,
            userId: 'current-user' // TODO: Get from auth context
          }
        );
        
        console.log('Enhanced processing completed:', {
          success: processingResult.success,
          processingTime: processingResult.processingTime,
          hasOcr: !!processingResult.ocrResult,
          hasAiAnalysis: !!processingResult.aiAnalysis,
          hasEmbeddings: !!processingResult.embeddings
        });
        
        if (processingResult.success) {
          uploadProgress = 100;
          uploadStatus = 'completed';
          
          // Trigger success callback with enhanced results
          const enhancedResult = {
            success: true,
            documentId: processingResult.evidenceId,
            url: '',
            objectName: processingResult.evidenceId,
            message: 'Evidence processed successfully with Enhanced RAG pipeline',
            processing: {
              metadata: processingResult.metadata,
              ocrResult: processingResult.ocrResult,
              aiAnalysis: processingResult.aiAnalysis,
              embeddings: processingResult.embeddings,
              processingTime: processingResult.processingTime
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
          
        } else {
          uploadStatus = 'error';
          processingError = processingResult.error || 'Processing failed';
          onUploadError?.(processingError);
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
      <label for="case_id">Case ID</label>
      <input
        id="case_id"
        name="case_id"
        type="text"
        bind:value={$form.case_id}
        placeholder="Enter case ID (optional)"
        disabled={disabled || $submitting}
        class="form-input"
        class:error={$errors.case_id}
      />
      {#if $errors.case_id}
        <div class="error-message">{$errors.case_id}</div>
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
        on:click={() => fileInput?.click()}
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
              <button type="button" class="remove-file" on:click={removeFile}>
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

    <!-- Evidence Type -->
    <div class="form-group">
      <label for="evidence_type">Evidence Type *</label>
      <select
        id="evidence_type"
        name="evidence_type"
        bind:value={$form.evidence_type}
        required
        disabled={disabled || $submitting}
        class="form-select"
      >
        <option value="UNKNOWN">Auto-detect from file</option>
        <option value="PDF">PDF Document</option>
        <option value="IMAGE">Image/Photo</option>
        <option value="VIDEO">Video Recording</option>
        <option value="AUDIO">Audio Recording</option>
        <option value="TEXT">Text Document</option>
        <option value="LINK">Web Link/URL</option>
      </select>
      {#if $errors.evidence_type}
        <div class="error-message">{$errors.evidence_type}</div>
      {/if}
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
      <label for="tags">Tags</label>
      <input
        id="tags"
        name="tags"
        type="text"
        bind:value={$form.tags}
        placeholder="e.g., contract, confidential, priority"
        disabled={disabled || $submitting}
        class="form-input"
      />
      {#if $errors.tags}
        <div class="error-message">{$errors.tags}</div>
      {/if}
    </div>

    <!-- Confidentiality Level -->
    <div class="form-group">
      <label for="confidentialityLevel">Confidentiality Level</label>
      <select
        id="confidentialityLevel"
        name="confidentialityLevel"
        bind:value={$form.confidentialityLevel}
        disabled={disabled || $submitting}
        class="form-select"
      >
        <option value="public">Public</option>
        <option value="standard">Standard</option>
        <option value="confidential">Confidential</option>
        <option value="classified">Classified</option>
        <option value="restricted">Restricted</option>
      </select>
    </div>

    <!-- Chain of Custody -->
    <div class="form-group">
      <label for="collectedBy">Collected By</label>
      <input
        id="collectedBy"
        name="collectedBy"
        type="text"
        bind:value={$form.collectedBy}
        placeholder="Officer/person who collected the evidence"
        disabled={disabled || $submitting}
        class="form-input"
      />
    </div>

    <div class="form-group">
      <label for="location">Collection Location</label>
      <input
        id="location"
        name="location"
        type="text"
        bind:value={$form.location}
        placeholder="Where the evidence was collected"
        disabled={disabled || $submitting}
        class="form-input"
      />
    </div>

    <!-- Evidence Admissibility -->
    <div class="form-group">
      <label class="checkbox-label">
        <input
          type="checkbox"
          name="isAdmissible"
          bind:checked={$form.isAdmissible}
          disabled={disabled || $submitting}
        />
        Evidence is admissible in court
      </label>
    </div>

    <!-- AI Processing Options -->
    <div class="form-group">
      <h3>AI Processing Options</h3>
      <div class="checkbox-grid">
        <label class="checkbox-label">
          <input
            type="checkbox"
            name="enableOcr"
            bind:checked={$form.enableOcr}
            disabled={disabled || $submitting}
          />
          Enable OCR (text extraction)
        </label>

        <label class="checkbox-label">
          <input
            type="checkbox"
            name="enableAiAnalysis"
            bind:checked={$form.enableAiAnalysis}
            disabled={disabled || $submitting}
          />
          Enable AI analysis
        </label>

        <label class="checkbox-label">
          <input
            type="checkbox"
            name="enableEmbeddings"
            bind:checked={$form.enableEmbeddings}
            disabled={disabled || $submitting}
          />
          Generate vector embeddings
        </label>

        <label class="checkbox-label">
          <input
            type="checkbox"
            name="enableSummarization"
            bind:checked={$form.enableSummarization}
            disabled={disabled || $submitting}
          />
          Generate summary
        </label>
      </div>
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
          on:click={() => {
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