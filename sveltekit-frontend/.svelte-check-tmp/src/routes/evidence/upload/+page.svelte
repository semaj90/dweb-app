<!--
  Evidence Upload Page - SvelteKit + Zod + Superforms Integration
  Rich metadata support with type-safe validation
-->
<script lang="ts">
  import { superForm } from 'sveltekit-superforms/client';
  import { zod } from 'sveltekit-superforms/adapters';
  import { evidenceUploadSchema, validateFileSize, validateFileType, getFileTypeFromMime, generateMetadataFromFile } from '$lib/schemas/evidence-upload.js';
  import type { PageData } from './$types.js';
  
  let { data }: { data: PageData } = $props();
  
  // Initialize Superform with Zod validation
  const { form, errors, enhance, submitting, message } = superForm(data.form, {
    validators: zod(evidenceUploadSchema),
    resetForm: false,
    invalidateAll: true
  });
  
  // File upload state
  let selectedFile: File | null = $state(null);
  let filePreview: string | null = $state(null);
  let dragOver = $state(false);
  let uploading = $state(false);
  let progressPercent = $state(0);
  let metadata = $state<any>(null);
  
  // Handle file selection
  async function handleFileSelect(file: File) {
    selectedFile = file;
    
    // Validate file size
    if (!validateFileSize(file)) {
      $errors.file = ['File size exceeds 100MB limit'];
      selectedFile = null;
      return;
    }
    
    // Auto-detect evidence type from file
    const detectedType = getFileTypeFromMime(file.type);
    if (detectedType !== 'UNKNOWN') {
      $form.evidence_type = detectedType as any;
    }
    
    // Validate file type against evidence type
    if (!validateFileType(file, $form.evidence_type)) {
      $errors.file = [`File type ${file.type} not supported for ${$form.evidence_type} evidence`];
      selectedFile = null;
      return;
    }
    
    // Generate file preview for images
    if (file.type.startsWith('image/')) {
      filePreview = URL.createObjectURL(file);
    } else {
      filePreview = null;
    }
    
    // Generate metadata preview
    try {
      metadata = await generateMetadataFromFile(file, $form.evidence_type);
    } catch (error) {
      console.warn('Failed to generate metadata preview:', error);
      metadata = null;
    }
    
    // Clear any file errors
    if ($errors.file) {
      delete $errors.file;
      $errors = $errors;
    }
  }
  
  // File input change handler
  function onFileChange(event: Event) {
    const target = event.target as HTMLInputElement;
    const file = target.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  }
  
  // Drag and drop handlers
  function onDragOver(event: DragEvent) {
    event.preventDefault();
    dragOver = true;
  }
  
  function onDragLeave(event: DragEvent) {
    event.preventDefault();
    dragOver = false;
  }
  
  function onDrop(event: DragEvent) {
    event.preventDefault();
    dragOver = false;
    
    const file = event.dataTransfer?.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  }
  
  // Evidence type change handler
  function onEvidenceTypeChange() {
    if (selectedFile) {
      // Re-validate file when evidence type changes
      if (!validateFileType(selectedFile, $form.evidence_type)) {
        $errors.file = [`File type ${selectedFile.type} not supported for ${$form.evidence_type} evidence`];
      } else if ($errors.file) {
        delete $errors.file;
        $errors = $errors;
      }
    }
  }
  
  // Format file size for display
  function formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }
</script>

<svelte:head>
  <title>Upload Evidence - Legal AI Platform</title>
</svelte:head>

<div class="nes-container with-title is-centered" style="margin: 20px;">
  <p class="title">Legal AI Evidence Upload</p>
  
  <div class="nes-container is-rounded" style="margin: 20px 0;">
    <h1 class="title">📁 Upload Evidence</h1>
    <p>Add new evidence to your case with automatic metadata extraction and AI processing.</p>
    
    <!-- Service Status Indicator -->
    <div class="service-status" style="margin: 15px 0; padding: 10px; border: 1px solid #ccc; background: #f9f9f9; border-radius: 4px;">
      <p style="margin: 0; font-size: 0.9em;">
        🔧 <strong>Processing Services:</strong> 
        <span style="color: #28a745;">✅ Go Upload Service (Connected)</span> |
        <span style="color: #28a745;">✅ Local OCR Processing</span> |
        <span style="color: #28a745;">✅ Database Storage</span>
      </p>
      <p style="margin: 5px 0 0 0; font-size: 0.8em; color: #666;">
        Your files will be processed by multiple AI services for enhanced analysis.
      </p>
    </div>
      
      {#if $message}
        <div class="nes-container {$message.type === 'success' ? 'is-success' : 'is-error'}" style="margin: 10px 0;">
          <p>{$message.text}</p>
        </div>
      {/if}
      
      <form method="POST" action="?/upload" enctype="multipart/form-data" use:enhance class="space-y-6">
        
        <!-- Case Selection -->
        <div class="nes-field" style="margin: 15px 0;">
          <label for="case_id">⚖️ Select Case *</label>
          <div class="nes-select">
            <select
              name="case_id"
              id="case_id"
              required
              disabled={$submitting}
              bind:value={$form.case_id}
            >
              <option value="">Choose a case...</option>
              {#each data.cases as caseItem}
                <option value={caseItem.id}>
                  {caseItem.case_number ? `${caseItem.case_number}: ` : ''}{caseItem.title}
                  {caseItem.status !== 'active' ? ` (${caseItem.status})` : ''}
                </option>
              {/each}
            </select>
          </div>
          {#if $errors.case_id}
            <p class="nes-text is-error">{$errors.case_id}</p>
          {/if}
        </div>
        
        <!-- Evidence Title -->
        <div class="nes-field" style="margin: 15px 0;">
          <label for="title">📝 Evidence Title *</label>
          <input
            type="text"
            name="title"
            id="title"
            required
            disabled={$submitting}
            bind:value={$form.title}
            class="nes-input"
            placeholder="e.g., Signed Contract Document"
          />
          {#if $errors.title}
            <p class="nes-text is-error">{$errors.title}</p>
          {/if}
        </div>
        
        <!-- Evidence Description -->
        <div class="nes-field" style="margin: 15px 0;">
          <label for="description">📄 Description</label>
          <textarea
            name="description"
            id="description"
            rows="3"
            disabled={$submitting}
            bind:value={$form.description}
            class="nes-textarea"
            placeholder="Brief description of the evidence..."
          ></textarea>
        </div>
        
        <!-- Evidence Type -->
        <div class="nes-field" style="margin: 15px 0;">
          <label for="evidence_type">🗂️ Evidence Type</label>
          <div class="nes-select">
            <select
              name="evidence_type"
              id="evidence_type"
              disabled={$submitting}
              bind:value={$form.evidence_type}
              change={onEvidenceTypeChange}
            >
              <option value="UNKNOWN">🔍 Auto-detect from file</option>
              <option value="PDF">📄 PDF Document</option>
              <option value="IMAGE">🖼️ Image/Photo</option>
              <option value="VIDEO">🎥 Video Recording</option>
              <option value="AUDIO">🎵 Audio Recording</option>
              <option value="TEXT">📝 Text Document</option>
              <option value="LINK">🔗 Web Link/URL</option>
            </select>
          </div>
          {#if $errors.evidence_type}
            <p class="nes-text is-error">{$errors.evidence_type}</p>
          {/if}
        </div>
        
        <!-- File Upload Area -->
        {#if $form.evidence_type !== 'LINK'}
          <div class="nes-field" style="margin: 15px 0;">
            <label>📎 File Upload *</label>
            
            <!-- Drag and Drop Zone -->
            <div
              class="nes-container {dragOver ? 'is-success' : ''} {$errors.file ? 'is-error' : ''}"
              style="padding: 30px; text-align: center; cursor: pointer;"
              ondragover={onDragOver}
              ondragleave={onDragLeave}
              ondrop={onDrop}
            >
              {#if selectedFile}
                <div class="space-y-4">
                  {#if filePreview}
                    <img src={filePreview} alt="Preview" class="max-w-xs max-h-48 mx-auto rounded-lg shadow-md" />
                  {:else}
                    <div class="w-16 h-16 mx-auto bg-gray-100 rounded-lg flex items-center justify-center">
                      <svg class="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                      </svg>
                    </div>
                  {/if}
                  
                  <div>
                    <p class="font-medium text-gray-900">{selectedFile.name}</p>
                    <p class="text-sm text-gray-500">{formatFileSize(selectedFile.size)} • {selectedFile.type}</p>
                  </div>
                  
                  <button
                    type="button"
                    click={() => { selectedFile = null; filePreview = null; metadata = null; }}
                    class="text-sm text-red-600 hover:text-red-800"
                  >
                    Remove file
                  </button>
                </div>
              {:else}
                <div class="space-y-4">
                  <svg class="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                    <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
                  </svg>
                  <div>
                    <p class="text-gray-600">Drag and drop your file here, or</p>
                    <label for="file" class="cursor-pointer">
                      <span class="text-blue-600 hover:text-blue-800 font-medium">click to browse</span>
                      <input
                        type="file"
                        name="file"
                        id="file"
                        class="sr-only"
                        disabled={$submitting}
                        change={onFileChange}
                      />
                    </label>
                  </div>
                  <p class="text-sm text-gray-500">
                    Maximum file size: 100MB
                  </p>
                </div>
              {/if}
            </div>
            
            {#if $errors.file}
              <p class="mt-1 text-sm text-red-600">{$errors.file}</p>
            {/if}
          </div>
        {/if}
        
        <!-- Link URL (for LINK type evidence) -->
        {#if $form.evidence_type === 'LINK'}
          <div>
            <label for="link_url" class="block text-sm font-medium text-gray-700 mb-2">
              URL *
            </label>
            <input
              type="url"
              name="link_url"
              id="link_url"
              required
              disabled={$submitting}
              bind:value={$form.link_url}
              class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              placeholder="https://example.com/document"
            />
            {#if $errors.link_url}
              <p class="mt-1 text-sm text-red-600">{$errors.link_url}</p>
            {/if}
          </div>
        {/if}
        
        <!-- Enhanced Evidence Fields -->
        <div class="space-y-4">
          <!-- Tags -->
          <div>
            <label for="tags" class="block text-sm font-medium text-gray-700 mb-2">
              Tags (comma-separated)
            </label>
            <input
              type="text"
              name="tags"
              id="tags"
              bind:value={$form.tags}
              disabled={$submitting}
              class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              placeholder="e.g., contract, confidential, priority"
            />
            {#if $errors.tags}
              <p class="mt-1 text-sm text-red-600">{$errors.tags}</p>
            {/if}
          </div>

          <!-- Confidentiality Level -->
          <div>
            <label for="confidentialityLevel" class="block text-sm font-medium text-gray-700 mb-2">
              Confidentiality Level
            </label>
            <select
              name="confidentialityLevel"
              id="confidentialityLevel"
              bind:value={$form.confidentialityLevel}
              disabled={$submitting}
              class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="public">Public</option>
              <option value="standard">Standard</option>
              <option value="confidential">Confidential</option>
              <option value="classified">Classified</option>
              <option value="restricted">Restricted</option>
            </select>
          </div>

          <!-- Chain of Custody Information -->
          <div>
            <label for="collectedBy" class="block text-sm font-medium text-gray-700 mb-2">
              Collected By
            </label>
            <input
              type="text"
              name="collectedBy"
              id="collectedBy"
              bind:value={$form.collectedBy}
              disabled={$submitting}
              class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              placeholder="Officer/person who collected the evidence"
            />
          </div>

          <div>
            <label for="location" class="block text-sm font-medium text-gray-700 mb-2">
              Collection Location
            </label>
            <input
              type="text"
              name="location"
              id="location"
              bind:value={$form.location}
              disabled={$submitting}
              class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              placeholder="Where the evidence was collected"
            />
          </div>

          <div>
            <label for="collectedAt" class="block text-sm font-medium text-gray-700 mb-2">
              Collection Date & Time
            </label>
            <input
              type="datetime-local"
              name="collectedAt"
              id="collectedAt"
              bind:value={$form.collectedAt}
              disabled={$submitting}
              class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>

          <!-- Evidence Admissibility -->
          <div class="flex items-center">
            <input
              type="checkbox"
              name="isAdmissible"
              id="isAdmissible"
              bind:checked={$form.isAdmissible}
              disabled={$submitting}
              class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
            />
            <label for="isAdmissible" class="ml-2 block text-sm text-gray-900">
              Evidence is admissible in court
            </label>
          </div>

          <!-- AI Processing Options -->
          <div class="border-t pt-4">
            <h3 class="text-sm font-medium text-gray-900 mb-3">AI Processing Options</h3>
            <div class="space-y-2">
              <div class="flex items-center">
                <input
                  type="checkbox"
                  name="enableOcr"
                  id="enableOcr"
                  bind:checked={$form.enableOcr}
                  disabled={$submitting}
                  class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <label for="enableOcr" class="ml-2 block text-sm text-gray-900">
                  Enable OCR (text extraction from PDFs and images)
                </label>
              </div>

              <div class="flex items-center">
                <input
                  type="checkbox"
                  name="enableAiAnalysis"
                  id="enableAiAnalysis"
                  bind:checked={$form.enableAiAnalysis}
                  disabled={$submitting}
                  class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <label for="enableAiAnalysis" class="ml-2 block text-sm text-gray-900">
                  Enable AI analysis and legal concept extraction
                </label>
              </div>

              <div class="flex items-center">
                <input
                  type="checkbox"
                  name="enableEmbeddings"
                  id="enableEmbeddings"
                  bind:checked={$form.enableEmbeddings}
                  disabled={$submitting}
                  class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <label for="enableEmbeddings" class="ml-2 block text-sm text-gray-900">
                  Generate vector embeddings for semantic search
                </label>
              </div>

              <div class="flex items-center">
                <input
                  type="checkbox"
                  name="enableSummarization"
                  id="enableSummarization"
                  bind:checked={$form.enableSummarization}
                  disabled={$submitting}
                  class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <label for="enableSummarization" class="ml-2 block text-sm text-gray-900">
                  Generate document summary
                </label>
              </div>
            </div>
          </div>
        </div>

        <!-- Metadata Preview -->
        {#if metadata}
          <div class="bg-gray-50 rounded-lg p-4">
            <h3 class="text-sm font-medium text-gray-700 mb-2">Detected Metadata</h3>
            <div class="text-sm text-gray-600">
              <pre class="whitespace-pre-wrap">{JSON.stringify(metadata, null, 2)}</pre>
            </div>
          </div>
        {/if}
        
        <!-- Submit Button -->
        <div style="text-align: center; margin: 20px 0;">
          <button
            type="button"
            click={() => history.back()}
            disabled={$submitting}
            class="nes-btn"
          >
            ← Cancel
          </button>
          
          <button
            type="submit"
            disabled={$submitting || (!selectedFile && $form.evidence_type !== 'LINK') || !$form.case_id || !$form.title}
            class="nes-btn is-success"
            style="margin-left: 10px;"
          >
            {#if $submitting}
              🔄 Uploading...
            {:else}
              📁 Upload Evidence
            {/if}
          </button>
        </div>
      </form>
    </div>
  </div>