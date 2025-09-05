<!-- Document Upload Page with MinIO Integration -->
<script lang="ts">
  import MinIOUpload from '$lib/components/upload/MinIOUpload.svelte';
  import { page } from '$app/stores';
  import { goto } from '$app/navigation';
  import type { PageData } from './$types';

  interface Props {
    data: PageData
  }

  let { data } = $props();

  // Extract case ID from URL params if provided
  const caseId = $page.url.searchParams.get('caseId') || '';

  // Upload completion handler
  function handleUploadComplete(result: any) {
    console.log('Upload completed:', result);
    
    // Show success notification
    const notification = {
      type: 'success',
      title: 'Upload Successful',
      message: `Document "${result.objectName}" has been uploaded and is being processed.`,
      documentId: result.documentId,
      url: result.url
    };
    
    // Store notification in session storage for display
    sessionStorage.setItem('uploadNotification', JSON.stringify(notification));
    
    // Redirect to document or case view
    if (caseId) {
      goto(`/cases/${caseId}/documents`);
    } else {
      goto('/documents');
    }
  }

  // Upload error handler
  function handleUploadError(error: string) {
    console.error('Upload error:', error);
    
    // Show error notification
    const notification = {
      type: 'error',
      title: 'Upload Failed',
      message: error
    };
    
    sessionStorage.setItem('uploadNotification', JSON.stringify(notification));
  }

  // Recent uploads state
  let showRecentUploads = $state(false);
  let recentUploads = $state<any[]>([]);

  // Load recent uploads
  async function loadRecentUploads() {
    try {
      const response = await fetch('/api/documents/recent');
      if (response.ok) {
        recentUploads = await response.json();
        showRecentUploads = true;
      }
    } catch (error) {
      console.error('Failed to load recent uploads:', error);
    }
  }
</script>

<svelte:head>
  <title>Upload Document - Legal AI Assistant</title>
  <meta name="description" content="Upload and process legal documents with AI analysis" />
</svelte:head>

<div class="upload-page">
  <div class="page-header">
    <h1>Upload Document</h1>
    <p class="page-description">
      Upload legal documents for AI-powered analysis, text extraction, and vector search indexing.
    </p>
  </div>

  <div class="upload-container">
    <!-- Main Upload Form -->
    <div class="upload-section">
      <MinIOUpload
        {data}
        {caseId}
        onUploadComplete={handleUploadComplete}
        onUploadError={handleUploadError}
      />
    </div>

    <!-- Sidebar with Information -->
    <div class="info-sidebar">
      <div class="info-card">
        <h3>📄 Supported Formats</h3>
        <ul>
          <li>PDF Documents</li>
          <li>Microsoft Word (.doc, .docx)</li>
          <li>Plain Text Files</li>
          <li>Images (JPEG, PNG, TIFF)</li>
        </ul>
      </div>

      <div class="info-card">
        <h3>🤖 AI Processing</h3>
        <p>Uploaded documents are automatically processed with:</p>
        <ul>
          <li>Text extraction and OCR</li>
          <li>Vector embeddings generation</li>
          <li>Semantic search indexing</li>
          <li>Entity recognition</li>
          <li>Document classification</li>
        </ul>
      </div>

      <div class="info-card">
        <h3>🔒 Security</h3>
        <ul>
          <li>End-to-end encryption</li>
          <li>Secure MinIO storage</li>
          <li>Access control by case</li>
          <li>Audit trail logging</li>
        </ul>
      </div>

      <!-- Recent Uploads -->
      <div class="info-card">
        <div class="card-header">
          <h3>📋 Recent Uploads</h3>
          <button
            type="button"
            onclick={loadRecentUploads}
            class="text-button"
          >
            {showRecentUploads ? 'Refresh' : 'Show'}
          </button>
        </div>

        {#if showRecentUploads}
          {#if recentUploads.length > 0}
            <div class="recent-uploads">
              {#each recentUploads.slice(0, 5) as upload}
                <div class="upload-item">
                  <div class="upload-icon">📄</div>
                  <div class="upload-details">
                    <div class="upload-name">{upload.filename}</div>
                    <div class="upload-meta">
                      {upload.documentType} • {upload.caseId}
                    </div>
                  </div>
                  <div class="upload-status">
                    {#if upload.processingStatus === 'completed'}
                      ✅
                    {:else if upload.processingStatus === 'processing'}
                      ⏳
                    {:else if upload.processingStatus === 'failed'}
                      ❌
                    {:else}
                      📤
                    {/if}
                  </div>
                </div>
              {/each}
            </div>
          {:else}
            <p class="no-uploads">No recent uploads found</p>
          {/if}
        {/if}
      </div>
    </div>
  </div>

  <!-- Help Section -->
  <div class="help-section">
    <h2>Need Help?</h2>
    <div class="help-grid">
      <div class="help-card">
        <h4>🚀 Quick Start</h4>
        <p>Select your case ID, choose your document type, and drag & drop your file to get started.</p>
      </div>
      
      <div class="help-card">
        <h4>📊 Processing Status</h4>
        <p>Track your document processing status and get notified when AI analysis is complete.</p>
      </div>
      
      <div class="help-card">
        <h4>🔍 Search Integration</h4>
        <p>Uploaded documents are automatically indexed for semantic search and similarity matching.</p>
      </div>
      
      <div class="help-card">
        <h4>💼 Case Management</h4>
        <p>Documents are organized by case ID for easy management and cross-referencing.</p>
      </div>
    </div>
  </div>
</div>

<style>
  .upload-page {
    max-width: 1400px;
    margin: 0 auto;
    padding: 2rem;
  }

  .page-header {
    text-align: center
    margin-bottom: 3rem;
  }

  .page-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
  }

  .page-description {
    font-size: 1.125rem;
    color: var(--text-secondary);
    max-width: 600px;
    margin: 0 auto;
  }

  .upload-container {
    display: grid
    grid-template-columns: 1fr 350px;
    gap: 3rem;
    margin-bottom: 4rem;
  }

  @media (max-width: 1024px) {
    .upload-container {
      grid-template-columns: 1fr;
      gap: 2rem;
    }
  }

  .upload-section {
    min-height: 600px;
  }

  .info-sidebar {
    display: flex
    flex-direction: column
    gap: 1.5rem;
  }

  .info-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.5rem;
  }

  .info-card h3 {
    margin: 0 0 1rem 0;
    color: var(--text-primary);
    font-size: 1.125rem;
  }

  .info-card ul {
    margin: 0;
    padding-left: 1.25rem;
    color: var(--text-secondary);
  }

  .info-card li {
    margin-bottom: 0.5rem;
  }

  .info-card p {
    margin: 0 0 1rem 0;
    color: var(--text-secondary);
  }

  .card-header {
    display: flex
    justify-content: space-between;
    align-items: center
    margin-bottom: 1rem;
  }

  .text-button {
    background: none
    border: none
    color: var(--accent-primary);
    cursor: pointer
    font-size: 0.875rem;
    text-decoration: underline
  }

  .text-button:hover {
    color: var(--accent-primary-dark);
  }

  .recent-uploads {
    display: flex
    flex-direction: column
    gap: 0.75rem;
  }

  .upload-item {
    display: flex
    align-items: center
    gap: 0.75rem;
    padding: 0.75rem;
    background: var(--bg-primary);
    border-radius: 6px;
    border: 1px solid var(--border-color);
  }

  .upload-icon {
    font-size: 1.25rem;
    opacity: 0.7;
  }

  .upload-details {
    flex: 1;
    min-width: 0;
  }

  .upload-name {
    font-weight: 500;
    font-size: 0.875rem;
    color: var(--text-primary);
    white-space: nowrap
    overflow: hidden
    text-overflow: ellipsis
  }

  .upload-meta {
    font-size: 0.75rem;
    color: var(--text-secondary);
    margin-top: 0.25rem;
  }

  .upload-status {
    font-size: 1rem;
  }

  .no-uploads {
    color: var(--text-secondary);
    font-style: italic
    text-align: center
    margin: 1rem 0;
  }

  .help-section {
    background: var(--bg-secondary);
    border-radius: 16px;
    padding: 3rem;
    border: 1px solid var(--border-color);
  }

  .help-section h2 {
    text-align: center
    margin: 0 0 2rem 0;
    color: var(--text-primary);
  }

  .help-grid {
    display: grid
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.5rem;
  }

  .help-card {
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 1.5rem;
  }

  .help-card h4 {
    margin: 0 0 0.75rem 0;
    color: var(--text-primary);
    font-size: 1rem;
  }

  .help-card p {
    margin: 0;
    color: var(--text-secondary);
    font-size: 0.875rem;
    line-height: 1.5;
  }
</style>