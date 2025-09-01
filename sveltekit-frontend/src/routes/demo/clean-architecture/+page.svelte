<script lang="ts">
  import { onMount } from 'svelte';
  import { browser } from '$app/environment';

  // Component state
  let cases = $state<any[]>([]);
  let selectedCase = $state<any>(null);
  let isLoading = $state(false);
  let error = $state<string | null>(null);
  let success = $state<string | null>(null);

  // Form states
  let showCreateForm = $state(false);
  let showUploadModal = $state(false);
  let createFormData = $state({
    title: '',
    description: '',
    priority: 'medium',
    status: 'active'
  });

  // Upload states
  let uploadFile = $state<File | null>(null);
  let uploadProgress = $state<number>(0);
  let uploadStatus = $state<string>('');
  let currentJobId = $state<string | null>(null);
  let jobStatus = $state<any>(null);

  // Search states
  let searchQuery = $state<string>('');
  let searchResults = $state<any[]>([]);
  let searchLoading = $state<boolean>(false);
  let showSearchModal = $state<boolean>(false);
  let ragResponse = $state<any>(null);
  let searchHistory = $state<any[]>([]);

  // Polling interval for job status
let statusPollingInterval = $state<number | null >(null);

  // Load cases on component mount
  onMount(async () => {
    await loadCases();
  });

  // Clear messages after timeout
  $effect(() => {
    if (success) {
      setTimeout(() => success = null, 5000);
    }
    if (error) {
      setTimeout(() => error = null, 5000);
    }
  });

  // Clean up polling on unmount
  onMount(() => {
    return () => {
      if (statusPollingInterval) {
        clearInterval(statusPollingInterval);
      }
    };
  });

  /**
   * Load cases from API
   */
  async function loadCases() {
    if (!browser) return;

    isLoading = true;
    error = null;

    try {
      const response = await fetch('/api/v1/cases');
      const result = await response.json();

      if (result.success) {
        cases = result.data.cases || [];
        console.log('Cases loaded:', cases.length);
      } else {
        throw new Error(result.error || 'Failed to load cases');
      }
    } catch (e) {
      error = `Failed to load cases: ${e instanceof Error ? e.message : 'Unknown error'}`;
      console.error('Load cases error:', e);
    } finally {
      isLoading = false;
    }
  }

  /**
   * Create new case
   */
  async function createCase() {
    if (!browser) return;
    
    isLoading = true;
    error = null;

    try {
      const response = await fetch('/api/v1/cases', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(createFormData)
      });

      const result = await response.json();

      if (result.success) {
        success = `Case "${createFormData.title}" created successfully`;
        showCreateForm = false;
        createFormData = {
          title: '',
          description: '',
          priority: 'medium',
          status: 'active'
        };
        await loadCases(); // Refresh cases list
      } else {
        throw new Error(result.error || 'Failed to create case');
      }
    } catch (e) {
      error = `Failed to create case: ${e instanceof Error ? e.message : 'Unknown error'}`;
      console.error('Create case error:', e);
    } finally {
      isLoading = false;
    }
  }

  /**
   * Load specific case details
   */
  async function loadCase(caseId: string) {
    if (!browser) return;

    isLoading = true;
    error = null;

    try {
      const response = await fetch(`/api/v1/cases/${caseId}`);
      const result = await response.json();

      if (result.success) {
        selectedCase = result.data.case;
        console.log('Case loaded:', selectedCase);
      } else {
        throw new Error(result.error || 'Failed to load case');
      }
    } catch (e) {
      error = `Failed to load case: ${e instanceof Error ? e.message : 'Unknown error'}`;
      console.error('Load case error:', e);
    } finally {
      isLoading = false;
    }
  }

  /**
   * Handle file upload with pre-signed URL
   */
  async function handleFileUpload() {
    if (!uploadFile || !selectedCase?.id) {
      error = 'Please select a file and case first';
      return;
    }

    isLoading = true;
    error = null;
    uploadProgress = 0;
    uploadStatus = 'Preparing upload...';

    try {
      // Step 1: Get pre-signed upload URL
      const presignedResponse = await fetch('/api/v1/upload/presigned', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          fileName: uploadFile.name,
          contentType: uploadFile.type,
          contentLength: uploadFile.size,
          caseId: selectedCase.id,
          evidenceType: uploadFile.type.startsWith('image/') ? 'image' : 'document'
        })
      });

      const presignedResult = await presignedResponse.json();

      if (!presignedResult.success) {
        throw new Error(presignedResult.error || 'Failed to get upload URL');
      }

      const { uploadId, presignedPost } = presignedResult.data;
      uploadStatus = 'Uploading to MinIO...';
      uploadProgress = 10;

      // Step 2: Upload directly to MinIO using pre-signed POST
      const formData = new FormData();
      
      // Add required fields from pre-signed POST
      Object.entries(presignedPost.fields).forEach(([key, value]) => {
        formData.append(key, value as string);
      });
      
      // Add the file (must be last)
      formData.append('file', uploadFile);

      const uploadResponse = await fetch(presignedPost.url, {
        method: 'POST',
        body: formData
      });

      if (!uploadResponse.ok) {
        throw new Error(`Upload failed: ${uploadResponse.statusText}`);
      }

      uploadStatus = 'Upload completed, processing...';
      uploadProgress = 50;
      currentJobId = uploadId;

      // Step 3: Start polling for job status
      startJobStatusPolling(uploadId);

      success = `File "${uploadFile.name}" uploaded successfully`;
      showUploadModal = false;
      uploadFile = null;

    } catch (e) {
      error = `Upload failed: ${e instanceof Error ? e.message : 'Unknown error'}`;
      console.error('Upload error:', e);
      uploadProgress = 0;
      uploadStatus = '';
    } finally {
      isLoading = false;
    }
  }

  /**
   * Start polling for job status
   */
  function startJobStatusPolling(jobId: string) {
    if (statusPollingInterval) {
      clearInterval(statusPollingInterval);
    }

    const pollStatus = async () => {
      try {
        const response = await fetch(`/api/v1/jobs/${jobId}/status`);
        const result = await response.json();

        if (result.success) {
          jobStatus = result.data.job;
          
          if (result.data.job.progress) {
            uploadProgress = result.data.job.progress.percentage;
            uploadStatus = result.data.job.progress.message;
          }

          // Stop polling if job is complete or failed
          if (['completed', 'failed'].includes(result.data.job.status)) {
            if (statusPollingInterval) {
              clearInterval(statusPollingInterval);
              statusPollingInterval = null;
            }
            
            if (result.data.job.status === 'completed') {
              uploadProgress = 100;
              uploadStatus = 'Processing completed';
              success = 'Document processed successfully';
              
              // Refresh case data to show new evidence
              if (selectedCase) {
                await loadCase(selectedCase.id);
              }
            } else {
              uploadProgress = 0;
              uploadStatus = 'Processing failed';
              error = result.data.job.error || 'Processing failed';
            }
          }
        }
      } catch (e) {
        console.error('Status polling error:', e);
      }
    };

    // Poll immediately, then every 2 seconds
    pollStatus();
    statusPollingInterval = setInterval(pollStatus, 2000);
  }

  /**
   * Handle file selection
   */
  function handleFileSelect(event: Event) {
    const target = event.target as HTMLInputElement;
    if (target.files && target.files[0]) {
      uploadFile = target.files[0];
    }
  }

  /**
   * Format file size for display
   */
  function formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  /**
   * Format date for display
   */
  function formatDate(dateString: string): string {
    return new Date(dateString).toLocaleDateString() + ' ' + new Date(dateString).toLocaleTimeString();
  }

  /**
   * Perform semantic search within selected case
   */
  async function performSearch() {
    if (!browser || !selectedCase?.id || !searchQuery.trim()) {
      error = 'Please select a case and enter a search query';
      return;
    }

    searchLoading = true;
    error = null;

    try {
      const response = await fetch(`/api/v1/cases/${selectedCase.id}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query: searchQuery.trim(),
          limit: 10,
          threshold: 0.7,
          includeRAG: true
        })
      });

      const result = await response.json();

      if (result.success) {
        searchResults = result.data.results || [];
        ragResponse = result.data.rag;
        console.log('Search completed:', searchResults.length, 'results');
        
        if (searchResults.length === 0) {
          error = 'No relevant evidence found for your search query';
        } else {
          success = `Found ${searchResults.length} relevant pieces of evidence`;
        }
      } else {
        throw new Error(result.error || 'Search failed');
      }
    } catch (e) {
      error = `Search failed: ${e instanceof Error ? e.message : 'Unknown error'}`;
      console.error('Search error:', e);
      searchResults = [];
      ragResponse = null;
    } finally {
      searchLoading = false;
    }
  }

  /**
   * Load search history for selected case
   */
  async function loadSearchHistory() {
    if (!browser || !selectedCase?.id) return;

    try {
      const response = await fetch(`/api/v1/cases/${selectedCase.id}/search/history`);
      const result = await response.json();

      if (result.success) {
        searchHistory = result.data.history || [];
      }
    } catch (e) {
      console.error('Failed to load search history:', e);
    }
  }

  /**
   * Clear search results
   */
  function clearSearch() {
    searchQuery = '';
    searchResults = [];
    ragResponse = null;
    error = null;
  }

  /**
   * Get similarity percentage for display
   */
  function getSimilarityPercentage(similarity: number): string {
    return Math.round(similarity * 100) + '%';
  }

  /**
   * Get similarity color based on score
   */
  function getSimilarityColor(similarity: number): string {
    if (similarity >= 0.9) return 'text-green-600 bg-green-50';
    if (similarity >= 0.8) return 'text-blue-600 bg-blue-50';
    if (similarity >= 0.7) return 'text-yellow-600 bg-yellow-50';
    return 'text-gray-600 bg-gray-50';
  }
</script>

<div class="clean-architecture-demo p-6 max-w-7xl mx-auto">
  <div class="mb-8">
    <h1 class="text-3xl font-bold text-gray-900 mb-2">
      Clean Architecture Demo
    </h1>
    <p class="text-gray-600">
      SvelteKit → Drizzle → PostgreSQL/pgvector → MinIO → Redis → RAG middleware
    </p>
  </div>

  <!-- Status Messages -->
  {#if success}
    <div class="alert alert-success mb-4 p-4 bg-green-50 border border-green-200 rounded-lg">
      <div class="flex items-center">
        <svg class="w-5 h-5 text-green-400 mr-2" fill="currentColor" viewBox="0 0 20 20">
          <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/>
        </svg>
        <span class="text-green-700">{success}</span>
      </div>
    </div>
  {/if}

  {#if error}
    <div class="alert alert-error mb-4 p-4 bg-red-50 border border-red-200 rounded-lg">
      <div class="flex items-center">
        <svg class="w-5 h-5 text-red-400 mr-2" fill="currentColor" viewBox="0 0 20 20">
          <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"/>
        </svg>
        <span class="text-red-700">{error}</span>
      </div>
    </div>
  {/if}

  <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
    <!-- Cases List -->
    <div class="cases-section">
      <div class="flex justify-between items-center mb-4">
        <h2 class="text-xl font-semibold text-gray-900">Cases</h2>
        <button
          class="btn btn-primary px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          on:onclick={() => showCreateForm = true}
          disabled={isLoading}
        >
          Create Case
        </button>
      </div>

      {#if isLoading && cases.length === 0}
        <div class="loading flex items-center justify-center p-8">
          <svg class="animate-spin h-8 w-8 text-blue-600 mr-2" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
            <path class="opacity-75" fill="currentColor" d="m4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
          </svg>
          <span>Loading cases...</span>
        </div>
      {:else}
        <div class="cases-list space-y-3">
          {#each cases as caseItem (caseItem.id)}
            <div
              class="case-card p-4 border border-gray-200 rounded-lg hover:border-blue-300 cursor-pointer transition-colors"
              class:selected={selectedCase?.id === caseItem.id}
              on:onclick={() => loadCase(caseItem.id)}
            >
              <div class="flex justify-between items-start mb-2">
                <h3 class="font-medium text-gray-900">{caseItem.title}</h3>
                <div class="flex items-center space-x-2">
                  <span class="status-badge px-2 py-1 text-xs rounded-full"
                        class:bg-green-100={caseItem.status === 'active'}
                        class:text-green-800={caseItem.status === 'active'}
                        class:bg-yellow-100={caseItem.status === 'pending'}
                        class:text-yellow-800={caseItem.status === 'pending'}
                        class:bg-gray-100={caseItem.status === 'archived'}
                        class:text-gray-800={caseItem.status === 'archived'}>
                    {caseItem.status}
                  </span>
                  <span class="priority-badge px-2 py-1 text-xs rounded-full"
                        class:bg-red-100={caseItem.priority === 'high'}
                        class:text-red-800={caseItem.priority === 'high'}
                        class:bg-yellow-100={caseItem.priority === 'medium'}
                        class:text-yellow-800={caseItem.priority === 'medium'}
                        class:bg-blue-100={caseItem.priority === 'low'}
                        class:text-blue-800={caseItem.priority === 'low'}>
                    {caseItem.priority}
                  </span>
                </div>
              </div>
              <p class="text-sm text-gray-600 mb-2 line-clamp-2">{caseItem.description}</p>
              <div class="text-xs text-gray-500">
                Created: {formatDate(caseItem.createdAt)}
                {#if caseItem.caseNumber}
                  • #{caseItem.caseNumber}
                {/if}
              </div>
            </div>
          {:else}
            <div class="empty-state text-center py-8">
              <svg class="mx-auto h-12 w-12 text-gray-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
              </svg>
              <h3 class="text-sm font-medium text-gray-900 mb-1">No cases found</h3>
              <p class="text-sm text-gray-500">Get started by creating your first case.</p>
            </div>
          {/each}
        </div>
      {/if}
    </div>

    <!-- Case Details & Upload -->
    <div class="case-details-section">
      {#if selectedCase}
        <div class="case-details mb-6">
          <div class="flex justify-between items-start mb-4">
            <h2 class="text-xl font-semibold text-gray-900">{selectedCase.title}</h2>
            <div class="flex space-x-2">
              <button
                class="btn btn-search px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
                on:onclick={() => { showSearchModal = true; loadSearchHistory(); }}
                disabled={isLoading}
              >
                Search Evidence
              </button>
              <button
                class="btn btn-secondary px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
                on:onclick={() => showUploadModal = true}
                disabled={isLoading}
              >
                Upload Evidence
              </button>
            </div>
          </div>

          <div class="case-info bg-gray-50 p-4 rounded-lg mb-4">
            <div class="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span class="font-medium">Status:</span> {selectedCase.status}
              </div>
              <div>
                <span class="font-medium">Priority:</span> {selectedCase.priority}
              </div>
              <div>
                <span class="font-medium">Case Number:</span> {selectedCase.caseNumber || 'N/A'}
              </div>
              <div>
                <span class="font-medium">Created:</span> {formatDate(selectedCase.createdAt)}
              </div>
            </div>
            <div class="mt-3">
              <span class="font-medium">Description:</span>
              <p class="mt-1 text-gray-700">{selectedCase.description}</p>
            </div>
          </div>

          {#if jobStatus}
            <div class="job-status bg-blue-50 p-4 rounded-lg mb-4">
              <h3 class="font-medium text-blue-900 mb-2">Processing Status</h3>
              <div class="mb-2">
                <div class="flex justify-between text-sm">
                  <span>Status: {jobStatus.status}</span>
                  <span>{uploadProgress}%</span>
                </div>
                <div class="w-full bg-blue-200 rounded-full h-2 mt-1">
                  <div 
                    class="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style="width: {uploadProgress}%"
                  ></div>
                </div>
              </div>
              {#if uploadStatus}
                <p class="text-sm text-blue-700">{uploadStatus}</p>
              {/if}
              {#if jobStatus.results}
                <div class="mt-3 text-sm">
                  <div class="font-medium mb-1">Processing Results:</div>
                  <ul class="list-disc list-inside text-blue-700 space-y-1">
                    {#if jobStatus.results.textExtracted}
                      <li>Text extracted successfully</li>
                    {/if}
                    {#if jobStatus.results.vectorsGenerated}
                      <li>Vector embeddings generated</li>
                    {/if}
                    {#if jobStatus.results.metadataExtracted}
                      <li>Metadata extracted</li>
                    {/if}
                  </ul>
                </div>
              {/if}
            </div>
          {/if}
        </div>
      {:else}
        <div class="empty-case text-center py-16">
          <svg class="mx-auto h-16 w-16 text-gray-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"/>
          </svg>
          <h3 class="text-lg font-medium text-gray-900 mb-2">Select a Case</h3>
          <p class="text-gray-500">Choose a case from the list to view details and upload evidence.</p>
        </div>
      {/if}
    </div>
  </div>
</div>

<!-- Create Case Modal -->
{#if showCreateForm}
  <div class="modal fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
    <div class="modal-content bg-white rounded-lg p-6 w-full max-w-md">
      <h2 class="text-xl font-semibold mb-4">Create New Case</h2>
      
      <form submit={async (e) => { e.preventDefault(); await createCase(); }}>
        <div class="form-group mb-4">
          <label class="block text-sm font-medium text-gray-700 mb-2">
            Case Title *
          </label>
          <input
            type="text"
            bind:value={createFormData.title}
            required
            class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            placeholder="Enter case title"
          />
        </div>

        <div class="form-group mb-4">
          <label class="block text-sm font-medium text-gray-700 mb-2">
            Description *
          </label>
          <textarea
            bind:value={createFormData.description}
            required
            rows="3"
            class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            placeholder="Describe the case..."
          ></textarea>
        </div>

        <div class="grid grid-cols-2 gap-4 mb-6">
          <div class="form-group">
            <label class="block text-sm font-medium text-gray-700 mb-2">
              Priority
            </label>
            <select 
              bind:value={createFormData.priority}
              class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
              <option value="urgent">Urgent</option>
            </select>
          </div>

          <div class="form-group">
            <label class="block text-sm font-medium text-gray-700 mb-2">
              Status
            </label>
            <select 
              bind:value={createFormData.status}
              class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="active">Active</option>
              <option value="pending">Pending</option>
              <option value="closed">Closed</option>
            </select>
          </div>
        </div>

        <div class="flex justify-end space-x-3">
          <button
            type="button"
            on:onclick={() => showCreateForm = false}
            class="px-4 py-2 text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200"
            disabled={isLoading}
          >
            Cancel
          </button>
          <button
            type="submit"
            class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
            disabled={isLoading}
          >
            {isLoading ? 'Creating...' : 'Create Case'}
          </button>
        </div>
      </form>
    </div>
  </div>
{/if}

<!-- Upload Modal -->
{#if showUploadModal}
  <div class="modal fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
    <div class="modal-content bg-white rounded-lg p-6 w-full max-w-md">
      <h2 class="text-xl font-semibold mb-4">Upload Evidence</h2>
      
      <form submit={async (e) => { e.preventDefault(); await handleFileUpload(); }}>
        <div class="form-group mb-4">
          <label class="block text-sm font-medium text-gray-700 mb-2">
            Select File *
          </label>
          <input
            type="file"
            change={handleFileSelect}
            required
            class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            accept=".pdf,.doc,.docx,.txt,.jpg,.jpeg,.png,.gif,.mp4,.avi,.zip"
          />
          {#if uploadFile}
            <div class="mt-2 text-sm text-gray-600">
              Selected: {uploadFile.name} ({formatFileSize(uploadFile.size)})
            </div>
          {/if}
        </div>

        {#if uploadProgress > 0}
          <div class="progress mb-4">
            <div class="flex justify-between text-sm mb-1">
              <span>Upload Progress</span>
              <span>{uploadProgress}%</span>
            </div>
            <div class="w-full bg-gray-200 rounded-full h-2">
              <div 
                class="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style="width: {uploadProgress}%"
              ></div>
            </div>
            {#if uploadStatus}
              <p class="text-sm text-gray-600 mt-1">{uploadStatus}</p>
            {/if}
          </div>
        {/if}

        <div class="flex justify-end space-x-3">
          <button
            type="button"
            on:onclick={() => { showUploadModal = false; uploadFile = null; uploadProgress = 0; }}
            class="px-4 py-2 text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200"
            disabled={isLoading}
          >
            Cancel
          </button>
          <button
            type="submit"
            class="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50"
            disabled={isLoading || !uploadFile}
          >
            {isLoading ? 'Uploading...' : 'Upload File'}
          </button>
        </div>
      </form>
    </div>
  </div>
{/if}

<!-- Search Evidence Modal -->
{#if showSearchModal}
  <div class="modal fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
    <div class="modal-content bg-white rounded-lg p-6 w-full max-w-4xl max-h-[90vh] overflow-y-auto">
      <div class="flex justify-between items-center mb-6">
        <h2 class="text-2xl font-semibold">Search Evidence - {selectedCase?.title}</h2>
        <button 
          on:onclick={() => { showSearchModal = false; clearSearch(); }}
          class="text-gray-500 hover:text-black text-2xl"
        >
          &times;
        </button>
      </div>

      <!-- Search Form -->
      <form submit={async (e) => { e.preventDefault(); await performSearch(); }}>
        <div class="flex gap-4 mb-6">
          <div class="flex-1">
            <input
              type="text"
              bind:value={searchQuery}
              placeholder="Enter your search query (e.g., 'What evidence supports the timeline?')"
              class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
              required
            />
          </div>
          <button
            type="submit"
            class="px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50"
            disabled={searchLoading || !searchQuery.trim()}
          >
            {#if searchLoading}
              <svg class="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
                <path class="opacity-75" fill="currentColor" d="m4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
              </svg>
            {:else}
              Search
            {/if}
          </button>
          {#if searchResults.length > 0}
            <button
              type="button"
              on:onclick={clearSearch}
              class="px-4 py-3 text-gray-600 bg-gray-100 rounded-lg hover:bg-gray-200"
            >
              Clear
            </button>
          {/if}
        </div>
      </form>

      <!-- RAG Response -->
      {#if ragResponse}
        <div class="rag-response bg-blue-50 border border-blue-200 rounded-lg p-6 mb-6">
          <h3 class="font-semibold text-blue-900 mb-3 flex items-center">
            <svg class="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-3a1 1 0 00-.867.5 1 1 0 11-1.731-1A3 3 0 0113 8a3.001 3.001 0 01-2 2.83V11a1 1 0 11-2 0v-1a1 1 0 011-1 1 1 0 100-2zm0 8a1 1 0 100-2 1 1 0 000 2z" clip-rule="evenodd"/>
            </svg>
            AI Analysis (Confidence: {Math.round(ragResponse.confidence * 100)}%)
          </h3>
          <div class="prose text-blue-800">
            {ragResponse.answer}
          </div>
          <div class="mt-3 text-sm text-blue-600">
            Model: {ragResponse.model} • {ragResponse.sources?.length || 0} sources
          </div>
        </div>
      {/if}

      <!-- Search Results -->
      {#if searchResults.length > 0}
        <div class="search-results">
          <h3 class="font-semibold text-gray-900 mb-4">
            Found {searchResults.length} relevant evidence items
          </h3>
          <div class="space-y-4">
            {#each searchResults as result, index (result.id)}
              <div class="result-card border border-gray-200 rounded-lg p-4">
                <div class="flex justify-between items-start mb-2">
                  <h4 class="font-medium text-gray-900">{result.title}</h4>
                  <div class="flex items-center space-x-2">
                    <span class="evidence-type px-2 py-1 text-xs rounded-full bg-gray-100 text-gray-800">
                      {result.evidenceType}
                    </span>
                    <span class="similarity-score px-2 py-1 text-xs rounded-full {getSimilarityColor(result.similarity)}">
                      {getSimilarityPercentage(result.similarity)} match
                    </span>
                  </div>
                </div>
                
                <div class="content mb-3">
                  <p class="text-sm text-gray-700 line-clamp-4">
                    {result.content.substring(0, 300)}...
                  </p>
                </div>
                
                <div class="metadata text-xs text-gray-500 flex justify-between">
                  <span>
                    Created: {formatDate(result.createdAt)}
                    {#if result.filePath}
                      • File: {result.filePath.split('/').pop()}
                    {/if}
                  </span>
                  <span>
                    Source #{index + 1}
                  </span>
                </div>
              </div>
            {:else}
              <div class="empty-results text-center py-8">
                <svg class="mx-auto h-12 w-12 text-gray-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
                </svg>
                <h3 class="text-sm font-medium text-gray-900 mb-1">No results found</h3>
                <p class="text-sm text-gray-500">Try adjusting your search query or upload more evidence.</p>
              </div>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Search History -->
      {#if searchHistory.length > 0 && searchResults.length === 0}
        <div class="search-history">
          <h3 class="font-semibold text-gray-900 mb-4">Recent Searches</h3>
          <div class="space-y-2">
            {#each searchHistory.slice(0, 5) as historyItem (historyItem.timestamp)}
              <button
                class="w-full text-left p-3 bg-gray-50 hover:bg-gray-100 rounded-lg transition-colors"
                on:onclick={() => { searchQuery = historyItem.query; performSearch(); }}
              >
                <div class="font-medium text-sm text-gray-900">{historyItem.query}</div>
                <div class="text-xs text-gray-500 mt-1">
                  {historyItem.resultCount} results • {formatDate(new Date(historyItem.timestamp).toISOString())}
                </div>
              </button>
            {/each}
          </div>
        </div>
      {/if}
    </div>
  </div>
{/if}

<style>
  .selected {
    @apply border-blue-300 bg-blue-50;
  }

  .line-clamp-2 {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  .modal {
    backdrop-filter: blur(4px);
  }

  .loading svg {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
</style>