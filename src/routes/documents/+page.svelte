<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { goto } from '$app/navigation';
  import { enhance } from '$app/forms';
  import { writable } from 'svelte/store';
  
  // UI Components (assuming they exist based on the project structure)
  import Button from '$lib/components/ui/enhanced/Button.svelte';
  import Card from '$lib/components/ui/enhanced/Card.svelte';
  import Input from '$lib/components/ui/enhanced/Input.svelte';
  import Select from '$lib/components/ui/select/SelectStandard.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  import Modal from '$lib/components/ui/Modal.svelte';
  import LoadingSpinner from '$lib/components/LoadingSpinner.svelte';
  import SearchInput from '$lib/components/SearchInput.svelte';

  // Types
  interface Document {
    id: string;
    title: string;
    documentType: string;
    jurisdiction: string;
    practiceArea?: string;
    fileName?: string;
    fileSize?: number;
    processingStatus: 'pending' | 'processing' | 'completed' | 'error';
    isConfidential: boolean;
    hasEmbeddings: boolean;
    createdAt: string;
    updatedAt: string;
    associatedCases: Array<{
      caseId: string;
      caseTitle: string;
      caseNumber: string;
      relationship: string;
      importance: string;
    }>;
    caseCount: number;
  }

  interface SearchFilters {
    documentType?: string;
    jurisdiction?: string;
    practiceArea?: string;
    status?: string;
    isConfidential?: boolean;
    hasEmbeddings?: boolean;
    search?: string;
  }

  // State management
  let documents: Document[] = [];
  let filteredDocuments: Document[] = [];
  let loading = true;
  let error: string | null = null;
  let searchQuery = '';
  let isSearching = false;
  
  // Pagination
  let currentPage = 1;
  let totalPages = 1;
  let totalDocuments = 0;
  let pageSize = 20;
  
  // Filters
  let filters: SearchFilters = {};
  let availableFilters = {
    documentTypes: [],
    jurisdictions: [],
    practiceAreas: []
  };
  
  // Sorting
  let sortBy = 'updated';
  let sortOrder = 'desc';
  
  // Modals
  let showUploadModal = false;
  let showFilterModal = false;
  let showSearchModal = false;
  let selectedDocument: Document | null = null;
  
  // Upload state
  let uploadFiles: FileList | null = null;
  let uploadProgress = 0;
  let isUploading = false;

  // Reactive statements
  $: {
    // Update URL with current filters and pagination
    updateURL();
  }

  onMount(async () => {
    await loadDocuments();
    await loadFilterOptions();
  });

  // Load documents from API
  async function loadDocuments() {
    try {
      loading = true;
      error = null;

      const params = new URLSearchParams({
        limit: pageSize.toString(),
        offset: ((currentPage - 1) * pageSize).toString(),
        sortBy,
        sortOrder,
        includeContent: 'false',
        includeAnalysis: 'false',
      });

      // Add filters to params
      Object.entries(filters).forEach(([key, value]) => {
        if (value !== undefined && value !== null && value !== '') {
          params.append(key, value.toString());
        }
      });

      const response = await fetch(`/api/documents?${params}`);
      const data = await response.json();

      if (data.success) {
        documents = data.documents;
        totalDocuments = data.pagination.total;
        totalPages = data.pagination.totalPages;
      } else {
        error = data.error || 'Failed to load documents';
      }
    } catch (err) {
      error = 'Network error occurred';
      console.error('Load documents error:', err);
    } finally {
      loading = false;
    }
  }

  // Load available filter options
  async function loadFilterOptions() {
    try {
      const response = await fetch('/api/documents/search?type=filters');
      const data = await response.json();
      
      if (data.success) {
        availableFilters = data.filters;
      }
    } catch (err) {
      console.error('Load filter options error:', err);
    }
  }

  // Search functionality
  async function performSearch() {
    if (!searchQuery.trim()) {
      await loadDocuments();
      return;
    }

    try {
      isSearching = true;
      error = null;

      const searchData = {
        query: searchQuery,
        searchType: 'hybrid',
        limit: pageSize,
        offset: (currentPage - 1) * pageSize,
        includeContent: false,
        includeAnalysis: false,
        ...filters
      };

      const response = await fetch('/api/documents/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(searchData)
      });

      const data = await response.json();

      if (data.success) {
        documents = data.results;
        totalDocuments = data.pagination.total;
        totalPages = data.pagination.totalPages;
      } else {
        error = data.error || 'Search failed';
      }
    } catch (err) {
      error = 'Search error occurred';
      console.error('Search error:', err);
    } finally {
      isSearching = false;
    }
  }

  // Upload documents
  async function uploadDocuments() {
    if (!uploadFiles || uploadFiles.length === 0) return;

    try {
      isUploading = true;
      uploadProgress = 0;

      for (let i = 0; i < uploadFiles.length; i++) {
        const file = uploadFiles[i];
        const formData = new FormData();
        formData.append('file', file);
        formData.append('documentType', 'evidence'); // Default type
        formData.append('jurisdiction', 'federal');
        formData.append('includeEmbeddings', 'true');
        formData.append('generateAnalysis', 'true');

        const response = await fetch('/api/documents/upload', {
          method: 'POST',
          body: formData
        });

        const result = await response.json();
        
        if (!result.success) {
          throw new Error(result.error || 'Upload failed');
        }

        uploadProgress = ((i + 1) / uploadFiles.length) * 100;
      }

      // Reload documents after upload
      await loadDocuments();
      showUploadModal = false;
      uploadFiles = null;
      
    } catch (err) {
      error = err instanceof Error ? err.message : 'Upload failed';
      console.error('Upload error:', err);
    } finally {
      isUploading = false;
      uploadProgress = 0;
    }
  }

  // Pagination handlers
  function goToPage(page: number) {
    currentPage = page;
    loadDocuments();
  }

  function nextPage() {
    if (currentPage < totalPages) {
      currentPage++;
      loadDocuments();
    }
  }

  function previousPage() {
    if (currentPage > 1) {
      currentPage--;
      loadDocuments();
    }
  }

  // Filter handlers
  function applyFilters() {
    currentPage = 1; // Reset to first page
    if (searchQuery.trim()) {
      performSearch();
    } else {
      loadDocuments();
    }
    showFilterModal = false;
  }

  function clearFilters() {
    filters = {};
    searchQuery = '';
    currentPage = 1;
    loadDocuments();
    showFilterModal = false;
  }

  // Sort handlers
  function sortDocuments(field: string) {
    if (sortBy === field) {
      sortOrder = sortOrder === 'asc' ? 'desc' : 'asc';
    } else {
      sortBy = field;
      sortOrder = 'desc';
    }
    loadDocuments();
  }

  // URL management
  function updateURL() {
    const params = new URLSearchParams();
    
    if (currentPage > 1) params.set('page', currentPage.toString());
    if (sortBy !== 'updated') params.set('sortBy', sortBy);
    if (sortOrder !== 'desc') params.set('sortOrder', sortOrder);
    if (searchQuery) params.set('q', searchQuery);
    
    Object.entries(filters).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== '') {
        params.set(key, value.toString());
      }
    });

    const newUrl = `${window.location.pathname}${params.toString() ? '?' + params.toString() : ''}`;
    history.replaceState({}, '', newUrl);
  }

  // Utility functions
  function formatFileSize(bytes?: number): string {
    if (!bytes) return 'Unknown';
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  }

  function formatDate(dateString: string): string {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'completed': return 'bg-green-100 text-green-800';
      case 'processing': return 'bg-blue-100 text-blue-800';
      case 'pending': return 'bg-yellow-100 text-yellow-800';
      case 'error': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  }

  function getTypeColor(type: string): string {
    switch (type) {
      case 'contract': return 'bg-purple-100 text-purple-800';
      case 'motion': return 'bg-blue-100 text-blue-800';
      case 'evidence': return 'bg-orange-100 text-orange-800';
      case 'correspondence': return 'bg-gray-100 text-gray-800';
      case 'brief': return 'bg-indigo-100 text-indigo-800';
      case 'regulation': return 'bg-green-100 text-green-800';
      case 'case_law': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  }
</script>

<svelte:head>
  <title>Legal Documents - Document Management System</title>
  <meta name="description" content="Manage and search legal documents with AI-powered analysis and vector search capabilities" />
</svelte:head>

<div class="min-h-screen bg-gray-50 py-8">
  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
    <!-- Header -->
    <div class="mb-8">
      <div class="flex items-center justify-between">
        <div>
          <h1 class="text-3xl font-bold text-gray-900">Legal Documents</h1>
          <p class="mt-2 text-gray-600">
            Manage and search through {totalDocuments.toLocaleString()} legal documents
          </p>
        </div>
        
        <div class="flex space-x-4">
          <Button 
            variant="outline" 
            on:click={() => showFilterModal = true}
            class="flex items-center space-x-2"
          >
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.707A1 1 0 013 7V4z" />
            </svg>
            <span>Filters</span>
            {#if Object.keys(filters).length > 0}
              <Badge variant="secondary">{Object.keys(filters).length}</Badge>
            {/if}
          </Button>
          
          <Button 
            on:click={() => showUploadModal = true}
            class="flex items-center space-x-2"
          >
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
            <span>Upload Documents</span>
          </Button>
        </div>
      </div>
    </div>

    <!-- Search Bar -->
    <div class="mb-6">
      <div class="flex space-x-4">
        <div class="flex-1">
          <SearchInput
            bind:value={searchQuery}
            placeholder="Search documents by title, content, or metadata..."
            on:search={performSearch}
            loading={isSearching}
          />
        </div>
        
        <select
          bind:value={sortBy}
          on:change={loadDocuments}
          class="px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="updated">Last Updated</option>
          <option value="created">Date Created</option>
          <option value="title">Title</option>
          <option value="type">Document Type</option>
          <option value="size">File Size</option>
        </select>
        
        <button
          on:click={() => sortOrder = sortOrder === 'asc' ? 'desc' : 'asc'}
          class="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
          title="Toggle sort order"
        >
          {#if sortOrder === 'asc'}
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 15l7-7 7 7" />
            </svg>
          {:else}
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
            </svg>
          {/if}
        </button>
      </div>
    </div>

    <!-- Active Filters -->
    {#if Object.keys(filters).length > 0 || searchQuery}
      <div class="mb-6 flex flex-wrap items-center gap-2">
        <span class="text-sm text-gray-600">Active filters:</span>
        
        {#if searchQuery}
          <Badge variant="secondary" class="flex items-center space-x-1">
            <span>Search: "{searchQuery}"</span>
            <button on:click={() => { searchQuery = ''; loadDocuments(); }} class="ml-1 hover:text-red-600">
              <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </Badge>
        {/if}
        
        {#each Object.entries(filters) as [key, value]}
          {#if value !== undefined && value !== null && value !== ''}
            <Badge variant="secondary" class="flex items-center space-x-1">
              <span>{key}: {value}</span>
              <button on:click={() => { delete filters[key]; filters = filters; applyFilters(); }} class="ml-1 hover:text-red-600">
                <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </Badge>
          {/if}
        {/each}
        
        <Button variant="ghost" size="sm" on:click={clearFilters}>
          Clear all
        </Button>
      </div>
    {/if}

    <!-- Error Display -->
    {#if error}
      <div class="mb-6 bg-red-50 border border-red-200 rounded-lg p-4">
        <div class="flex">
          <svg class="w-5 h-5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <div class="ml-3">
            <h3 class="text-sm font-medium text-red-800">Error</h3>
            <p class="text-sm text-red-700">{error}</p>
          </div>
        </div>
      </div>
    {/if}

    <!-- Documents Grid -->
    {#if loading}
      <div class="flex justify-center items-center py-12">
        <LoadingSpinner size="lg" />
      </div>
    {:else if documents.length === 0}
      <div class="text-center py-12">
        <svg class="w-12 h-12 text-gray-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
        <h3 class="text-lg font-medium text-gray-900 mb-2">No documents found</h3>
        <p class="text-gray-600 mb-4">
          {searchQuery || Object.keys(filters).length > 0 
            ? 'Try adjusting your search or filters' 
            : 'Upload your first document to get started'}
        </p>
        <Button on:click={() => showUploadModal = true}>
          Upload Documents
        </Button>
      </div>
    {:else}
      <div class="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {#each documents as document (document.id)}
          <Card class="hover:shadow-lg transition-shadow cursor-pointer" on:click={() => goto(`/documents/${document.id}`)}>
            <div class="p-6">
              <!-- Document Header -->
              <div class="flex items-start justify-between mb-4">
                <div class="flex-1 min-w-0">
                  <h3 class="text-lg font-semibold text-gray-900 truncate" title={document.title}>
                    {document.title}
                  </h3>
                  {#if document.fileName}
                    <p class="text-sm text-gray-600 truncate" title={document.fileName}>
                      {document.fileName}
                    </p>
                  {/if}
                </div>
                
                {#if document.isConfidential}
                  <Badge variant="destructive" size="sm">Confidential</Badge>
                {/if}
              </div>

              <!-- Document Metadata -->
              <div class="space-y-2 mb-4">
                <div class="flex items-center justify-between">
                  <span class="text-sm text-gray-600">Type:</span>
                  <Badge class={getTypeColor(document.documentType)} size="sm">
                    {document.documentType.replace('_', ' ')}
                  </Badge>
                </div>
                
                <div class="flex items-center justify-between">
                  <span class="text-sm text-gray-600">Status:</span>
                  <Badge class={getStatusColor(document.processingStatus)} size="sm">
                    {document.processingStatus}
                  </Badge>
                </div>
                
                <div class="flex items-center justify-between">
                  <span class="text-sm text-gray-600">Jurisdiction:</span>
                  <span class="text-sm font-medium text-gray-900">{document.jurisdiction}</span>
                </div>
                
                {#if document.practiceArea}
                  <div class="flex items-center justify-between">
                    <span class="text-sm text-gray-600">Practice Area:</span>
                    <span class="text-sm font-medium text-gray-900">
                      {document.practiceArea.replace('_', ' ')}
                    </span>
                  </div>
                {/if}
                
                {#if document.fileSize}
                  <div class="flex items-center justify-between">
                    <span class="text-sm text-gray-600">Size:</span>
                    <span class="text-sm font-medium text-gray-900">
                      {formatFileSize(document.fileSize)}
                    </span>
                  </div>
                {/if}
              </div>

              <!-- Document Features -->
              <div class="flex items-center space-x-2 mb-4">
                {#if document.hasEmbeddings}
                  <Badge variant="secondary" size="sm" title="Vector embeddings available for semantic search">
                    <svg class="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                    </svg>
                    AI Indexed
                  </Badge>
                {/if}
                
                {#if document.caseCount > 0}
                  <Badge variant="outline" size="sm" title="Associated with {document.caseCount} case(s)">
                    <svg class="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                    </svg>
                    {document.caseCount} Case{document.caseCount !== 1 ? 's' : ''}
                  </Badge>
                {/if}
              </div>

              <!-- Timestamps -->
              <div class="text-xs text-gray-500 border-t pt-3">
                <div class="flex justify-between">
                  <span>Created: {formatDate(document.createdAt)}</span>
                  <span>Updated: {formatDate(document.updatedAt)}</span>
                </div>
              </div>
            </div>
          </Card>
        {/each}
      </div>

      <!-- Pagination -->
      {#if totalPages > 1}
        <div class="mt-8 flex items-center justify-between">
          <div class="text-sm text-gray-700">
            Showing {((currentPage - 1) * pageSize) + 1} to {Math.min(currentPage * pageSize, totalDocuments)} of {totalDocuments.toLocaleString()} documents
          </div>
          
          <div class="flex space-x-2">
            <Button 
              variant="outline" 
              disabled={currentPage === 1}
              on:click={previousPage}
            >
              Previous
            </Button>
            
            {#each Array.from({length: Math.min(totalPages, 5)}, (_, i) => {
              const page = Math.max(1, Math.min(totalPages - 4, currentPage - 2)) + i;
              return page;
            }) as pageNum}
              <Button
                variant={currentPage === pageNum ? "default" : "outline"}
                on:click={() => goToPage(pageNum)}
              >
                {pageNum}
              </Button>
            {/each}
            
            <Button 
              variant="outline" 
              disabled={currentPage === totalPages}
              on:click={nextPage}
            >
              Next
            </Button>
          </div>
        </div>
      {/if}
    {/if}
  </div>
</div>

<!-- Upload Modal -->
{#if showUploadModal}
  <Modal title="Upload Documents" on:close={() => showUploadModal = false}>
    <div class="space-y-6">
      <div>
        <label class="block text-sm font-medium text-gray-700 mb-2">
          Select files to upload
        </label>
        <input
          type="file"
          multiple
          accept=".pdf,.doc,.docx,.txt,.json"
          bind:files={uploadFiles}
          class="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
        />
        <p class="mt-2 text-sm text-gray-600">
          Supported formats: PDF, Word documents, Text files, JSON. Max 50MB per file.
        </p>
      </div>

      {#if isUploading}
        <div class="space-y-2">
          <div class="flex justify-between text-sm">
            <span>Uploading...</span>
            <span>{Math.round(uploadProgress)}%</span>
          </div>
          <div class="w-full bg-gray-200 rounded-full h-2">
            <div class="bg-blue-600 h-2 rounded-full transition-all duration-300" style="width: {uploadProgress}%"></div>
          </div>
        </div>
      {/if}

      <div class="flex justify-end space-x-4">
        <Button variant="outline" on:click={() => showUploadModal = false} disabled={isUploading}>
          Cancel
        </Button>
        <Button on:click={uploadDocuments} disabled={!uploadFiles || uploadFiles.length === 0 || isUploading}>
          {isUploading ? 'Uploading...' : 'Upload Documents'}
        </Button>
      </div>
    </div>
  </Modal>
{/if}

<!-- Filter Modal -->
{#if showFilterModal}
  <Modal title="Filter Documents" on:close={() => showFilterModal = false}>
    <div class="space-y-6">
      <div>
        <label class="block text-sm font-medium text-gray-700 mb-2">Document Type</label>
        <select bind:value={filters.documentType} class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500">
          <option value="">All types</option>
          {#each availableFilters.documentTypes as type}
            <option value={type}>{type.replace('_', ' ')}</option>
          {/each}
        </select>
      </div>

      <div>
        <label class="block text-sm font-medium text-gray-700 mb-2">Jurisdiction</label>
        <select bind:value={filters.jurisdiction} class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500">
          <option value="">All jurisdictions</option>
          {#each availableFilters.jurisdictions as jurisdiction}
            <option value={jurisdiction}>{jurisdiction}</option>
          {/each}
        </select>
      </div>

      <div>
        <label class="block text-sm font-medium text-gray-700 mb-2">Practice Area</label>
        <select bind:value={filters.practiceArea} class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500">
          <option value="">All practice areas</option>
          {#each availableFilters.practiceAreas as area}
            <option value={area}>{area.replace('_', ' ')}</option>
          {/each}
        </select>
      </div>

      <div>
        <label class="block text-sm font-medium text-gray-700 mb-2">Processing Status</label>
        <select bind:value={filters.status} class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500">
          <option value="">All statuses</option>
          <option value="completed">Completed</option>
          <option value="processing">Processing</option>
          <option value="pending">Pending</option>
          <option value="error">Error</option>
        </select>
      </div>

      <div class="flex items-center space-x-4">
        <label class="flex items-center space-x-2">
          <input type="checkbox" bind:checked={filters.isConfidential} class="rounded border-gray-300 text-blue-600 focus:ring-blue-500">
          <span class="text-sm text-gray-700">Confidential only</span>
        </label>

        <label class="flex items-center space-x-2">
          <input type="checkbox" bind:checked={filters.hasEmbeddings} class="rounded border-gray-300 text-blue-600 focus:ring-blue-500">
          <span class="text-sm text-gray-700">AI indexed only</span>
        </label>
      </div>

      <div class="flex justify-end space-x-4">
        <Button variant="outline" on:click={clearFilters}>Clear All</Button>
        <Button on:click={applyFilters}>Apply Filters</Button>
      </div>
    </div>
  </Modal>
{/if}