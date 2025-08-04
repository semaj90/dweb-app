<script lang="ts">
  import { page } from '$app/stores';
  import { goto } from '$app/navigation';
  import { Button } from 'bits-ui';
  import { Card } from 'bits-ui';
  import { Tabs } from 'bits-ui';
  import { Badge } from 'bits-ui';
  import { Separator } from 'bits-ui';
  import DocumentUploader from '$lib/components/DocumentUploader.svelte';
  import VectorSearchInterface from '$lib/components/VectorSearchInterface.svelte';
  import AISummarization from '$lib/components/AISummarization.svelte';
  import { fly, fade } from 'svelte/transition';
  import type { LegalDocument } from '$lib/database/schema/legal-documents.js';
  import type { ProcessingResult } from '$lib/ai/processing-pipeline.js';

  // State
  let activeTab = $state('upload');
  let selectedDocument = $state<LegalDocument | null>(null);
  let recentUploads = $state<Array<{
    id: string;
    filename: string;
    status: string;
    uploadedAt: Date;
    documentId?: string;
  }>>([]);
  let searchResults = $state<Array<LegalDocument & { similarity: number }>>([]);
  let workspaceStats = $state({
    totalDocuments: 0,
    totalProcessed: 0,
    totalSearches: 0,
    avgProcessingTime: 0
  });

  // Reactive derived values
  let hasDocuments = $derived(workspaceStats.totalDocuments > 0);
  let hasSearchResults = $derived(searchResults.length > 0);

  /**
   * Handle document upload completion
   */
  function handleUploadComplete(event: CustomEvent<{ fileId: string; result: ProcessingResult }>) {
    const { fileId, result } = event.detail;
    
    if (result.status === 'completed' && result.result) {
      // Add to recent uploads
      recentUploads = [
        {
          id: fileId,
          filename: result.result.document.fileName || 'Unknown',
          status: 'completed',
          uploadedAt: new Date(),
          documentId: result.result.documentId
        },
        ...recentUploads.slice(0, 9) // Keep last 10
      ];

      // Update stats
      workspaceStats.totalDocuments++;
      workspaceStats.totalProcessed++;
      workspaceStats.avgProcessingTime = (
        (workspaceStats.avgProcessingTime * (workspaceStats.totalProcessed - 1) + result.metadata.processingTime) / 
        workspaceStats.totalProcessed
      );

      // Auto-switch to search tab after successful upload
      setTimeout(() => {
        activeTab = 'search';
      }, 2000);
    }
  }

  /**
   * Handle search completion
   */
  function handleSearchComplete(event: CustomEvent<{ 
    query: string; 
    results: Array<LegalDocument & { similarity: number }> 
  }>) {
    const { results } = event.detail;
    searchResults = results;
    workspaceStats.totalSearches++;
  }

  /**
   * Handle document selection from search
   */
  function handleDocumentSelect(event: CustomEvent<{ document: LegalDocument }>) {
    const { document } = event.detail;
    selectedDocument = document;
    activeTab = 'analyze';
  }

  /**
   * Load recent document for analysis
   */
  async function loadDocumentForAnalysis(documentId: string) {
    try {
      const response = await fetch(`/api/documents/${documentId}`);
      if (response.ok) {
        const document = await response.json();
        selectedDocument = document;
        activeTab = 'analyze';
      }
    } catch (error) {
      console.error('Failed to load document:', error);
    }
  }

  /**
   * Clear selected document
   */
  function clearSelectedDocument() {
    selectedDocument = null;
  }

  /**
   * Format processing time
   */
  function formatProcessingTime(ms: number): string {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}m`;
  }

  /**
   * Format file size
   */
  function formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  /**
   * Get status color
   */
  function getStatusColor(status: string): string {
    switch (status) {
      case 'completed': return 'bg-green-100 text-green-800';
      case 'processing': return 'bg-blue-100 text-blue-800';
      case 'error': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  }
</script>

<svelte:head>
  <title>AI Legal Workspace - Document Analysis & Search</title>
  <meta name="description" content="AI-powered legal document analysis workspace with upload, search, and summarization capabilities" />
</svelte:head>

<div class="ai-workspace min-h-screen bg-gray-50">
  <!-- Header -->
  <header class="bg-white shadow-sm border-b">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex items-center justify-between h-16">
        <div class="flex items-center space-x-4">
          <div class="flex items-center space-x-2">
            <div class="h-8 w-8 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-lg flex items-center justify-center">
              <svg class="h-5 w-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v14m-6.364-9.364l12.728 12.728" />
              </svg>
            </div>
            <h1 class="text-xl font-bold text-gray-900">AI Legal Workspace</h1>
          </div>
        </div>

        <!-- Stats Overview -->
        <div class="flex items-center space-x-6 text-sm text-gray-600">
          <div class="flex items-center space-x-1">
            <svg class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6-4h6m2 5.291A7.962 7.962 0 0112 15c-2.34 0-4.462-.881-6.065-2.325" />
            </svg>
            <span>{workspaceStats.totalDocuments} documents</span>
          </div>
          <div class="flex items-center space-x-1">
            <svg class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <span>{workspaceStats.totalSearches} searches</span>
          </div>
          {#if workspaceStats.totalProcessed > 0}
            <div class="flex items-center space-x-1">
              <svg class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span>avg {formatProcessingTime(workspaceStats.avgProcessingTime)}</span>
            </div>
          {/if}
        </div>
      </div>
    </div>
  </header>

  <!-- Main Content -->
  <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
    <Tabs.Root bind:value={activeTab} class="space-y-6">
      <!-- Tab Navigation -->
      <Tabs.List class="grid w-full grid-cols-4 lg:w-2/3 mx-auto">
        <Tabs.Trigger value="upload" class="flex items-center space-x-2">
          <svg class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
          <span>Upload</span>
        </Tabs.Trigger>
        
        <Tabs.Trigger value="search" class="flex items-center space-x-2">
          <svg class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <span>Search</span>
        </Tabs.Trigger>
        
        <Tabs.Trigger value="analyze" class="flex items-center space-x-2">
          <svg class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          <span>Analyze</span>
        </Tabs.Trigger>
        
        <Tabs.Trigger value="history" class="flex items-center space-x-2">
          <svg class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span>History</span>
        </Tabs.Trigger>
      </Tabs.List>

      <!-- Upload Tab -->
      <Tabs.Content value="upload">
        <div class="space-y-6">
          <div class="text-center">
            <h2 class="text-3xl font-bold text-gray-900 mb-2">
              Upload Legal Documents
            </h2>
            <p class="text-lg text-gray-600 max-w-2xl mx-auto">
              Upload PDF, DOCX, or text files for AI-powered analysis including summarization, 
              entity extraction, risk assessment, and semantic search indexing.
            </p>
          </div>

          <DocumentUploader
            accept=".pdf,.docx,.txt,.json"
            maxSize={50 * 1024 * 1024}
            multiple={true}
            on:complete={handleUploadComplete}
            class="max-w-4xl mx-auto"
          />

          <!-- Recent Uploads Quick Access -->
          {#if recentUploads.length > 0}
            <Card.Root class="max-w-4xl mx-auto">
              <div class="p-6">
                <h3 class="text-lg font-semibold text-gray-900 mb-4">Recent Uploads</h3>
                <div class="space-y-3">
                  {#each recentUploads.slice(0, 5) as upload}
                    <div class="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div class="flex items-center space-x-3">
                        <svg class="h-5 w-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6-4h6m2 5.291A7.962 7.962 0 0112 15c-2.34 0-4.462-.881-6.065-2.325" />
                        </svg>
                        <div>
                          <p class="font-medium text-gray-900">{upload.filename}</p>
                          <p class="text-sm text-gray-500">
                            {upload.uploadedAt.toLocaleTimeString()}
                          </p>
                        </div>
                      </div>
                      <div class="flex items-center space-x-2">
                        <Badge.Root class={getStatusColor(upload.status)}>
                          {upload.status}
                        </Badge.Root>
                        {#if upload.documentId}
                          <Button.Root
                            size="sm"
                            variant="outline"
                            on:click={() => loadDocumentForAnalysis(upload.documentId!)}
                          >
                            Analyze
                          </Button.Root>
                        {/if}
                      </div>
                    </div>
                  {/each}
                </div>
              </div>
            </Card.Root>
          {/if}
        </div>
      </Tabs.Content>

      <!-- Search Tab -->
      <Tabs.Content value="search">
        <div class="space-y-6">
          <div class="text-center">
            <h2 class="text-3xl font-bold text-gray-900 mb-2">
              Semantic Document Search
            </h2>
            <p class="text-lg text-gray-600 max-w-2xl mx-auto">
              Search through your legal documents using AI-powered semantic understanding. 
              Find relevant documents even when they don't contain exact keyword matches.
            </p>
          </div>

          <VectorSearchInterface
            on:search={handleSearchComplete}
            on:select={handleDocumentSelect}
            maxResults={20}
            showFilters={true}
          />
        </div>
      </Tabs.Content>

      <!-- Analyze Tab -->
      <Tabs.Content value="analyze">
        <div class="space-y-6">
          <div class="text-center">
            <h2 class="text-3xl font-bold text-gray-900 mb-2">
              AI Document Analysis
            </h2>
            <p class="text-lg text-gray-600 max-w-2xl mx-auto">
              Get comprehensive AI-powered analysis including summaries, entity extraction, 
              risk assessment, and custom insights for your legal documents.
            </p>
          </div>

          {#if selectedDocument}
            <div transition:fade>
              <!-- Document Info -->
              <Card.Root class="max-w-5xl mx-auto mb-6">
                <div class="p-6 bg-gradient-to-r from-blue-50 to-indigo-50 border-b">
                  <div class="flex items-center justify-between">
                    <div>
                      <h3 class="text-xl font-bold text-gray-900">{selectedDocument.title}</h3>
                      <div class="flex items-center space-x-4 mt-2 text-sm text-gray-600">
                        <span>Type: {selectedDocument.documentType}</span>
                        {#if selectedDocument.practiceArea}
                          <span>•</span>
                          <span>Area: {selectedDocument.practiceArea.replace('_', ' ')}</span>
                        {/if}
                        <span>•</span>
                        <span>Jurisdiction: {selectedDocument.jurisdiction}</span>
                        {#if selectedDocument.fileSize}
                          <span>•</span>
                          <span>Size: {formatFileSize(selectedDocument.fileSize)}</span>
                        {/if}
                      </div>
                    </div>
                    <Button.Root
                      variant="outline"
                      on:click={clearSelectedDocument}
                    >
                      Close Analysis
                    </Button.Root>
                  </div>
                </div>
              </Card.Root>

              <!-- AI Analysis Component -->
              <AISummarization
                documentContent={selectedDocument.content}
                documentTitle={selectedDocument.title}
                documentId={selectedDocument.id}
                autoSummarize={true}
                showAnalysisTools={true}
                class="max-w-5xl mx-auto"
              />
            </div>
          {:else}
            <!-- No Document Selected -->
            <div class="text-center py-12">
              <div class="mx-auto h-12 w-12 text-gray-400 mb-4">
                <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6-4h6m2 5.291A7.962 7.962 0 0112 15c-2.34 0-4.462-.881-6.065-2.325" />
                </svg>
              </div>
              <h3 class="text-lg font-medium text-gray-900 mb-2">Select a document to analyze</h3>
              <p class="text-gray-500 mb-4">
                Choose a document from your search results or recent uploads to begin AI analysis
              </p>
              <div class="space-x-4">
                <Button.Root
                  on:click={() => activeTab = 'search'}
                  disabled={!hasDocuments}
                >
                  Search Documents
                </Button.Root>
                <Button.Root
                  variant="outline"
                  on:click={() => activeTab = 'upload'}
                >
                  Upload New Document
                </Button.Root>
              </div>
            </div>
          {/if}
        </div>
      </Tabs.Content>

      <!-- History Tab -->
      <Tabs.Content value="history">
        <div class="space-y-6">
          <div class="text-center">
            <h2 class="text-3xl font-bold text-gray-900 mb-2">
              Workspace History
            </h2>
            <p class="text-lg text-gray-600 max-w-2xl mx-auto">
              Review your document uploads, searches, and analysis history.
            </p>
          </div>

          <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 max-w-6xl mx-auto">
            <!-- Recent Uploads -->
            <Card.Root>
              <div class="p-6">
                <h3 class="text-lg font-semibold text-gray-900 mb-4">Recent Uploads</h3>
                {#if recentUploads.length > 0}
                  <div class="space-y-3">
                    {#each recentUploads as upload}
                      <div class="flex items-center justify-between p-3 border border-gray-200 rounded-lg">
                        <div>
                          <p class="font-medium text-gray-900">{upload.filename}</p>
                          <p class="text-sm text-gray-500">
                            {upload.uploadedAt.toLocaleString()}
                          </p>
                        </div>
                        <div class="flex items-center space-x-2">
                          <Badge.Root class={getStatusColor(upload.status)}>
                            {upload.status}
                          </Badge.Root>
                          {#if upload.documentId}
                            <Button.Root
                              size="sm"
                              variant="outline"
                              on:click={() => loadDocumentForAnalysis(upload.documentId!)}
                            >
                              View
                            </Button.Root>
                          {/if}
                        </div>
                      </div>
                    {/each}
                  </div>
                {:else}
                  <p class="text-gray-500 text-center py-8">No uploads yet</p>
                {/if}
              </div>
            </Card.Root>

            <!-- Search Results -->
            <Card.Root>
              <div class="p-6">
                <h3 class="text-lg font-semibold text-gray-900 mb-4">Recent Search Results</h3>
                {#if hasSearchResults}
                  <div class="space-y-3">
                    {#each searchResults.slice(0, 5) as result}
                      <div class="p-3 border border-gray-200 rounded-lg">
                        <div class="flex items-start justify-between">
                          <div class="flex-1">
                            <p class="font-medium text-gray-900 text-sm">{result.title}</p>
                            <p class="text-xs text-gray-500 mt-1">
                              {Math.round(result.similarity * 100)}% similarity
                            </p>
                          </div>
                          <Button.Root
                            size="sm"
                            variant="outline"
                            on:click={() => handleDocumentSelect({ detail: { document: result } })}
                          >
                            Analyze
                          </Button.Root>
                        </div>
                      </div>
                    {/each}
                  </div>
                {:else}
                  <p class="text-gray-500 text-center py-8">No search results yet</p>
                {/if}
              </div>
            </Card.Root>
          </div>

          <!-- Workspace Statistics -->
          <Card.Root class="max-w-4xl mx-auto">
            <div class="p-6">
              <h3 class="text-lg font-semibold text-gray-900 mb-4">Workspace Statistics</h3>
              <div class="grid grid-cols-2 md:grid-cols-4 gap-6">
                <div class="text-center">
                  <div class="text-2xl font-bold text-blue-600">{workspaceStats.totalDocuments}</div>
                  <div class="text-sm text-gray-500">Total Documents</div>
                </div>
                <div class="text-center">
                  <div class="text-2xl font-bold text-green-600">{workspaceStats.totalProcessed}</div>
                  <div class="text-sm text-gray-500">Processed</div>
                </div>
                <div class="text-center">
                  <div class="text-2xl font-bold text-purple-600">{workspaceStats.totalSearches}</div>
                  <div class="text-sm text-gray-500">Searches</div>
                </div>
                <div class="text-center">
                  <div class="text-2xl font-bold text-orange-600">
                    {workspaceStats.totalProcessed > 0 ? formatProcessingTime(workspaceStats.avgProcessingTime) : '-'}
                  </div>
                  <div class="text-sm text-gray-500">Avg Process Time</div>
                </div>
              </div>
            </div>
          </Card.Root>
        </div>
      </Tabs.Content>
    </Tabs.Root>
  </main>
</div>

<style>
  .ai-workspace {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }
</style>