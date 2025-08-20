<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { goto } from '$app/navigation';
  import { browser } from '$app/environment';
  
  // UI Components
  import Button from '$lib/components/ui/enhanced/Button.svelte';
  import Card from '$lib/components/ui/enhanced/Card.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  import Modal from '$lib/components/ui/Modal.svelte';
  import LoadingSpinner from '$lib/components/LoadingSpinner.svelte';

  // Types
  interface Document {
    id: string;
    title: string;
    content: string;
    documentType: string;
    jurisdiction: string;
    practiceArea?: string;
    fileName?: string;
    fileSize?: number;
    mimeType?: string;
    fileHash?: string;
    processingStatus: 'pending' | 'processing' | 'completed' | 'error';
    isConfidential: boolean;
    retentionDate?: string;
    createdAt: string;
    updatedAt: string;
    createdBy?: string;
    lastModifiedBy?: string;
    analysisResults?: {
      entities: Array<{ type: string; value: string; confidence: number }>;
      keyTerms: string[];
      sentimentScore: number;
      complexityScore: number;
      confidenceLevel: number;
      extractedDates: string[];
      extractedAmounts: string[];
      parties: string[];
      obligations: string[];
      risks: Array<{ type: string; severity: 'low' | 'medium' | 'high'; description: string }>;
    };
    hasEmbeddings: boolean;
    embeddingStatus: {
      hasContentEmbedding: boolean;
      hasTitleEmbedding: boolean;
    };
    associatedCases: Array<{
      caseId: string;
      caseTitle: string;
      caseNumber: string;
      relationship: string;
      importance: string;
    }>;
  }

  interface SimilarDocument {
    id: string;
    title: string;
    documentType: string;
    similarity: number;
    rank: number;
  }

  // State
  let document: Document | null = null;
  let similarDocuments: SimilarDocument[] = [];
  let loading = true;
  let error: string | null = null;
  let showDeleteModal = false;
  let showAnalysisModal = false;
  let showSimilarModal = false;
  let isDeleting = false;
  let isRegeneratingEmbeddings = false;
  let activeTab = 'overview';

  // Get document ID from URL
  $: documentId = $page.params.id;

  onMount(async () => {
    if (browser && documentId) {
      await loadDocument();
    }
  });

  // Load document details
  async function loadDocument() {
    try {
      loading = true;
      error = null;

      const response = await fetch(`/api/documents/upload?id=${documentId}`);
      const data = await response.json();

      if (data.success) {
        document = data.document;
        // Load similar documents if embeddings are available
        if (document?.hasEmbeddings) {
          await loadSimilarDocuments();
        }
      } else {
        error = data.error || 'Document not found';
      }
    } catch (err) {
      error = 'Failed to load document';
      console.error('Load document error:', err);
    } finally {
      loading = false;
    }
  }

  // Load similar documents
  async function loadSimilarDocuments() {
    if (!document?.hasEmbeddings) return;

    try {
      // Use the vector search to find similar documents
      const response = await fetch('/api/documents/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: document.title,
          searchType: 'semantic',
          limit: 5,
          threshold: 0.6
        })
      });

      const data = await response.json();
      if (data.success) {
        // Filter out the current document
        similarDocuments = data.results
          .filter((doc: any) => doc.id !== document?.id)
          .slice(0, 4);
      }
    } catch (err) {
      console.error('Load similar documents error:', err);
    }
  }

  // Delete document
  async function deleteDocument() {
    if (!document) return;

    try {
      isDeleting = true;
      const response = await fetch(`/api/documents/upload?id=${document.id}`, {
        method: 'DELETE'
      });

      const data = await response.json();
      if (data.success) {
        goto('/documents');
      } else {
        error = data.error || 'Failed to delete document';
      }
    } catch (err) {
      error = 'Failed to delete document';
      console.error('Delete error:', err);
    } finally {
      isDeleting = false;
      showDeleteModal = false;
    }
  }

  // Regenerate embeddings
  async function regenerateEmbeddings() {
    if (!document) return;

    try {
      isRegeneratingEmbeddings = true;
      // This would trigger re-processing of the document
      const response = await fetch('/api/documents', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'reprocess',
          documentIds: [document.id]
        })
      });

      const data = await response.json();
      if (data.success) {
        // Reload document to get updated status
        await loadDocument();
      } else {
        error = data.error || 'Failed to regenerate embeddings';
      }
    } catch (err) {
      error = 'Failed to regenerate embeddings';
      console.error('Regenerate embeddings error:', err);
    } finally {
      isRegeneratingEmbeddings = false;
    }
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
      month: 'long',
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

  function getRiskColor(severity: string): string {
    switch (severity) {
      case 'high': return 'bg-red-100 text-red-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'low': return 'bg-green-100 text-green-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  }

  function copyToClipboard(text: string) {
    if (browser) {
      navigator.clipboard.writeText(text);
    }
  }
</script>

<svelte:head>
  <title>{document?.title || 'Document'} - Legal Document System</title>
  <meta name="description" content="View and analyze legal document with AI-powered insights" />
</svelte:head>

<div class="min-h-screen bg-gray-50">
  {#if loading}
    <div class="flex justify-center items-center py-20">
      <LoadingSpinner size="lg" />
    </div>
  {:else if error}
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div class="bg-red-50 border border-red-200 rounded-lg p-6">
        <div class="flex">
          <svg class="w-5 h-5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <div class="ml-3">
            <h3 class="text-sm font-medium text-red-800">Error</h3>
            <p class="text-sm text-red-700">{error}</p>
            <div class="mt-4">
              <Button variant="outline" on:click={() => goto('/documents')}>
                Back to Documents
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  {:else if document}
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <!-- Header -->
      <div class="mb-8">
        <nav class="flex mb-4" aria-label="Breadcrumb">
          <ol class="flex items-center space-x-4">
            <li>
              <a href="/documents" class="text-gray-400 hover:text-gray-500">Documents</a>
            </li>
            <li>
              <svg class="flex-shrink-0 h-5 w-5 text-gray-300" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clip-rule="evenodd" />
              </svg>
            </li>
            <li>
              <span class="text-gray-500 truncate max-w-md" title={document.title}>
                {document.title}
              </span>
            </li>
          </ol>
        </nav>

        <div class="flex items-start justify-between">
          <div class="flex-1 min-w-0">
            <h1 class="text-2xl font-bold text-gray-900 break-words">
              {document.title}
            </h1>
            {#if document.fileName}
              <p class="mt-1 text-gray-600">
                File: {document.fileName}
              </p>
            {/if}
          </div>
          
          <div class="flex items-center space-x-4 ml-6">
            {#if document.isConfidential}
              <Badge variant="destructive">Confidential</Badge>
            {/if}
            
            <Badge class={getStatusColor(document.processingStatus)}>
              {document.processingStatus}
            </Badge>
            
            <div class="flex space-x-2">
              <Button variant="outline" size="sm" on:click={() => copyToClipboard(document.id)}>
                Copy ID
              </Button>
              
              {#if document.hasEmbeddings}
                <Button variant="outline" size="sm" on:click={() => showSimilarModal = true}>
                  Similar Docs
                </Button>
              {:else}
                <Button 
                  variant="outline" 
                  size="sm" 
                  on:click={regenerateEmbeddings}
                  disabled={isRegeneratingEmbeddings}
                >
                  {isRegeneratingEmbeddings ? 'Generating...' : 'Generate Embeddings'}
                </Button>
              {/if}
              
              <Button variant="destructive" size="sm" on:click={() => showDeleteModal = true}>
                Delete
              </Button>
            </div>
          </div>
        </div>
      </div>

      <!-- Document Metadata -->
      <div class="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-8">
        <Card class="lg:col-span-3">
          <div class="p-6">
            <h2 class="text-lg font-semibold mb-4">Document Information</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label class="text-sm font-medium text-gray-700">Document Type</label>
                <div class="mt-1">
                  <Badge class={getTypeColor(document.documentType)}>
                    {document.documentType.replace('_', ' ')}
                  </Badge>
                </div>
              </div>
              
              <div>
                <label class="text-sm font-medium text-gray-700">Jurisdiction</label>
                <p class="mt-1 text-sm text-gray-900">{document.jurisdiction}</p>
              </div>
              
              {#if document.practiceArea}
                <div>
                  <label class="text-sm font-medium text-gray-700">Practice Area</label>
                  <p class="mt-1 text-sm text-gray-900">
                    {document.practiceArea.replace('_', ' ')}
                  </p>
                </div>
              {/if}
              
              {#if document.fileSize}
                <div>
                  <label class="text-sm font-medium text-gray-700">File Size</label>
                  <p class="mt-1 text-sm text-gray-900">{formatFileSize(document.fileSize)}</p>
                </div>
              {/if}
              
              {#if document.mimeType}
                <div>
                  <label class="text-sm font-medium text-gray-700">File Type</label>
                  <p class="mt-1 text-sm text-gray-900">{document.mimeType}</p>
                </div>
              {/if}
              
              <div>
                <label class="text-sm font-medium text-gray-700">Created</label>
                <p class="mt-1 text-sm text-gray-900">{formatDate(document.createdAt)}</p>
              </div>
              
              <div>
                <label class="text-sm font-medium text-gray-700">Last Updated</label>
                <p class="mt-1 text-sm text-gray-900">{formatDate(document.updatedAt)}</p>
              </div>
              
              {#if document.retentionDate}
                <div>
                  <label class="text-sm font-medium text-gray-700">Retention Date</label>
                  <p class="mt-1 text-sm text-gray-900">{formatDate(document.retentionDate)}</p>
                </div>
              {/if}
            </div>
          </div>
        </Card>

        <Card>
          <div class="p-6">
            <h2 class="text-lg font-semibold mb-4">AI Features</h2>
            <div class="space-y-3">
              <div class="flex items-center justify-between">
                <span class="text-sm text-gray-700">Vector Embeddings</span>
                {#if document.hasEmbeddings}
                  <Badge variant="secondary" size="sm">Available</Badge>
                {:else}
                  <Badge variant="outline" size="sm">Missing</Badge>
                {/if}
              </div>
              
              <div class="flex items-center justify-between">
                <span class="text-sm text-gray-700">Content Analysis</span>
                {#if document.analysisResults}
                  <Badge variant="secondary" size="sm">Complete</Badge>
                {:else}
                  <Badge variant="outline" size="sm">Pending</Badge>
                {/if}
              </div>
              
              <div class="flex items-center justify-between">
                <span class="text-sm text-gray-700">Semantic Search</span>
                {#if document.hasEmbeddings}
                  <Badge variant="secondary" size="sm">Enabled</Badge>
                {:else}
                  <Badge variant="outline" size="sm">Disabled</Badge>
                {/if}
              </div>
            </div>
            
            {#if document.embeddingStatus}
              <div class="mt-4 pt-4 border-t text-xs text-gray-600">
                <div class="space-y-1">
                  <div class="flex justify-between">
                    <span>Content:</span>
                    <span class={document.embeddingStatus.hasContentEmbedding ? 'text-green-600' : 'text-red-600'}>
                      {document.embeddingStatus.hasContentEmbedding ? '✓' : '✗'}
                    </span>
                  </div>
                  <div class="flex justify-between">
                    <span>Title:</span>
                    <span class={document.embeddingStatus.hasTitleEmbedding ? 'text-green-600' : 'text-red-600'}>
                      {document.embeddingStatus.hasTitleEmbedding ? '✓' : '✗'}
                    </span>
                  </div>
                </div>
              </div>
            {/if}
          </div>
        </Card>
      </div>

      <!-- Associated Cases -->
      {#if document.associatedCases && document.associatedCases.length > 0}
        <Card class="mb-8">
          <div class="p-6">
            <h2 class="text-lg font-semibold mb-4">Associated Cases</h2>
            <div class="grid gap-4">
              {#each document.associatedCases as case_}
                <div class="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <div>
                    <h3 class="font-medium text-gray-900">{case_.caseTitle}</h3>
                    <p class="text-sm text-gray-600">Case #{case_.caseNumber}</p>
                  </div>
                  <div class="flex items-center space-x-2">
                    <Badge variant="outline" size="sm">{case_.relationship}</Badge>
                    <Badge 
                      variant={case_.importance === 'critical' ? 'destructive' : case_.importance === 'high' ? 'default' : 'outline'} 
                      size="sm"
                    >
                      {case_.importance}
                    </Badge>
                    <Button variant="ghost" size="sm" on:click={() => goto(`/cases/${case_.caseId}`)}>
                      View Case
                    </Button>
                  </div>
                </div>
              {/each}
            </div>
          </div>
        </Card>
      {/if}

      <!-- Tabs for Content and Analysis -->
      <div class="mb-6">
        <nav class="flex space-x-8">
          <button
            class="pb-2 text-sm font-medium border-b-2 {activeTab === 'overview' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700'}"
            on:click={() => activeTab = 'overview'}
          >
            Content
          </button>
          
          {#if document.analysisResults}
            <button
              class="pb-2 text-sm font-medium border-b-2 {activeTab === 'analysis' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700'}"
              on:click={() => activeTab = 'analysis'}
            >
              AI Analysis
            </button>
          {/if}
          
          {#if similarDocuments.length > 0}
            <button
              class="pb-2 text-sm font-medium border-b-2 {activeTab === 'similar' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700'}"
              on:click={() => activeTab = 'similar'}
            >
              Similar Documents
            </button>
          {/if}
        </nav>
      </div>

      <!-- Tab Content -->
      {#if activeTab === 'overview'}
        <Card>
          <div class="p-6">
            <h2 class="text-lg font-semibold mb-4">Document Content</h2>
            <div class="prose max-w-none">
              <div class="whitespace-pre-wrap text-sm text-gray-900 border rounded-lg p-4 bg-gray-50 max-h-96 overflow-y-auto">
                {document.content}
              </div>
            </div>
          </div>
        </Card>
      {/if}

      {#if activeTab === 'analysis' && document.analysisResults}
        <div class="space-y-6">
          <!-- Analysis Overview -->
          <Card>
            <div class="p-6">
              <h2 class="text-lg font-semibold mb-4">Analysis Overview</h2>
              <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="text-center p-4 bg-blue-50 rounded-lg">
                  <div class="text-2xl font-bold text-blue-600">
                    {Math.round(document.analysisResults.sentimentScore * 100)}%
                  </div>
                  <div class="text-sm text-gray-600">Sentiment Score</div>
                </div>
                
                <div class="text-center p-4 bg-green-50 rounded-lg">
                  <div class="text-2xl font-bold text-green-600">
                    {Math.round(document.analysisResults.complexityScore * 100)}%
                  </div>
                  <div class="text-sm text-gray-600">Complexity Score</div>
                </div>
                
                <div class="text-center p-4 bg-purple-50 rounded-lg">
                  <div class="text-2xl font-bold text-purple-600">
                    {Math.round(document.analysisResults.confidenceLevel * 100)}%
                  </div>
                  <div class="text-sm text-gray-600">Confidence Level</div>
                </div>
              </div>
            </div>
          </Card>

          <!-- Entities and Key Terms -->
          <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <div class="p-6">
                <h3 class="text-lg font-semibold mb-4">Extracted Entities</h3>
                {#if document.analysisResults.entities.length > 0}
                  <div class="space-y-2">
                    {#each document.analysisResults.entities as entity}
                      <div class="flex items-center justify-between p-2 bg-gray-50 rounded">
                        <span class="text-sm">{entity.value}</span>
                        <div class="flex items-center space-x-2">
                          <Badge variant="outline" size="sm">{entity.type}</Badge>
                          <span class="text-xs text-gray-600">
                            {Math.round(entity.confidence * 100)}%
                          </span>
                        </div>
                      </div>
                    {/each}
                  </div>
                {:else}
                  <p class="text-gray-500 text-sm">No entities extracted</p>
                {/if}
              </div>
            </Card>

            <Card>
              <div class="p-6">
                <h3 class="text-lg font-semibold mb-4">Key Terms</h3>
                {#if document.analysisResults.keyTerms.length > 0}
                  <div class="flex flex-wrap gap-2">
                    {#each document.analysisResults.keyTerms as term}
                      <Badge variant="secondary" size="sm">{term}</Badge>
                    {/each}
                  </div>
                {:else}
                  <p class="text-gray-500 text-sm">No key terms identified</p>
                {/if}
              </div>
            </Card>
          </div>

          <!-- Risks -->
          {#if document.analysisResults.risks.length > 0}
            <Card>
              <div class="p-6">
                <h3 class="text-lg font-semibold mb-4">Identified Risks</h3>
                <div class="space-y-3">
                  {#each document.analysisResults.risks as risk}
                    <div class="p-4 border rounded-lg">
                      <div class="flex items-start justify-between">
                        <div class="flex-1">
                          <div class="flex items-center space-x-2 mb-2">
                            <Badge class={getRiskColor(risk.severity)} size="sm">
                              {risk.severity.toUpperCase()}
                            </Badge>
                            <span class="font-medium text-gray-900">{risk.type}</span>
                          </div>
                          <p class="text-sm text-gray-700">{risk.description}</p>
                        </div>
                      </div>
                    </div>
                  {/each}
                </div>
              </div>
            </Card>
          {/if}

          <!-- Parties and Obligations -->
          {#if document.analysisResults.parties.length > 0 || document.analysisResults.obligations.length > 0}
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {#if document.analysisResults.parties.length > 0}
                <Card>
                  <div class="p-6">
                    <h3 class="text-lg font-semibold mb-4">Identified Parties</h3>
                    <div class="space-y-2">
                      {#each document.analysisResults.parties as party}
                        <div class="p-2 bg-gray-50 rounded text-sm">{party}</div>
                      {/each}
                    </div>
                  </div>
                </Card>
              {/if}

              {#if document.analysisResults.obligations.length > 0}
                <Card>
                  <div class="p-6">
                    <h3 class="text-lg font-semibold mb-4">Key Obligations</h3>
                    <div class="space-y-2">
                      {#each document.analysisResults.obligations as obligation}
                        <div class="p-2 bg-gray-50 rounded text-sm">{obligation}</div>
                      {/each}
                    </div>
                  </div>
                </Card>
              {/if}
            </div>
          {/if}
        </div>
      {/if}

      {#if activeTab === 'similar' && similarDocuments.length > 0}
        <Card>
          <div class="p-6">
            <h2 class="text-lg font-semibold mb-4">Similar Documents</h2>
            <div class="space-y-4">
              {#each similarDocuments as simDoc}
                <div class="flex items-center justify-between p-4 border rounded-lg hover:bg-gray-50">
                  <div class="flex-1">
                    <h3 class="font-medium text-gray-900">{simDoc.title}</h3>
                    <div class="flex items-center space-x-2 mt-1">
                      <Badge class={getTypeColor(simDoc.documentType)} size="sm">
                        {simDoc.documentType.replace('_', ' ')}
                      </Badge>
                      <span class="text-sm text-gray-600">
                        {Math.round(simDoc.similarity * 100)}% similar
                      </span>
                    </div>
                  </div>
                  <Button variant="ghost" size="sm" on:click={() => goto(`/documents/${simDoc.id}`)}>
                    View Document
                  </Button>
                </div>
              {/each}
            </div>
          </div>
        </Card>
      {/if}
    </div>
  {/if}
</div>

<!-- Delete Confirmation Modal -->
{#if showDeleteModal}
  <Modal title="Delete Document" on:close={() => showDeleteModal = false}>
    <div class="space-y-4">
      <p class="text-gray-700">
        Are you sure you want to delete "{document?.title}"? This action cannot be undone.
      </p>
      
      <div class="flex justify-end space-x-4">
        <Button variant="outline" on:click={() => showDeleteModal = false} disabled={isDeleting}>
          Cancel
        </Button>
        <Button variant="destructive" on:click={deleteDocument} disabled={isDeleting}>
          {isDeleting ? 'Deleting...' : 'Delete Document'}
        </Button>
      </div>
    </div>
  </Modal>
{/if}

<!-- Similar Documents Modal -->
{#if showSimilarModal}
  <Modal title="Similar Documents" on:close={() => showSimilarModal = false}>
    <div class="space-y-4">
      {#if similarDocuments.length > 0}
        {#each similarDocuments as simDoc}
          <div class="flex items-center justify-between p-4 border rounded-lg">
            <div>
              <h3 class="font-medium text-gray-900">{simDoc.title}</h3>
              <p class="text-sm text-gray-600">{Math.round(simDoc.similarity * 100)}% similar</p>
            </div>
            <Button variant="outline" size="sm" on:click={() => goto(`/documents/${simDoc.id}`)}>
              View
            </Button>
          </div>
        {/each}
      {:else}
        <p class="text-gray-500">No similar documents found.</p>
      {/if}
    </div>
  </Modal>
{/if}