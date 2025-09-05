<script lang="ts">
  interface Props {
    selectedDocument: Document | null ;
    searchQuery: string
  }
  let {
    selectedDocument = null,
    searchQuery = ''
  } = $props();



  import { onMount } from 'svelte';

  interface Document {
    id: string
    title: string
    documentType: string
    content: string
    caseId?: string;
  }

  interface SimilarityResult {
    id: string
    title: string
    documentType: string
    content: string
    similarity: number
    caseId?: string;
  }

   // Document | null = null;
  let { searchQuery = $bindable() } = $props(); // string = '';

  let similarDocuments: SimilarityResult[] = [];
  let isLoading: boolean = false;
  let error: string | null = null;

  async function performSemanticSearch(query: string) {
    if (!query.trim()) {
      similarDocuments = [];
      return;
    }

    isLoading = true;
    error = null;

    try {
      const response = await fetch('/api/semantic-search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query.trim(),
          limit: 5,
          threshold: 0.3,
        }),
      });

      const data = await response.json();

      if (data.success) {
        similarDocuments = data.results;
      } else {
        error = data.error || 'Search failed';
        similarDocuments = [];
      }
    } catch (err) {
      error = err instanceof Error ? err.message : 'Search failed';
      similarDocuments = [];
    } finally {
      isLoading = false;
    }
  }

  // Reactive search when query changes
  $effect(() => { if (searchQuery) {
    performSemanticSearch(searchQuery);
  }

  onMount(() => {
    if (searchQuery) {
      performSemanticSearch(searchQuery);
    }
  });
</script>

<!-- Search Input -->
<div class="mb-6">
  <label for="search-query" class="block text-sm font-medium mb-2">
    🔍 Semantic Search Legal Documents
  </label>
  <div class="flex gap-2">
    <input
      id="search-query"
      type="text"
      bind:value={searchQuery}
      placeholder="Enter search query (e.g., 'property deed transfer', 'contract liability'...)"
      class="flex-1 px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
    />
    <button
      onclick={() => performSemanticSearch(searchQuery)}
      disabled={isLoading || !searchQuery.trim()}
      class="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
    >
      {isLoading ? '🔄' : '🔍'}
    </button>
  </div>
</div>

<!-- Selected Document Display -->
{#if selectedDocument}
  <div class="bg-white rounded-lg shadow-md p-6 mb-6 border">
    <h2 class="text-xl font-bold mb-4 text-gray-800">
      📄 Document Analysis: {selectedDocument.title}
    </h2>

    <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
      <div>
        <p class="text-sm text-gray-600"><strong>ID:</strong> {selectedDocument.id}</p>
        <p class="text-sm text-gray-600"><strong>Type:</strong> {selectedDocument.documentType}</p>
      </div>
      <div>
        {#if selectedDocument.caseId}
          <p class="text-sm text-gray-600"><strong>Case ID:</strong> {selectedDocument.caseId}</p>
        {/if}
      </div>
    </div>

    <div class="mb-4">
      <h3 class="font-bold mb-2 text-gray-700">Content Preview</h3>
      <div class="bg-gray-50 p-3 rounded border max-h-40 overflow-y-auto">
        <p class="text-sm text-gray-700 whitespace-pre-wrap">{selectedDocument.content.slice(0, 500)}{selectedDocument.content.length > 500 ? '...' : ''}</p>
      </div>
    </div>
  </div>
{/if}

<!-- Search Results -->
<div class="bg-white rounded-lg shadow-md p-6 border">
  <h2 class="text-xl font-bold mb-4 text-gray-800">
    🎯 Similar Documents {searchQuery ? `for "${searchQuery}"` : ''}
  </h2>

  {#if error}
    <div class="bg-red-50 border border-red-200 rounded-md p-4 mb-4">
      <div class="flex">
        <div class="flex-shrink-0">
          <span class="text-red-400">❌</span>
        </div>
        <div class="ml-3">
          <h3 class="text-sm font-medium text-red-800">Search Error</h3>
          <p class="text-sm text-red-700 mt-1">{error}</p>
        </div>
      </div>
    </div>
  {/if}

  {#if isLoading}
    <div class="flex items-center justify-center py-8">
      <div class="flex items-center space-x-2">
        <div class="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
        <p class="text-gray-600">Searching for similar documents...</p>
      </div>
    </div>
  {:else if similarDocuments.length > 0}
    <div class="space-y-4">
      {#each similarDocuments as doc, index}
        <div class="border rounded-lg p-4 hover:shadow-md transition-shadow">
          <div class="flex justify-between items-start mb-2">
            <h4 class="font-semibold text-gray-900">{doc.title}</h4>
            <div class="flex items-center space-x-2">
              <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                {doc.documentType}
              </span>
              <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                {(doc.similarity * 100).toFixed(1)}% match
              </span>
            </div>
          </div>

          <div class="text-sm text-gray-600 mb-2">
            <p><strong>ID:</strong> {doc.id}</p>
            {#if doc.caseId}
              <p><strong>Case:</strong> {doc.caseId}</p>
            {/if}
          </div>

          <div class="bg-gray-50 p-3 rounded text-sm">
            <p class="text-gray-700">{doc.content.slice(0, 200)}{doc.content.length > 200 ? '...' : ''}</p>
          </div>

          <div class="mt-3 flex justify-end">
            <button
              onclick={() => selectedDocument = doc}
              class="text-blue-600 hover:text-blue-800 text-sm font-medium"
            >
              View Details →
            </button>
          </div>
        </div>
      {/each}
    </div>
  {:else if searchQuery}
    <div class="text-center py-8">
      <div class="text-gray-400 text-4xl mb-4">📄</div>
      <p class="text-gray-600 mb-2">No similar documents found for "{searchQuery}"</p>
      <p class="text-sm text-gray-500">Try adjusting your search terms or lowering the similarity threshold</p>
    </div>
  {:else}
    <div class="text-center py-8">
      <div class="text-gray-400 text-4xl mb-4">🔍</div>
      <p class="text-gray-600 mb-2">Enter a search query to find similar legal documents</p>
      <p class="text-sm text-gray-500">Use natural language like "property deed transfer" or "contract liability clauses"</p>
    </div>
  {/if}
</div>
