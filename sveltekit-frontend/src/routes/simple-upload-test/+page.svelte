<!-- Simple Upload Test Page -->
<script lang="ts">
  import SimpleFileUpload from '$lib/components/ai/SimpleFileUpload.svelte';
  import { onMount } from 'svelte';

  let uploadResults = $state<any[]>([]);
  let searchQuery = $state('');
  let searchResults = $state<any[]>([]);
  let isSearching = $state(false);

  function handleUploadComplete(result: any) {
    console.log('Upload completed:', result);
    uploadResults = [...uploadResults, result];
  }

  async function performSearch() {
    if (!searchQuery.trim()) return;

    isSearching = true;
    try {
      const response = await fetch('/api/rag/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: searchQuery,
          searchType: 'semantic',
          limit: 5
        })
      });

      if (response.ok) {
        const result = await response.json();
        searchResults = result.results || [];
        console.log('Search results:', result);
      } else {
        console.error('Search failed:', await response.text());
      }
    } catch (error) {
      console.error('Search error:', error);
    } finally {
      isSearching = false;
    }
  }
</script>

<svelte:head>
  <title>Simple RAG Upload Test - Legal AI System</title>
</svelte:head>

<div class="container mx-auto p-6 max-w-4xl">
  <h1 class="text-3xl font-bold mb-6">Simple RAG Upload Test</h1>

  <!-- Upload Section -->
  <div class="mb-8">
    <h2 class="text-xl font-semibold mb-4">File Upload & Processing</h2>
    <SimpleFileUpload onUploadComplete={handleUploadComplete} />
  </div>

  <!-- Search Section -->
  <div class="mb-8">
    <h2 class="text-xl font-semibold mb-4">Search Documents</h2>
    <div class="flex gap-2 mb-4">
      <input
        type="text"
        bind:value={searchQuery}
        placeholder="Search uploaded documents..."
        class="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        onkeydown={(e) => e.key === 'Enter' && performSearch()}
      />
      <button
        onclick={performSearch}
        disabled={isSearching}
        class="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
      >
        {isSearching ? 'Searching...' : 'Search'}
      </button>
    </div>

    {#if searchResults.length > 0}
      <div class="space-y-3">
        <h3 class="font-medium">Search Results:</h3>
        {#each searchResults as result}
          <div class="border rounded-lg p-4">
            <div class="font-medium">{result.filename || result.title || 'Unknown Document'}</div>
            {#if result.similarity}
              <div class="text-sm text-gray-600">Similarity: {(result.similarity * 100).toFixed(1)}%</div>
            {/if}
            <div class="text-sm mt-2">{result.content}</div>
            {#if result.metadata}
              <div class="text-xs text-gray-500 mt-2">
                {JSON.stringify(result.metadata)}
              </div>
            {/if}
          </div>
        {/each}
      </div>
    {/if}
  </div>

  <!-- Upload Results -->
  {#if uploadResults.length > 0}
    <div class="mb-8">
      <h2 class="text-xl font-semibold mb-4">Upload Results</h2>
      <div class="space-y-4">
        {#each uploadResults as result, index}
          <div class="border rounded-lg p-4">
            <h3 class="font-medium">Upload #{index + 1}</h3>
            <pre class="text-xs bg-gray-50 p-2 rounded mt-2 overflow-auto max-h-40">
{JSON.stringify(result, null, 2)}
            </pre>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- API Test Section -->
  <div class="mt-8 p-4 bg-gray-50 rounded-lg">
    <h3 class="font-medium mb-2">Quick API Tests</h3>
    <div class="flex gap-2 flex-wrap">
      <button
        onclick={() => fetch('/api/rag/status').then(r => r.json()).then(console.log)}
        class="px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600"
      >
        Test Status API
      </button>
      <button
        onclick={() => fetch('/api/ai/embeddings').then(r => r.json()).then(console.log)}
        class="px-3 py-1 bg-green-500 text-white rounded text-sm hover:bg-green-600"
      >
        Test Embeddings API
      </button>
    </div>
    <p class="text-xs text-gray-600 mt-2">Check browser console for results</p>
  </div>
</div>

<style>
  pre {
    white-space: pre-wrap;
    word-break: break-all;
  }
</style>
