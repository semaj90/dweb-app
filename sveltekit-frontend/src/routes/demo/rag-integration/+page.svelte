<!-- src/routes/demo/rag-integration/+page.svelte -->
<!-- RAG Demo: SvelteKit 2 + Postgres pgvector + Gemma3 Integration -->

<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';

  interface RAGSource {
    id: string;
    document_id: string;
    title: string;
    document_type: string;
    similarity: number;
    preview: string;
  }

  interface RAGResponse {
    answer: string;
    sources: RAGSource[];
    metadata: {
      query: string;
      caseId?: string;
      resultCount: number;
      processingTime: number;
      model: string;
    };
  }

  // State
let query = $state('');
let loading = $state(false);
let response = $state<RAGResponse | null >(null);
let error = $state<string | null >(null);
  // caseId derived from route params (runes mode) but must be mutable for binding
let caseId = $state('');
  $effect(() => {
    caseId = $page.params.caseId ?? '';
  });

  async function runRAGQuery() {
    if (!query.trim()) return;

    loading = true;
    error = null;
    response = null;

    try {
      const requestBody = {
        query: query.trim(),
        ...(caseId && { caseId }),
        options: {
          limit: 8,
          model: 'gemma3-legal',
          maxTokens: 800,
          temperature: 0.1
        }
      };

      console.log('RAG Request:', requestBody);

      const res = await fetch('/api/v1/rag', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.message || `HTTP ${res.status}: ${res.statusText}`);
      }

      const data = await res.json();
      response = data;
      console.log('RAG Response:', data);

    } catch (err) {
      console.error('RAG query failed:', err);
      error = err instanceof Error ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  function getSimilarityColor(similarity: number): string {
    if (similarity < 0.3) return 'bg-green-100 text-green-800';
    if (similarity < 0.6) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  }

  function getSimilarityLabel(similarity: number): string {
    if (similarity < 0.3) return 'High';
    if (similarity < 0.6) return 'Medium';
    return 'Low';
  }

  // Example queries for demonstration
  const exampleQueries = [
    "What are the key legal issues in this case?",
    "Summarize the evidence and its implications",
    "What precedents are relevant to this matter?",
    "Analyze the contractual obligations mentioned",
    "What are the potential risks and recommendations?"
  ];

  function setExampleQuery(exampleQuery: string) {
    query = exampleQuery;
  }

  // Health check on mount
  onMount(async () => {
    try {
      const healthRes = await fetch('/api/v1/rag');
      console.log('RAG service health:', await healthRes.json());
    } catch (err) {
      console.warn('RAG service health check failed:', err);
    }
  });
</script>

<svelte:head>
  <title>RAG Integration Demo - Legal AI Platform</title>
  <meta name="description" content="Retrieval-Augmented Generation with Postgres pgvector and Gemma3" />
</svelte:head>

<div class="container mx-auto p-6 max-w-6xl">
  <div class="mb-8">
    <h1 class="text-3xl font-bold text-gray-900 mb-2">
      RAG Integration Demo
    </h1>
    <p class="text-gray-600">
      Query legal documents using PostgreSQL pgvector similarity search + Gemma3 AI
    </p>
    {#if caseId}
      <div class="mt-2">
        <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
          Case ID: {caseId}
        </span>
      </div>
    {/if}
  </div>

  <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
    <!-- Query Input Section -->
    <div class="lg:col-span-2">
      <div class="bg-white shadow rounded-lg">
        <div class="px-4 py-5 sm:p-6">
          <h3 class="text-lg leading-6 font-medium text-gray-900 mb-4">
            Ask a Legal Question
          </h3>

          <div class="space-y-4">
            <!-- Query Input -->
            <div>
              <label for="query" class="block text-sm font-medium text-gray-700 mb-2">
                Your Question
              </label>
              <textarea
                id="query"
                bind:value={query}
                rows="4"
                class="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Ask about legal documents, contracts, evidence, precedents..."
                disabled={loading}
              />
            </div>

            <!-- Case ID Input (optional) -->
            <div>
              <label for="caseId" class="block text-sm font-medium text-gray-700 mb-2">
                Case ID (optional)
              </label>
              <input
                id="caseId"
                type="text"
                bind:value={caseId}
                class="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Filter by specific case ID"
                disabled={loading}
              />
            </div>

            <!-- Action Button -->
            <div class="flex justify-between items-center">
              <button
                onclick={runRAGQuery}
                disabled={loading || !query.trim()}
                class="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {#if loading}
                  <svg class="animate-spin -ml-1 mr-3 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Processing...
                {:else}
                  🔍 Run RAG Query
                {/if}
              </button>

              {#if response?.metadata}
                <div class="text-sm text-gray-500">
                  Processed in {response.metadata.processingTime}ms
                </div>
              {/if}
            </div>

            {#if error}
              <div class="rounded-md bg-red-50 p-4">
                <div class="text-sm">
                  <div class="text-red-800 font-medium">Error</div>
                  <div class="text-red-600 mt-1">{error}</div>
                </div>
              </div>
            {/if}
          </div>
        </div>
      </div>

      <!-- Response Section -->
      {#if response}
        <div class="bg-white shadow rounded-lg mt-6">
          <div class="px-4 py-5 sm:p-6">
            <div class="flex items-center justify-between mb-4">
              <h3 class="text-lg leading-6 font-medium text-gray-900">
                AI Response
              </h3>
              <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                {response.metadata.model}
              </span>
            </div>

            <div class="prose prose-gray max-w-none">
              <div class="whitespace-pre-wrap bg-gray-50 p-4 rounded-md border text-sm">
                {response.answer}
              </div>
            </div>
          </div>
        </div>
      {/if}

      <!-- Sources Section -->
      {#if response?.sources?.length}
        <div class="bg-white shadow rounded-lg mt-6">
          <div class="px-4 py-5 sm:p-6">
            <h3 class="text-lg leading-6 font-medium text-gray-900 mb-4">
              Sources ({response.sources.length})
            </h3>

            <div class="space-y-4">
              {#each response.sources as source, i}
                <div class="border border-gray-200 rounded-lg p-4">
                  <div class="flex items-start justify-between mb-2">
                    <div class="flex items-center space-x-2">
                      <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                        Source {i + 1}
                      </span>
                      <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                        {source.document_type}
                      </span>
                      <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium {getSimilarityColor(source.similarity)}">
                        {getSimilarityLabel(source.similarity)} Match
                      </span>
                    </div>
                    <div class="text-sm text-gray-500">
                      ID: {source.document_id}
                    </div>
                  </div>

                  {#if source.title}
                    <h4 class="font-medium text-gray-900 mb-2">
                      {source.title}
                    </h4>
                  {/if}

                  <p class="text-sm text-gray-600">
                    {source.preview}
                  </p>

                  <div class="mt-2 text-xs text-gray-400">
                    Similarity Score: {source.similarity.toFixed(4)}
                  </div>
                </div>
              {/each}
            </div>
          </div>
        </div>
      {/if}
    </div>

    <!-- Sidebar with Examples and Info -->
    <div class="space-y-6">
      <!-- Example Queries -->
      <div class="bg-white shadow rounded-lg">
        <div class="px-4 py-5 sm:p-6">
          <h3 class="text-lg leading-6 font-medium text-gray-900 mb-4">
            Example Queries
          </h3>

          <div class="space-y-2">
            {#each exampleQueries as exampleQuery}
              <button
                onclick={() => setExampleQuery(exampleQuery)}
                class="w-full text-left p-3 text-sm bg-gray-50 hover:bg-gray-100 rounded-md transition-colors"
                disabled={loading}
              >
                {exampleQuery}
              </button>
            {/each}
          </div>
        </div>
      </div>

      <!-- System Info -->
      <div class="bg-white shadow rounded-lg">
        <div class="px-4 py-5 sm:p-6">
          <h3 class="text-lg leading-6 font-medium text-gray-900 mb-4">
            System Architecture
          </h3>

          <div class="text-sm space-y-2">
            <div class="flex justify-between">
              <span class="text-gray-600">Frontend:</span>
              <span class="font-medium">SvelteKit 2</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-600">Vector DB:</span>
              <span class="font-medium">PostgreSQL + pgvector</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-600">Embeddings:</span>
              <span class="font-medium">nomic-embed-text</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-600">LLM:</span>
              <span class="font-medium">Gemma3 Legal</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-600">ORM:</span>
              <span class="font-medium">Drizzle</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Performance Metrics -->
      {#if response?.metadata}
        <div class="bg-white shadow rounded-lg">
          <div class="px-4 py-5 sm:p-6">
            <h3 class="text-lg leading-6 font-medium text-gray-900 mb-4">
              Query Metrics
            </h3>

            <div class="text-sm space-y-2">
              <div class="flex justify-between">
                <span class="text-gray-600">Results:</span>
                <span class="font-medium">{response.metadata.resultCount}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-600">Processing Time:</span>
                <span class="font-medium">{response.metadata.processingTime}ms</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-600">Model:</span>
                <span class="font-medium">{response.metadata.model}</span>
              </div>
              {#if response.metadata.caseId}
                <div class="flex justify-between">
                  <span class="text-gray-600">Case Scope:</span>
                  <span class="font-medium">{response.metadata.caseId}</span>
                </div>
              {/if}
            </div>
          </div>
        </div>
      {/if}
    </div>
  </div>
</div>