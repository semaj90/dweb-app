<!-- src/routes/cases/[caseId]/rag/+page.svelte -->
<!-- Case-specific RAG page with XState machine integration -->

<script lang="ts">
  import { page } from '$app/stores';
  import { useMachine } from '@xstate/svelte';
  import { createLegalCaseMachineForRoute } from '$lib/machines/legal-case-machine-factory';
  import { onMount } from 'svelte';

  // Create machine with caseId from route params
  // Use function form so $derived recomputes when $page changes (don't capture initial value only)
  const machine = $derived(() => createLegalCaseMachineForRoute($page));
  const [state, send] = useMachine(machine);

  // Reactive values from machine state (converted for runes mode)
  const currentCase = $derived(() => $state.context.currentCase);
  const loading = $derived(() => $state.context.loading);
  const error = $derived(() => $state.context.error);
  const aiAnalysis = $derived(() => $state.context.aiAnalysis);

  let query = '';

  // Auto-load case when component mounts or caseId changes
  $effect(() => {
    if ($page.params.caseId && !currentCase?.id) {
      send({ type: 'LOAD_CASE', caseId: $page.params.caseId, includeEvidence: true });
    }
  });

  function askQuestion() {
    if (!query.trim()) return;

    send({
      type: 'RAG_QUERY',
      query: query.trim()
    });
  }

  function findSimilarCases() {
    send({
      type: 'FIND_SIMILAR_CASES',
      threshold: 0.7
    });
  }

  function startAIAnalysis(analysisType: 'summary' | 'recommendation' | 'similarity' = 'summary') {
    send({
      type: 'START_AI_ANALYSIS',
      caseId: $page.params.caseId,
      analysisType
    });
  }

  // Machine state debugging (remove in production)
  $effect(() => { console.log('Machine state:', $state.value, $state.context); });
</script>

<svelte:head>
  <title>Case {$page.params.caseId} - RAG Analysis</title>
</svelte:head>

<div class="container mx-auto p-6">
  <!-- Case Header -->
  {#if currentCase}
    <div class="mb-6 p-4 bg-blue-50 rounded-lg">
      <h1 class="text-2xl font-bold text-blue-900">
        {currentCase.title || `Case ${currentCase.id}`}
      </h1>
      <p class="text-blue-700 mt-1">
        Status: {currentCase.status} | ID: {currentCase.id}
      </p>
    </div>
  {/if}

  <!-- Loading State -->
  {#if loading}
    <div class="flex items-center justify-center p-8">
      <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      <span class="ml-2 text-gray-600">Processing...</span>
    </div>
  {/if}

  <!-- Error State -->
  {#if error}
    <div class="bg-red-50 border border-red-200 rounded-md p-4 mb-6">
      <div class="text-red-800 font-medium">Error</div>
      <div class="text-red-600 text-sm mt-1">{error}</div>
    </div>
  {/if}

  <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
    <!-- RAG Query Section -->
    <div class="bg-white shadow rounded-lg p-6">
      <h2 class="text-lg font-medium mb-4">Ask About This Case</h2>

      <div class="space-y-4">
        <textarea
          bind:value={query}
          rows="4"
          class="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          placeholder="What would you like to know about this case?"
          disabled={loading}
        />

        <div class="flex space-x-2">
          <button
            click={askQuestion}
            disabled={loading || !query.trim()}
            class="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
          >
            Ask Question
          </button>

          <button
            click={findSimilarCases}
            disabled={loading}
            class="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50"
          >
            Find Similar Cases
          </button>
        </div>
      </div>

      <!-- Quick Actions -->
      <div class="mt-6 space-y-2">
        <h3 class="font-medium text-gray-900">Quick Analysis</h3>
        <div class="flex space-x-2">
          <button
            click={() => startAIAnalysis('summary')}
            disabled={loading}
            class="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 disabled:opacity-50"
          >
            Summarize
          </button>
          <button
            click={() => startAIAnalysis('recommendation')}
            disabled={loading}
            class="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 disabled:opacity-50"
          >
            Recommend Actions
          </button>
          <button
            click={() => startAIAnalysis('similarity')}
            disabled={loading}
            class="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 disabled:opacity-50"
          >
            Find Precedents
          </button>
        </div>
      </div>
    </div>

    <!-- Results Section -->
    <div class="bg-white shadow rounded-lg p-6">
      <h2 class="text-lg font-medium mb-4">Analysis Results</h2>

      {#if aiAnalysis.status === 'processing'}
        <div class="flex items-center text-blue-600">
          <div class="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2"></div>
          Processing {aiAnalysis.processingStep || 'analysis'}...
        </div>
      {:else if aiAnalysis.status === 'completed' && aiAnalysis.results}
        <div class="space-y-4">
          <!-- RAG Response -->
          {#if aiAnalysis.results.ragResponse}
            <div>
              <h3 class="font-medium text-gray-900 mb-2">AI Response</h3>
              <div class="bg-gray-50 p-3 rounded text-sm">
                {aiAnalysis.results.ragResponse.answer}
              </div>

              {#if aiAnalysis.results.ragResponse.sources?.length}
                <div class="mt-3">
                  <h4 class="font-medium text-gray-700 text-sm mb-2">Sources</h4>
                  <div class="space-y-2">
                    {#each aiAnalysis.results.ragResponse.sources as source}
                      <div class="text-xs bg-blue-50 p-2 rounded">
                        <div class="font-medium">{source.title}</div>
                        <div class="text-gray-600">
                          Type: {source.document_type} | Similarity: {source.similarity.toFixed(3)}
                        </div>
                      </div>
                    {/each}
                  </div>
                </div>
              {/if}
            </div>
          {/if}

          <!-- Similar Cases -->
          {#if aiAnalysis.results.similarity_cases?.length}
            <div>
              <h3 class="font-medium text-gray-900 mb-2">Similar Cases</h3>
              <div class="space-y-2">
                {#each aiAnalysis.results.similarity_cases as similarCase}
                  <div class="flex justify-between items-center p-2 bg-yellow-50 rounded text-sm">
                    <span>{similarCase.title}</span>
                    <span class="text-gray-600">{similarCase.similarity_score.toFixed(3)}</span>
                  </div>
                {/each}
              </div>
            </div>
          {/if}

          <!-- Key Findings -->
          {#if aiAnalysis.results.keyFindings?.length}
            <div>
              <h3 class="font-medium text-gray-900 mb-2">Key Findings</h3>
              <ul class="text-sm space-y-1">
                {#each aiAnalysis.results.keyFindings as finding}
                  <li class="flex items-start">
                    <span class="text-blue-600 mr-2">•</span>
                    {finding}
                  </li>
                {/each}
              </ul>
            </div>
          {/if}

          <!-- Recommendations -->
          {#if aiAnalysis.results.recommendations?.length}
            <div>
              <h3 class="font-medium text-gray-900 mb-2">Recommendations</h3>
              <ul class="text-sm space-y-1">
                {#each aiAnalysis.results.recommendations as recommendation}
                  <li class="flex items-start">
                    <span class="text-green-600 mr-2">→</span>
                    {recommendation}
                  </li>
                {/each}
              </ul>
            </div>
          {/if}
        </div>
      {:else if aiAnalysis.status === 'failed'}
        <div class="text-red-600 text-sm">
          Analysis failed. Please try again.
        </div>
      {:else}
        <div class="text-gray-500 text-sm">
          Use the controls on the left to analyze this case or ask questions.
        </div>
      {/if}
    </div>
  </div>

  <!-- Machine State Debug Panel (remove in production) -->
  {#if import.meta.env.DEV}
    <div class="mt-8 p-4 bg-gray-100 rounded text-xs">
      <details>
        <summary class="cursor-pointer font-medium">Machine State (Debug)</summary>
        <pre class="mt-2 text-xs overflow-auto">{JSON.stringify({
          state: $state.value,
          loading: $state.context.loading,
          error: $state.context.error,
          hasCase: !!$state.context.currentCase,
          aiAnalysisStatus: $state.context.aiAnalysis.status
        }, null, 2)}</pre>
      </details>
    </div>
  {/if}
</div>