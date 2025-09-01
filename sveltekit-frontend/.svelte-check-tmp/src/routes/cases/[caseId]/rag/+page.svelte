<!-- src/routes/cases/[caseId]/rag/+page.svelte -->
<!-- Case-specific RAG page with XState machine integration -->

<script lang="ts">
  import { useMachine } from '@xstate/svelte';
  import { createLegalCaseMachineForRoute } from '$lib/machines/legal-case-machine-factory';
  import { onMount, onDestroy } from 'svelte';
  import { get, writable, derived } from 'svelte/store';

  // helper to read caseId from current URL (client-side)
  function getCaseIdFromLocation(): string | undefined {
    if (typeof window === 'undefined') return undefined;
    const m = window.location.pathname.match(/\/cases\/([^\/]+)/);
    return m?.[1];
  }

  // create placeholders and a safe initial state to avoid SSR/hydration mismatches;
  // the real machine and useMachine will be instantiated on mount
  let machine: any;
  // provide a minimal writable store shape so reactive $machineStore usages are safe before mount
  let machineStore = writable({
    context: {
      currentCase: null,
      loading: false,
      error: null,
      aiAnalysis: { status: 'idle', results: null }
    },
    value: 'idle'
  });
  // typed send until replaced by useMachine result
  let send: (event: any) => void = () => {};

  onMount(() => {
    const caseId = getCaseIdFromLocation();
    // instantiate the machine with the current route snapshot and start the XState machine
    machine = createLegalCaseMachineForRoute({ params: { caseId } });
    const tuple = useMachine(machine);
    machineStore = tuple[0];
    send = tuple[1];

    // initial attempt to load case
    tryLoadCase({ params: { caseId } });

    // subscribe to popstate to detect client-side route changes
    const onPopState = () => tryLoadCase({ params: { caseId: getCaseIdFromLocation() } });
    window.addEventListener('popstate', onPopState);
    onDestroy(() => window.removeEventListener('popstate', onPopState));
  });

  // Derived stores (replace legacy reactive statements)
  const currentCase = derived(machineStore, ($s) => $s.context.currentCase);
  const loading = derived(machineStore, ($s) => $s.context.loading);
  const error = derived(machineStore, ($s) => $s.context.error);
  const aiAnalysis = derived(machineStore, ($s) => $s.context.aiAnalysis ?? { status: 'idle', results: null });

  // Derived page title so <title> has only text and {tags}
  const pageTitle = derived(currentCase, (c) => (c ? `Case ${c.id} - RAG Analysis` : 'Case Unknown - RAG Analysis'));

  // make `query` a reactive variable so bindings and updates are reactive and avoid the non-reactive-update warning
  let query = '';

  // helper to attempt loading case when route has caseId
  function tryLoadCase(p: { params?: { caseId?: string } }) {
    if (p?.params?.caseId && !get(currentCase)?.id) {
      send({ type: 'LOAD_CASE', caseId: p.params.caseId, includeEvidence: true });
    }
  }

  // askQuestion removed (unused) to satisfy the linter/compile checks

  function findSimilarCases() {
    send({
      type: 'FIND_SIMILAR_CASES',
      threshold: 0.7
    });
  }

  function startAIAnalysis(analysisType: 'summary' | 'recommendation' | 'similarity' = 'summary') {
    send({
      type: 'START_AI_ANALYSIS',
      caseId: getCaseIdFromLocation(),
      analysisType
    });
  }

  // helper to safely access a non-typed processingStep on aiAnalysis in the template
  // removed unused helper to satisfy linter/compile checks

// Machine state debugging (remove in production)
// expose a small derived store for template debugging and avoid reading undefined properties
const debugState = derived(machineStore, ($s) => ({
  state: $s?.value,
  loading: $s?.context?.loading,
  error: $s?.context?.error,
  hasCase: !!$s?.context?.currentCase,
  aiAnalysisStatus: $s?.context?.aiAnalysis?.status
}));

// safe JSON stringify to avoid "Converting circular structure to JSON" when debugging XState objects
function safeStringify(obj: any) {
  const seen = new WeakSet();
  return JSON.stringify(obj, function (_key, value) {
    if (typeof value === 'function') return '[Function]';
    if (typeof value === 'object' && value !== null) {
      if (seen.has(value)) return '[Circular]';
      seen.add(value);
    }
    return value;
  }, 2);
}

if (import.meta.env.DEV) {
  const unsub = machineStore.subscribe(($s) => console.log('Machine state:', $s?.value, $s?.context));
  onDestroy(unsub);
}
</script>

<svelte:head>
  <title>{$pageTitle}</title>
</svelte:head>

<div class="container mx-auto p-6">
  <!-- Case Header -->
  {#if $currentCase}
    <div class="mb-6 p-4 bg-blue-50 rounded-lg">
      <h1 class="text-2xl font-bold text-blue-900">
        {$currentCase.title || `Case ${$currentCase.id}`}
      </h1>
      <p class="text-blue-700 mt-1">
        Status: {$currentCase.status} | ID: {$currentCase.id}
      </p>
    </div>
  {/if}

  <!-- Loading State -->
  {#if $loading}
    <div class="flex items-center justify-center p-8">
      <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      <span class="ml-2 text-gray-600">Processing...</span>
    </div>
  {/if}

  <!-- Error State -->
  {#if $error}
    <div class="bg-red-50 border border-red-200 rounded-md p-4 mb-6">
      <div class="text-red-800 font-medium">Error</div>
      <div class="text-red-600 text-sm mt-1">{$error}</div>
    </div>
  {/if}

    <!-- Controls -->
    {#if $aiAnalysis.status !== 'completed'}
    <div class="mb-4">
      <textarea
        bind:value={query}
        rows="4"
        class="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
      ></textarea>

      <div class="mt-3 flex flex-wrap gap-2">
        <button
          click={findSimilarCases}
          disabled={$loading}
          class="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50"
        >
          Find Similar Cases
        </button>

        <button
          click={() => startAIAnalysis('summary')}
          disabled={$loading}
          class="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 disabled:opacity-50"
        >
          Summarize
        </button>

        <button
          click={() => startAIAnalysis('recommendation')}
          disabled={$loading}
          class="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 disabled:opacity-50"
        >
          Recommend Actions
        </button>

        <button
          click={() => startAIAnalysis('similarity')}
          disabled={$loading}
          class="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 disabled:opacity-50"
        >
          Find Precedents
        </button>
      </div>
    </div>
    {:else if $aiAnalysis.status === 'completed' && $aiAnalysis.results}
      <div class="space-y-4">
        <!-- RAG Response -->
        {#if $aiAnalysis.results?.ragResponse}
          <div>
            <h3 class="font-medium text-gray-900 mb-2">AI Response</h3>
            <div class="bg-gray-50 p-3 rounded text-sm">
              {$aiAnalysis.results.ragResponse?.answer}
            </div>

            {#if $aiAnalysis.results.ragResponse?.sources?.length}
              <div class="mt-3">
                <h4 class="font-medium text-gray-700 text-sm mb-2">Sources</h4>
                <div class="space-y-2">
                  {#each $aiAnalysis.results.ragResponse.sources as source}
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
        {#if $aiAnalysis.results?.similarity_cases?.length}
          <div>
            <h3 class="font-medium text-gray-900 mb-2">Similar Cases</h3>
            <div class="space-y-2">
              {#each $aiAnalysis.results.similarity_cases as similarCase}
                <div class="flex justify-between items-center p-2 bg-yellow-50 rounded text-sm">
                  <span>{similarCase.title}</span>
                  <span class="text-gray-600">{similarCase.similarity_score.toFixed(3)}</span>
                </div>
              {/each}
            </div>
          </div>
        {/if}

        <!-- Key Findings -->
        {#if $aiAnalysis.results?.keyFindings?.length}
          <div>
            <h3 class="font-medium text-gray-900 mb-2">Key Findings</h3>
            <ul class="text-sm space-y-1">
              {#each $aiAnalysis.results.keyFindings as finding}
                <li class="flex items-start">
                  <span class="text-blue-600 mr-2">•</span>
                  {finding}
                </li>
              {/each}
            </ul>
          </div>
        {/if}

        <!-- Recommendations -->
        {#if $aiAnalysis.results?.recommendations?.length}
          <div>
            <h3 class="font-medium text-gray-900 mb-2">Recommendations</h3>
            <ul class="text-sm space-y-1">
              {#each $aiAnalysis.results.recommendations as recommendation}
                <li class="flex items-start">
                  <span class="text-green-600 mr-2">→</span>
                  {recommendation}
                </li>
              {/each}
            </ul>
          </div>
        {/if}
      </div>
    {/if}

  <!-- Machine State Debug Panel (remove in production) -->
  {#if import.meta.env.DEV}
    <div class="mt-8 p-4 bg-gray-100 rounded text-xs">
      <div class="text-gray-500 text-sm">
        Use the controls on the left to analyze this case or ask questions.
      </div>

      <pre class="mt-2 text-xs overflow-auto">{JSON.stringify($debugState, null, 2)}</pre>
      <pre class="mt-2 text-xs overflow-auto">{safeStringify($machineStore)}</pre>
    </div>
  {/if}
</div>