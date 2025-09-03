<!-- 
  DID YOU MEAN SUGGESTIONS COMPONENT
  Ultra-fast search suggestions with QUIC optimization
  Supports 1000+ concurrent streams for instant suggestions
-->
<script lang="ts">
  import { onMount, createEventDispatcher } from 'svelte';
  import { didYouMeanService, type DidYouMeanQuery, type DidYouMeanSuggestion } from '$lib/services/did-you-mean-quic-graph.js';
  import { fade, fly } from 'svelte/transition';
  import { quintOut } from 'svelte/easing';

  // Props
  export let query = '';
  export let userIntent: 'search' | 'legal_research' | 'case_lookup' | 'document_analysis' = 'search';
  export let maxSuggestions = 5;
  export let showTypos = true;
  export let showSemantic = true;
  export let threshold = 0.3;
  export let context: { caseId?: string; practiceArea?: string; jurisdiction?: string } = {};
  export let debounceMs = 150;
  export let showMetrics = false;

  // State
  let suggestions: DidYouMeanSuggestion[] = [];
  let isLoading = false;
  let error: string | null = null;
  let metrics: any = null;
  let debounceTimer: number;
  let lastQuery = '';

  const dispatch = createEventDispatcher<{
    suggestion: { suggestion: string; originalQuery: string };
    metricsUpdate: { metrics: any };
  }>();

  // Reactive statement for query changes
  $: if (query !== lastQuery && query.trim().length > 0) {
    handleQueryChange(query);
    lastQuery = query;
  } else if (query.trim().length === 0) {
    suggestions = [];
    error = null;
    metrics = null;
  }

  async function handleQueryChange(newQuery: string) {
    // Clear existing timer
    if (debounceTimer) {
      clearTimeout(debounceTimer);
    }

    // Debounce the request
    debounceTimer = setTimeout(async () => {
      await fetchSuggestions(newQuery);
    }, debounceMs);
  }

  async function fetchSuggestions(searchQuery: string) {
    if (!searchQuery.trim()) return;

    isLoading = true;
    error = null;

    try {
      const suggestionQuery: DidYouMeanQuery = {
        originalQuery: searchQuery,
        userIntent,
        context: Object.keys(context).length > 0 ? context : undefined,
        options: {
          maxSuggestions,
          similarityThreshold: threshold,
          includeTypos: showTypos,
          includeSemanticSuggestions: showSemantic,
          graphDepth: 3
        }
      };

      const result = await didYouMeanService.generateSuggestions(suggestionQuery);
      
      suggestions = result.suggestions;
      metrics = {
        processingTimeMs: result.processingTimeMs,
        cacheInfo: result.cacheInfo,
        graphContext: result.graphContext
      };

      dispatch('metricsUpdate', { metrics });

    } catch (err: any) {
      console.error('Failed to fetch suggestions:', err);
      error = 'Failed to load suggestions';
      suggestions = [];
    } finally {
      isLoading = false;
    }
  }

  function handleSuggestionClick(suggestion: DidYouMeanSuggestion) {
    dispatch('suggestion', {
      suggestion: suggestion.suggestion,
      originalQuery: query
    });
  }

  function getSuggestionIcon(type: DidYouMeanSuggestion['suggestionType']): string {
    switch (type) {
      case 'typo': return '🔧';
      case 'semantic': return '🎯';
      case 'completion': return '💡';
      case 'graph_neighbor': return '🔗';
      case 'synonym': return '📝';
      default: return '💭';
    }
  }

  function getSuggestionColor(confidence: number): string {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-blue-600';
    if (confidence >= 0.4) return 'text-yellow-600';
    return 'text-gray-600';
  }

  function getConfidenceBar(confidence: number): string {
    const percentage = Math.round(confidence * 100);
    const color = confidence >= 0.7 ? 'bg-green-500' : confidence >= 0.5 ? 'bg-blue-500' : 'bg-yellow-500';
    return `${color} h-1 rounded-full transition-all duration-300`;
  }

  onMount(() => {
    return () => {
      if (debounceTimer) {
        clearTimeout(debounceTimer);
      }
    };
  });
</script>

{#if query.trim().length > 0}
  <div class="did-you-mean-container relative">
    <!-- Loading Indicator -->
    {#if isLoading}
      <div 
        class="loading-indicator flex items-center space-x-2 p-3 bg-blue-50 rounded-lg border border-blue-200"
        in:fade={{ duration: 200 }}
      >
        <div class="animate-spin w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full"></div>
        <span class="text-sm text-blue-700">Finding suggestions...</span>
        {#if showMetrics && metrics}
          <span class="text-xs text-blue-600">
            ({metrics.cacheInfo.quicStreamsUsed} QUIC streams)
          </span>
        {/if}
      </div>
    {/if}

    <!-- Error State -->
    {#if error}
      <div 
        class="error-container p-3 bg-red-50 rounded-lg border border-red-200"
        in:fade={{ duration: 200 }}
      >
        <div class="flex items-center space-x-2">
          <span class="text-red-500">⚠️</span>
          <span class="text-sm text-red-700">{error}</span>
        </div>
      </div>
    {/if}

    <!-- Suggestions List -->
    {#if suggestions.length > 0 && !isLoading}
      <div 
        class="suggestions-container bg-white rounded-lg border border-gray-200 shadow-lg max-h-80 overflow-y-auto"
        in:fly={{ y: -10, duration: 300, easing: quintOut }}
      >
        <!-- Header -->
        <div class="suggestions-header p-3 border-b border-gray-100 bg-gray-50">
          <div class="flex items-center justify-between">
            <span class="text-sm font-medium text-gray-700">
              Did you mean...
            </span>
            {#if showMetrics && metrics}
              <div class="flex items-center space-x-3 text-xs text-gray-500">
                <span>⚡ {metrics.processingTimeMs.toFixed(1)}ms</span>
                <span>🚀 {metrics.cacheInfo.quicStreamsUsed} streams</span>
                {#if metrics.graphContext}
                  <span>🕸️ {metrics.graphContext.nodesTraversed} nodes</span>
                {/if}
              </div>
            {/if}
          </div>
        </div>

        <!-- Suggestion Items -->
        <div class="suggestions-list">
          {#each suggestions as suggestion, index}
            <button
              class="suggestion-item w-full text-left p-3 hover:bg-gray-50 focus:bg-blue-50 focus:outline-none transition-colors duration-150 border-b border-gray-100 last:border-b-0"
              on:click={() => handleSuggestionClick(suggestion)}
              in:fly={{ y: 10, duration: 200, delay: index * 50 }}
            >
              <div class="flex items-start justify-between space-x-3">
                <!-- Main Content -->
                <div class="flex-1 min-w-0">
                  <div class="flex items-center space-x-2 mb-1">
                    <span class="suggestion-icon text-lg" title={suggestion.suggestionType}>
                      {getSuggestionIcon(suggestion.suggestionType)}
                    </span>
                    <span class="suggestion-text font-medium text-gray-900 truncate">
                      {suggestion.suggestion}
                    </span>
                  </div>
                  
                  <div class="text-sm text-gray-600 mb-2">
                    {suggestion.reasoning}
                  </div>
                  
                  <!-- Confidence Bar -->
                  <div class="confidence-container">
                    <div class="w-full bg-gray-200 rounded-full h-1">
                      <div 
                        class={getConfidenceBar(suggestion.confidence)}
                        style="width: {Math.round(suggestion.confidence * 100)}%"
                      ></div>
                    </div>
                  </div>
                </div>

                <!-- Metadata -->
                <div class="flex flex-col items-end space-y-1 text-xs">
                  <span class={`confidence-score font-medium ${getSuggestionColor(suggestion.confidence)}`}>
                    {Math.round(suggestion.confidence * 100)}%
                  </span>
                  
                  {#if suggestion.slotKey}
                    <span class="cache-indicator px-2 py-1 bg-green-100 text-green-800 rounded-full" title="Cached results available">
                      🚀 Cached
                    </span>
                  {/if}
                  
                  {#if suggestion.metadata?.popularQuery}
                    <span class="popularity-indicator px-2 py-1 bg-blue-100 text-blue-800 rounded-full" title="Popular query">
                      🔥 Popular
                    </span>
                  {/if}
                  
                  {#if suggestion.metadata?.practiceArea}
                    <span class="practice-area text-gray-500" title="Practice area">
                      {suggestion.metadata.practiceArea}
                    </span>
                  {/if}
                </div>
              </div>
            </button>
          {/each}
        </div>

        <!-- Footer with Graph Context -->
        {#if showMetrics && metrics?.graphContext}
          <div class="suggestions-footer p-3 border-t border-gray-100 bg-gray-50">
            <div class="flex items-center justify-between text-xs text-gray-600">
              <div class="flex items-center space-x-4">
                <span>Graph depth: {metrics.graphContext.maxDepth}</span>
                <span>Concepts: {metrics.graphContext.relevantConcepts.length}</span>
              </div>
              <div class="flex items-center space-x-2">
                <span>Cache hits: {metrics.cacheInfo.cacheHits}</span>
                <span>Misses: {metrics.cacheInfo.cacheMisses}</span>
              </div>
            </div>
            
            {#if metrics.graphContext.relevantConcepts.length > 0}
              <div class="relevant-concepts mt-2">
                <div class="text-xs text-gray-500 mb-1">Related concepts:</div>
                <div class="flex flex-wrap gap-1">
                  {#each metrics.graphContext.relevantConcepts.slice(0, 5) as concept}
                    <span class="concept-tag px-2 py-1 bg-blue-100 text-blue-700 rounded text-xs">
                      {concept}
                    </span>
                  {/each}
                  {#if metrics.graphContext.relevantConcepts.length > 5}
                    <span class="text-xs text-gray-500">
                      +{metrics.graphContext.relevantConcepts.length - 5} more
                    </span>
                  {/if}
                </div>
              </div>
            {/if}
          </div>
        {/if}
      </div>
    {/if}

    <!-- No Suggestions State -->
    {#if !isLoading && !error && suggestions.length === 0 && query.trim().length > 0}
      <div 
        class="no-suggestions p-3 bg-gray-50 rounded-lg border border-gray-200"
        in:fade={{ duration: 200 }}
      >
        <div class="flex items-center space-x-2">
          <span class="text-gray-400">💭</span>
          <span class="text-sm text-gray-600">No suggestions found for "{query}"</span>
        </div>
      </div>
    {/if}
  </div>
{/if}

<style>
  .did-you-mean-container {
    font-family: system-ui, -apple-system, sans-serif;
  }

  .suggestion-item:focus {
    box-shadow: inset 2px 0 0 #3b82f6;
  }

  .confidence-score {
    font-variant-numeric: tabular-nums;
  }

  .animate-spin {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  /* Accessibility improvements */
  .suggestion-item:focus {
    outline: 2px solid #3b82f6;
    outline-offset: -2px;
  }

  /* Mobile responsiveness */
  @media (max-width: 640px) {
    .suggestions-container {
      max-height: 60vh;
    }
    
    .suggestion-item {
      padding: 0.75rem;
    }
    
    .suggestions-header,
    .suggestions-footer {
      padding: 0.75rem;
    }
  }
</style>