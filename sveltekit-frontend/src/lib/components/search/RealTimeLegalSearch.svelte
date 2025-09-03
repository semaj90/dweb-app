<!--
  Real-Time Legal Search Component
  Optimized for Svelte 5 + SvelteKit 2 + bits-ui v2
  Features: WebSocket streaming, NATS messaging, vector search
-->

<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { derived } from 'svelte/store';
  import { debounce } from 'lodash-es';
  
  // TODO: Replace with melt-ui equivalent when available
  // import { Combobox } from 'bits-ui';
  import * as Command from '$lib/components/ui/command/index.js';
  
  // Real-time search service
  import { useRealTimeSearch } from '$lib/services/real-time-search.js';
  
  // Icons
  import {
    Search,
    Loader2,
    Zap,
    AlertCircle,
    TrendingUp,
    Filter,
    Settings,
    Wifi,
    WifiOff
  } from 'lucide-svelte';

  // Props with enhanced configuration
  interface Props {
    placeholder?: string;
    categories?: Array<'cases' | 'evidence' | 'precedents' | 'statutes' | 'criminals' | 'documents'>;
    enableRealTime?: boolean;
    enableVectorSearch?: boolean;
    enableAI?: boolean;
    maxResults?: number;
    autoSearch?: boolean;
    class?: string;
  }

  let {
    placeholder = 'Search cases, evidence, precedents, statutes...',
    categories = ['cases', 'evidence', 'precedents', 'statutes'],
    enableRealTime = true,
    enableVectorSearch = true,
    enableAI = true,
    maxResults = 20,
    autoSearch = true,
    class = ''
  }: Props = $props();

  // Real-time search hooks
  const { state, isReady, hasResults, searchStatus, search, disconnect } = useRealTimeSearch();

  // Local state
  let inputValue = $state('');
  let open = $state(false);
  let selectedResult: any = $state(null);
  let showFilters = $state(false);
  let searchHistory: string[] = $state([]);

  // Reactive computations
  let filteredResults = $derived($state.results.slice(0, maxResults));
  let isStreaming = $derived($searchStatus === 'searching' && enableRealTime);
  let connectionStatus = $derived($state.connectionStatus);
  let searchMetrics = $derived($state.searchMetrics);

  // Enhanced debounced search
  const debouncedSearch = debounce(async (query: string) => {
    if (!query.trim() || query.length < 2) return;

    try {
      await search(query, {
        categories,
        vectorSearch: enableVectorSearch,
        streamResults: enableRealTime,
        includeAI: enableAI
      });

      // Add to search history
      if (!searchHistory.includes(query)) {
        searchHistory = [query, ...searchHistory.slice(0, 9)]; // Keep last 10 searches
      }
    } catch (error) {
      console.error('❌ Search failed:', error);
    }
  }, 300);

  // Handle input changes
  function handleInputChange(value: string) {
    inputValue = value;
    
    if (autoSearch && value.trim().length >= 2) {
      debouncedSearch(value);
    }
  }

  // Handle result selection
  function handleSelect(result: any) {
    selectedResult = result;
    inputValue = result.title;
    open = false;
    
    // Dispatch custom event for parent components
    const event = new CustomEvent('select', {
      detail: { result, query: inputValue }
    });
    dispatchEvent(event);
  }

  // Manual search trigger
  function handleSearch() {
    if (inputValue.trim().length >= 2) {
      debouncedSearch(inputValue);
      open = true;
    }
  }

  // Clear search
  function handleClear() {
    inputValue = '';
    selectedResult = null;
    open = false;
  }

  // Get result type icon
  function getResultTypeIcon(type: string) {
    switch (type) {
      case 'case': return '⚖️';
      case 'evidence': return '🔍';
      case 'precedent': return '📚';
      case 'statute': return '📜';
      case 'criminal': return '👤';
      case 'document': return '📄';
      default: return '📋';
    }
  }

  // Get connection status color
  function getConnectionStatusColor(status: string) {
    switch (status) {
      case 'connected': return 'text-green-500';
      case 'connecting': return 'text-yellow-500';
      case 'error': return 'text-red-500';
      default: return 'text-gray-500';
    }
  }

  // Component lifecycle
  onMount(() => {
    console.log('🚀 Real-Time Legal Search Component mounted');
  });

  onDestroy(() => {
    disconnect();
    console.log('🔌 Real-Time Legal Search Component destroyed');
  });
</script>

<!-- Enhanced Real-Time Search Interface -->
<div class="real-time-search-container {className}">
  <!-- Search Header with Status -->
  <div class="flex items-center gap-2 mb-4">
    <div class="flex-1">
      <h3 class="text-lg font-semibold text-gray-900">Legal AI Search</h3>
      <p class="text-sm text-gray-600">
        Real-time search with vector similarity and AI enhancement
      </p>
    </div>
    
    <!-- Connection Status -->
    <div class="flex items-center gap-2">
      {#if enableRealTime}
        <div class="flex items-center gap-1 text-xs">
          {#if connectionStatus === 'connected'}
            <Wifi class="w-3 h-3 text-green-500" />
            <span class="text-green-600">Connected</span>
          {:else if connectionStatus === 'connecting'}
            <Loader2 class="w-3 h-3 text-yellow-500 animate-spin" />
            <span class="text-yellow-600">Connecting</span>
          {:else if connectionStatus === 'error'}
            <WifiOff class="w-3 h-3 text-red-500" />
            <span class="text-red-600">Offline</span>
          {:else}
            <WifiOff class="w-3 h-3 text-gray-400" />
            <span class="text-gray-500">Disconnected</span>
          {/if}
        </div>
      {/if}

      <!-- Search Metrics -->
      {#if searchMetrics.totalQueries > 0}
        <div class="text-xs text-gray-500">
          {searchMetrics.totalQueries} queries • 
          {searchMetrics.averageResponseTime}ms avg
        </div>
      {/if}
    </div>
  </div>

  <!-- Enhanced Search Input -->
  <Combobox.Root bind:open bind:inputValue onInputValueChange={handleInputChange}>
    <div class="relative">
      <!-- Search Input with Enhanced Styling -->
      <Combobox.Input
        class="flex h-12 w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm 
               placeholder:text-gray-500 focus:border-blue-500 focus:outline-none focus:ring-2 
               focus:ring-blue-200 disabled:cursor-not-allowed disabled:opacity-50
               {isStreaming ? 'pr-12' : 'pr-10'}"
        {placeholder}
        autocomplete="off"
        spellcheck="false"
      />

      <!-- Search Button & Status Indicators -->
      <div class="absolute inset-y-0 right-0 flex items-center pr-3">
        {#if isStreaming}
          <Loader2 class="h-4 w-4 animate-spin text-blue-500" />
        {:else if enableRealTime && connectionStatus === 'connected'}
          <Zap class="h-4 w-4 text-green-500" />
        {:else}
          <button
            type="button"
            class="p-1 hover:bg-gray-100 rounded"
            on:onclick={handleSearch}
            disabled={!inputValue.trim()}
          >
            <Search class="h-4 w-4 text-gray-500" />
          </button>
        {/if}
      </div>
    </div>

    <!-- Enhanced Search Results -->
    <Combobox.Content
      class="absolute z-50 mt-2 max-h-96 w-full overflow-y-auto rounded-lg border border-gray-200 
             bg-white shadow-lg data-[state=open]:animate-in data-[state=closed]:animate-out
             data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0"
    >
      {#if $state.error}
        <!-- Error State -->
        <div class="flex items-center gap-2 p-4 text-red-600">
          <AlertCircle class="h-4 w-4" />
          <span class="text-sm">{$state.error}</span>
        </div>
      {:else if isStreaming}
        <!-- Streaming State -->
        <div class="flex items-center gap-2 p-4 text-blue-600">
          <Loader2 class="h-4 w-4 animate-spin" />
          <span class="text-sm">Searching with AI enhancement...</span>
        </div>
        
        <!-- Streaming Results -->
        {#each filteredResults as result, index (result.id)}
          <Combobox.Item
            value={result.id}
            class="relative flex cursor-default select-none items-start gap-3 rounded-sm px-3 py-2 
                   text-sm outline-none hover:bg-gray-50 data-[highlighted]:bg-blue-50
                   {result.realTime ? 'animate-pulse border-l-2 border-blue-400' : ''}"
            onSelect={() => handleSelect(result)}
          >
            <!-- Result Type Icon -->
            <div class="mt-1 text-lg">
              {getResultTypeIcon(result.type)}
            </div>

            <!-- Result Content -->
            <div class="flex-1 min-w-0">
              <div class="flex items-start justify-between gap-2">
                <div class="font-medium text-gray-900 truncate">
                  {result.title}
                </div>
                <div class="flex items-center gap-1 text-xs text-gray-500 shrink-0">
                  {#if result.realTime}
                    <TrendingUp class="w-3 h-3 text-blue-500" />
                  {/if}
                  <span>{(result.score * 100).toFixed(0)}%</span>
                </div>
              </div>
              
              <div class="text-xs text-gray-600 mt-1 line-clamp-2">
                {result.content.substring(0, 120)}...
              </div>
              
              <!-- Enhanced Metadata -->
              <div class="flex items-center gap-2 mt-2 text-xs text-gray-500">
                <span class="capitalize bg-gray-100 px-2 py-1 rounded">
                  {result.type}
                </span>
                {#if result.metadata.jurisdiction}
                  <span>{result.metadata.jurisdiction}</span>
                {/if}
                {#if result.metadata.status}
                  <span class="capitalize">{result.metadata.status}</span>
                {/if}
                {#if result.realTime}
                  <span class="text-blue-500 font-medium">Live</span>
                {/if}
              </div>
            </div>
          </Combobox.Item>
        {/each}
      {:else if filteredResults.length > 0}
        <!-- Standard Results -->
        {#each filteredResults as result (result.id)}
          <Combobox.Item
            value={result.id}
            class="relative flex cursor-default select-none items-start gap-3 rounded-sm px-3 py-2 
                   text-sm outline-none hover:bg-gray-50 data-[highlighted]:bg-blue-50"
            onSelect={() => handleSelect(result)}
          >
            <!-- Result Type Icon -->
            <div class="mt-1 text-lg">
              {getResultTypeIcon(result.type)}
            </div>

            <!-- Result Content -->
            <div class="flex-1 min-w-0">
              <div class="flex items-start justify-between gap-2">
                <div class="font-medium text-gray-900 truncate">
                  {result.title}
                </div>
                <div class="text-xs text-gray-500 shrink-0">
                  {(result.score * 100).toFixed(0)}%
                </div>
              </div>
              
              <div class="text-xs text-gray-600 mt-1 line-clamp-2">
                {result.content.substring(0, 120)}...
              </div>
              
              <!-- Metadata Tags -->
              <div class="flex items-center gap-2 mt-2 text-xs text-gray-500">
                <span class="capitalize bg-gray-100 px-2 py-1 rounded">
                  {result.type}
                </span>
                {#if result.metadata.jurisdiction}
                  <span>{result.metadata.jurisdiction}</span>
                {/if}
                {#if result.metadata.date}
                  <span>{new Date(result.metadata.date).toLocaleDateString()}</span>
                {/if}
              </div>

              <!-- Highlights -->
              {#if result.highlights && result.highlights.length > 0}
                <div class="mt-2 text-xs text-blue-600">
                  <span class="font-medium">Highlights:</span>
                  {result.highlights[0].substring(0, 80)}...
                </div>
              {/if}
            </div>
          </Combobox.Item>
        {/each}

        <!-- Search Statistics -->
        <div class="border-t border-gray-200 px-3 py-2 text-xs text-gray-500">
          {filteredResults.length} results • 
          {searchMetrics.lastQueryTime}ms • 
          {enableVectorSearch ? 'Vector' : 'Text'} + {enableAI ? 'AI' : 'Standard'} search
        </div>
      {:else if inputValue.trim().length >= 2}
        <!-- No Results -->
        <div class="p-4 text-center text-sm text-gray-500">
          <Search class="h-8 w-8 mx-auto mb-2 text-gray-300" />
          <p>No results found for "{inputValue}"</p>
          <p class="text-xs mt-1">Try adjusting your search terms or categories</p>
        </div>
      {:else if searchHistory.length > 0}
        <!-- Search History -->
        <div class="p-3">
          <div class="text-xs font-medium text-gray-700 mb-2">Recent Searches</div>
          {#each searchHistory.slice(0, 5) as query}
            <button
              type="button"
              class="block w-full text-left text-xs text-gray-600 hover:text-gray-900 py-1"
              on:onclick={() => handleInputChange(query)}
            >
              {query}
            </button>
          {/each}
        </div>
      {/if}
    </Combobox.Content>
  </Combobox.Root>

  <!-- Search Status Bar -->
  {#if enableRealTime}
    <div class="mt-2 flex items-center justify-between text-xs text-gray-500">
      <div class="flex items-center gap-4">
        <span class="flex items-center gap-1">
          <div class="w-2 h-2 rounded-full {connectionStatus === 'connected' ? 'bg-green-400' : 'bg-gray-400'}"></div>
          Real-time {connectionStatus}
        </span>
        {#if enableVectorSearch}
          <span class="flex items-center gap-1">
            <Zap class="w-3 h-3" />
            Vector similarity
          </span>
        {/if}
        {#if enableAI}
          <span class="flex items-center gap-1">
            ✨ AI enhanced
          </span>
        {/if}
      </div>
      
      {#if searchMetrics.totalQueries > 0}
        <div>
          Performance: {searchMetrics.averageResponseTime}ms avg
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .real-time-search-container {
    @apply relative w-full max-w-2xl mx-auto;
  }
  
  .line-clamp-2 {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
</style>