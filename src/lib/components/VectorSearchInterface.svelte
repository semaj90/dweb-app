<script lang="ts">
  import { Button } from 'bits-ui';
  import { Input } from 'bits-ui';
  import { Card } from 'bits-ui';
  import { Badge } from 'bits-ui';
  import { Separator } from 'bits-ui';
  import { Select } from 'bits-ui';
  import { Checkbox } from 'bits-ui';
  import { createEventDispatcher } from 'svelte';
  import { fly, fade } from 'svelte/transition';
  import { aiPipeline } from '$lib/ai/processing-pipeline.js';
  import type { LegalDocument } from '$lib/database/schema/legal-documents.js';
  
  // Props
  let { 
    placeholder = "Search legal documents...",
    showFilters = true,
    maxResults = 20,
    class: className = ''
  } = $props();

  // State
  let searchQuery = $state('');
  let isSearching = $state(false);
  let searchResults = $state<Array<LegalDocument & { similarity: number }>>([]);
  let selectedDocument = $state<LegalDocument | null>(null);
  let searchHistory = $state<string[]>([]);
  let totalSearchTime = $state(0);

  // Filters
  let filters = $state({
    documentType: 'all',
    practiceArea: 'all',
    jurisdiction: 'all',
    dateRange: 'all',
    minSimilarity: 0.6
  });

  // Filter options
  const documentTypes = [
    { value: 'all', label: 'All Types' },
    { value: 'contract', label: 'Contract' },
    { value: 'motion', label: 'Motion' },
    { value: 'evidence', label: 'Evidence' },
    { value: 'correspondence', label: 'Correspondence' },
    { value: 'brief', label: 'Brief' },
    { value: 'regulation', label: 'Regulation' },
    { value: 'case_law', label: 'Case Law' }
  ];

  const practiceAreas = [
    { value: 'all', label: 'All Practice Areas' },
    { value: 'corporate', label: 'Corporate' },
    { value: 'litigation', label: 'Litigation' },
    { value: 'intellectual_property', label: 'Intellectual Property' },
    { value: 'employment', label: 'Employment' },
    { value: 'real_estate', label: 'Real Estate' },
    { value: 'criminal', label: 'Criminal' },
    { value: 'family', label: 'Family' },
    { value: 'tax', label: 'Tax' },
    { value: 'immigration', label: 'Immigration' },
    { value: 'environmental', label: 'Environmental' }
  ];

  const jurisdictions = [
    { value: 'all', label: 'All Jurisdictions' },
    { value: 'federal', label: 'Federal' },
    { value: 'state', label: 'State' },
    { value: 'local', label: 'Local' }
  ];

  // Event dispatcher
  const dispatch = createEventDispatcher<{
    search: { query: string; results: Array<LegalDocument & { similarity: number }> };
    select: { document: LegalDocument };
    filter: { filters: typeof filters };
  }>();

  /**
   * Perform semantic search
   */
  async function performSearch() {
    if (!searchQuery.trim() || isSearching) return;

    isSearching = true;
    const startTime = Date.now();

    try {
      // Build search options
      const searchOptions = {
        limit: maxResults,
        documentType: filters.documentType !== 'all' ? filters.documentType : undefined,
        practiceArea: filters.practiceArea !== 'all' ? filters.practiceArea : undefined,
        jurisdiction: filters.jurisdiction !== 'all' ? filters.jurisdiction : undefined,
        useCache: true
      };

      // Perform search
      const results = await aiPipeline.semanticSearch(searchQuery, searchOptions);
      
      // Filter by similarity threshold
      const filteredResults = results.filter(result => result.similarity >= filters.minSimilarity);

      searchResults = filteredResults;
      totalSearchTime = Date.now() - startTime;

      // Add to search history
      if (!searchHistory.includes(searchQuery)) {
        searchHistory = [searchQuery, ...searchHistory.slice(0, 9)]; // Keep last 10 searches
      }

      dispatch('search', { query: searchQuery, results: filteredResults });

    } catch (error) {
      console.error('Search failed:', error);
      searchResults = [];
    } finally {
      isSearching = false;
    }
  }

  /**
   * Handle search input
   */
  function handleSearchInput(event: Event) {
    const target = event.target as HTMLInputElement;
    searchQuery = target.value;
  }

  /**
   * Handle search on Enter key
   */
  function handleKeyDown(event: KeyboardEvent) {
    if (event.key === 'Enter') {
      performSearch();
    }
  }

  /**
   * Select a document
   */
  function selectDocument(document: LegalDocument & { similarity: number }) {
    selectedDocument = document;
    dispatch('select', { document });
  }

  /**
   * Use search history item
   */
  function useHistoryItem(query: string) {
    searchQuery = query;
    performSearch();
  }

  /**
   * Clear search results
   */
  function clearSearch() {
    searchQuery = '';
    searchResults = [];
    selectedDocument = null;
  }

  /**
   * Reset filters
   */
  function resetFilters() {
    filters = {
      documentType: 'all',
      practiceArea: 'all',
      jurisdiction: 'all',
      dateRange: 'all',
      minSimilarity: 0.6
    };
    dispatch('filter', { filters });
  }

  /**
   * Handle filter changes
   */
  function handleFilterChange() {
    dispatch('filter', { filters });
    if (searchQuery) {
      performSearch();
    }
  }

  // Helper functions
  function formatSimilarity(similarity: number): string {
    return `${Math.round(similarity * 100)}%`;
  }

  function formatDate(date: string | Date): string {
    return new Date(date).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  }

  function getDocumentTypeColor(type: string): string {
    const colors: Record<string, string> = {
      contract: 'bg-blue-100 text-blue-800',
      motion: 'bg-green-100 text-green-800',
      evidence: 'bg-yellow-100 text-yellow-800',
      correspondence: 'bg-purple-100 text-purple-800',
      brief: 'bg-red-100 text-red-800',
      regulation: 'bg-indigo-100 text-indigo-800',
      case_law: 'bg-gray-100 text-gray-800'
    };
    return colors[type] || 'bg-gray-100 text-gray-800';
  }

  function getSimilarityColor(similarity: number): string {
    if (similarity >= 0.9) return 'text-green-600';
    if (similarity >= 0.8) return 'text-blue-600';
    if (similarity >= 0.7) return 'text-yellow-600';
    return 'text-gray-600';
  }

  function truncateText(text: string, maxLength: number): string {
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
  }
</script>

<div class={`vector-search-interface ${className}`}>
  <!-- Search Header -->
  <div class="search-header bg-white rounded-lg shadow-sm border p-6 mb-6">
    <h2 class="text-2xl font-bold text-gray-900 mb-4">
      Semantic Document Search
    </h2>
    
    <div class="flex space-x-4">
      <!-- Search Input -->
      <div class="flex-1 relative">
        <Input.Root
          type="text"
          {placeholder}
          value={searchQuery}
          on:input={handleSearchInput}
          on:keydown={handleKeyDown}
          disabled={isSearching}
          class="w-full pr-12"
        />
        <div class="absolute inset-y-0 right-0 flex items-center pr-3">
          {#if isSearching}
            <div class="animate-spin h-5 w-5 border-2 border-blue-600 border-t-transparent rounded-full"></div>
          {:else}
            <svg class="h-5 w-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          {/if}
        </div>
      </div>

      <!-- Search Button -->
      <Button.Root
        on:click={performSearch}
        disabled={!searchQuery.trim() || isSearching}
        class="px-6"
      >
        {isSearching ? 'Searching...' : 'Search'}
      </Button.Root>

      <!-- Clear Button -->
      {#if searchQuery || searchResults.length > 0}
        <Button.Root
          variant="outline"
          on:click={clearSearch}
          class="px-4"
        >
          Clear
        </Button.Root>
      {/if}
    </div>

    <!-- Search History -->
    {#if searchHistory.length > 0}
      <div class="mt-4">
        <p class="text-sm text-gray-600 mb-2">Recent searches:</p>
        <div class="flex flex-wrap gap-2">
          {#each searchHistory as historyItem}
            <button
              type="button"
              class="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded-full transition-colors"
              on:click={() => useHistoryItem(historyItem)}
            >
              {historyItem}
            </button>
          {/each}
        </div>
      </div>
    {/if}
  </div>

  <div class="flex gap-6">
    <!-- Filters Sidebar -->
    {#if showFilters}
      <div class="w-80 flex-shrink-0">
        <Card.Root class="p-6 sticky top-6">
          <div class="flex items-center justify-between mb-4">
            <h3 class="text-lg font-semibold text-gray-900">Filters</h3>
            <Button.Root
              variant="ghost"
              size="sm"
              on:click={resetFilters}
              class="text-sm"
            >
              Reset
            </Button.Root>
          </div>

          <div class="space-y-6">
            <!-- Document Type Filter -->
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">
                Document Type
              </label>
              <Select.Root
                bind:selected={filters.documentType}
                onSelectedChange={handleFilterChange}
              >
                <Select.Trigger class="w-full">
                  <Select.Value placeholder="Select type..." />
                </Select.Trigger>
                <Select.Content>
                  {#each documentTypes as type}
                    <Select.Item value={type.value}>
                      {type.label}
                    </Select.Item>
                  {/each}
                </Select.Content>
              </Select.Root>
            </div>

            <!-- Practice Area Filter -->
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">
                Practice Area
              </label>
              <Select.Root
                bind:selected={filters.practiceArea}
                onSelectedChange={handleFilterChange}
              >
                <Select.Trigger class="w-full">
                  <Select.Value placeholder="Select area..." />
                </Select.Trigger>
                <Select.Content>
                  {#each practiceAreas as area}
                    <Select.Item value={area.value}>
                      {area.label}
                    </Select.Item>
                  {/each}
                </Select.Content>
              </Select.Root>
            </div>

            <!-- Jurisdiction Filter -->
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">
                Jurisdiction
              </label>
              <Select.Root
                bind:selected={filters.jurisdiction}
                onSelectedChange={handleFilterChange}
              >
                <Select.Trigger class="w-full">
                  <Select.Value placeholder="Select jurisdiction..." />
                </Select.Trigger>
                <Select.Content>
                  {#each jurisdictions as jurisdiction}
                    <Select.Item value={jurisdiction.value}>
                      {jurisdiction.label}
                    </Select.Item>
                  {/each}
                </Select.Content>
              </Select.Root>
            </div>

            <!-- Similarity Threshold -->
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">
                Minimum Similarity: {formatSimilarity(filters.minSimilarity)}
              </label>
              <input
                type="range"
                min="0.5"
                max="1"
                step="0.05"
                bind:value={filters.minSimilarity}
                on:change={handleFilterChange}
                class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
              <div class="flex justify-between text-xs text-gray-500 mt-1">
                <span>50%</span>
                <span>100%</span>
              </div>
            </div>
          </div>
        </Card.Root>
      </div>
    {/if}

    <!-- Results Area -->
    <div class="flex-1">
      <!-- Results Header -->
      {#if searchResults.length > 0}
        <div class="bg-white rounded-lg shadow-sm border p-4 mb-4">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-lg font-semibold text-gray-900">
                {searchResults.length} results found
              </p>
              <p class="text-sm text-gray-600">
                Search completed in {totalSearchTime}ms
              </p>
            </div>
            <div class="text-sm text-gray-500">
              Query: "{searchQuery}"
            </div>
          </div>
        </div>
      {/if}

      <!-- Search Results -->
      <div class="space-y-4">
        {#each searchResults as result, index}
          <div
            transition:fly={{ y: 20, delay: index * 50 }}
            class="bg-white rounded-lg shadow-sm border hover:shadow-md transition-shadow cursor-pointer"
            on:click={() => selectDocument(result)}
            role="button"
            tabindex="0"
            on:keydown={(e) => e.key === 'Enter' && selectDocument(result)}
          >
            <div class="p-6">
              <!-- Result Header -->
              <div class="flex items-start justify-between mb-3">
                <div class="flex-1">
                  <h3 class="text-lg font-semibold text-gray-900 hover:text-blue-600 transition-colors">
                    {result.title}
                  </h3>
                  <div class="flex items-center space-x-3 mt-1">
                    <Badge.Root class={getDocumentTypeColor(result.documentType)}>
                      {result.documentType.replace('_', ' ')}
                    </Badge.Root>
                    {#if result.practiceArea}
                      <Badge.Root variant="outline">
                        {result.practiceArea.replace('_', ' ')}
                      </Badge.Root>
                    {/if}
                    <span class="text-sm text-gray-500">
                      {result.jurisdiction}
                    </span>
                  </div>
                </div>
                <div class="text-right">
                  <div class={`text-lg font-bold ${getSimilarityColor(result.similarity)}`}>
                    {formatSimilarity(result.similarity)}
                  </div>
                  <div class="text-xs text-gray-500">similarity</div>
                </div>
              </div>

              <!-- Content Preview -->
              <div class="text-sm text-gray-700 mb-3">
                {truncateText(result.content, 300)}
              </div>

              <!-- Metadata -->
              <div class="flex items-center justify-between text-xs text-gray-500">
                <div class="flex items-center space-x-4">
                  <span>Created: {formatDate(result.createdAt)}</span>
                  {#if result.fileSize}
                    <span>Size: {Math.round(result.fileSize / 1024)} KB</span>
                  {/if}
                  {#if result.analysisResults?.confidenceLevel}
                    <span>Confidence: {Math.round(result.analysisResults.confidenceLevel * 100)}%</span>
                  {/if}
                </div>
                <div class="flex items-center space-x-2">
                  {#if result.analysisResults?.risks?.length}
                    <Badge.Root variant="destructive" class="text-xs">
                      {result.analysisResults.risks.length} risks
                    </Badge.Root>
                  {/if}
                  <svg class="h-4 w-4 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
                  </svg>
                </div>
              </div>
            </div>
          </div>
        {/each}
      </div>

      <!-- No Results -->
      {#if !isSearching && searchQuery && searchResults.length === 0}
        <div transition:fade class="text-center py-12">
          <div class="mx-auto h-12 w-12 text-gray-400 mb-4">
            <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.172 16.172a4 4 0 015.656 0M9 12h6m-6-4h6m2 5.291A7.962 7.962 0 0112 15c-2.34 0-4.462-.881-6.065-2.325" />
            </svg>
          </div>
          <h3 class="text-lg font-medium text-gray-900 mb-2">No results found</h3>
          <p class="text-gray-500">
            Try adjusting your search query or filters to find more documents
          </p>
        </div>
      {/if}

      <!-- Empty State -->
      {#if !isSearching && !searchQuery}
        <div class="text-center py-12">
          <div class="mx-auto h-12 w-12 text-gray-400 mb-4">
            <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>
          <h3 class="text-lg font-medium text-gray-900 mb-2">Start searching</h3>
          <p class="text-gray-500">
            Enter a search query to find relevant legal documents using AI-powered semantic search
          </p>
        </div>
      {/if}
    </div>
  </div>
</div>

<!-- Document Detail Modal -->
{#if selectedDocument}
  <div
    class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
    on:click={() => selectedDocument = null}
    transition:fade
  >
    <div
      class="bg-white rounded-lg max-w-4xl max-h-[90vh] overflow-hidden"
      on:click|stopPropagation
    >
      <div class="p-6 border-b">
        <div class="flex items-center justify-between">
          <h2 class="text-xl font-bold text-gray-900">
            {selectedDocument.title}
          </h2>
          <button
            type="button"
            class="text-gray-400 hover:text-gray-600"
            on:click={() => selectedDocument = null}
          >
            <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>
      <div class="p-6 overflow-y-auto max-h-[70vh]">
        <div class="prose max-w-none">
          <pre class="whitespace-pre-wrap font-sans text-sm">{selectedDocument.content}</pre>
        </div>
      </div>
    </div>
  </div>
{/if}

<style>
  .vector-search-interface {
    @apply max-w-7xl mx-auto;
  }
</style>