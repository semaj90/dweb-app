<!-- Universal Search Bar for Legal AI Platform -->
<!-- Cases, POI, Evidence, Documents, and more -->

<script lang="ts">
  import { onMount } from 'svelte';
  import { globalSearch, searchServices } from '$lib/services/search-service.js';
  import { hybridSearch } from '$lib/services/hybrid-vector-operations.js';
  import { optimizeComponent, optimizeForAnimations } from '$lib/utils/browser-performance.js';
  import type { SearchResult as ServiceSearchResult } from '$lib/types/search.types.js';
  import type { SearchResult, SearchOptions, SearchSuggestion } from './types.js';

  // Component props
  let {
    placeholder = "Search cases, persons of interest, evidence...",
    showFilters = true,
    showSuggestions = true,
    maxResults = 20,
    theme = 'light' as 'light' | 'dark' | 'yorha',
    onsearch = undefined as ((event: CustomEvent<{ query: string; results: SearchResult[] }>) => void) | undefined,
    onselect = undefined as ((event: CustomEvent<{ result: SearchResult }>) => void) | undefined,
    onclear = undefined as ((event: CustomEvent<void>) => void) | undefined
  } = $props();

  // Component state
  let searchInput = $state('');
  let isSearching = $state(false);
  let showResults = $state(false);
  let results: SearchResult[] = $state([]);
  let suggestions: SearchSuggestion[] = $state([]);
  let selectedCategories = $state(['cases', 'evidence', 'documents']);
  let recentSearches: string[] = $state([]);

  // Search configuration
  // TODO(metrics-ext): Consolidate searchOptions with future cross-index ranking & vector rerank pipeline after metrics auth/anomaly endpoints deployed.
  let searchOptions: SearchOptions = $state({
    categories: ['cases', 'evidence', 'precedents', 'statutes', 'criminals', 'documents'],
    enableVectorSearch: true,
    aiSuggestions: true,
    maxResults,
    similarityThreshold: 0.7,
    includeMetadata: true
  });

  // Modern Svelte 5 event handling - props instead of dispatcher

  // Available search categories
  const searchCategories = [
    { id: 'cases', label: 'Cases', icon: '📁', color: 'blue' },
    { id: 'criminals', label: 'Persons of Interest', icon: '👤', color: 'red' },
    { id: 'evidence', label: 'Evidence', icon: '🔍', color: 'green' },
    { id: 'documents', label: 'Documents', icon: '📄', color: 'purple' },
    { id: 'precedents', label: 'Precedents', icon: '⚖️', color: 'yellow' },
    { id: 'statutes', label: 'Statutes', icon: '📖', color: 'indigo' }
  ];

  // Recent searches and suggestions
let trendingSearches = $state([
    'fraud investigation',
    'corporate embezzlement',
    'witness testimony',
    'financial crimes',
    'evidence chain custody'
  ]);

  // Debounced search
let searchTimeout = $state<NodeJS.Timeout | null>(null);

  $effect(() => {
    if (searchInput) {
      if (searchTimeout) clearTimeout(searchTimeout);
      searchTimeout = setTimeout(() => performSearch(), 300);
    } else {
      results = [];
      showResults = false;
    }
  });

  // Load recent searches from localStorage and optimize component
  onMount(() => {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem('recentSearches');
      if (stored) {
        recentSearches = JSON.parse(stored);
      }

      // Apply Chrome Windows GPU optimizations
      const searchContainer = document.querySelector('.universal-search-container') as HTMLElement;
      if (searchContainer) {
        optimizeComponent(searchContainer);
      }

      // Optimize search results container for animations
      const resultsContainer = document.querySelector('.search-results-dropdown') as HTMLElement;
      if (resultsContainer) {
        optimizeForAnimations(resultsContainer);
      }
    }
  });

  async function performSearch() {
    if (!searchInput.trim() || isSearching) return;

    isSearching = true;
    showResults = true;

    try {
      // Use the new unified search API endpoint with Loki.js fuzzy search
      const response = await fetch('/api/search/unified', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query: searchInput,
          categories: selectedCategories,
          enableVectorSearch: searchOptions.enableVectorSearch,
          aiSuggestions: searchOptions.aiSuggestions,
          maxResults: searchOptions.maxResults,
          similarityThreshold: searchOptions.similarityThreshold,
          includeMetadata: searchOptions.includeMetadata
        })
      });

      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`);
      }

      const searchData = await response.json();

      if (searchData.success) {
        // Transform API results to component format
        results = searchData.results.map(result => ({
          id: result.id,
          title: result.title,
          type: result.type,
          content: result.content,
          score: result.score || result.similarity || 0,
          metadata: {
            ...result.metadata,
            date: result.createdAt ? new Date(result.createdAt).toISOString().split('T')[0] : new Date().toISOString().split('T')[0]
          }
        }));

        // Generate AI suggestions if enabled and not provided by API
        if (searchOptions.aiSuggestions && (!searchData.suggestions || searchData.suggestions.length === 0)) {
          suggestions = await generateSearchSuggestions(searchInput);
        } else {
          suggestions = searchData.suggestions || [];
        }
      } else {
        console.error('Search API returned error:', searchData.error);
        results = [];
      }

      // Save to recent searches
      saveRecentSearch(searchInput);

      if (onsearch) {
        onsearch(new CustomEvent('search', { detail: { query: searchInput, results } }));
      }

    } catch (error) {
      console.error('Search failed:', error);
      // Fallback to old search method if new API fails
      try {
        const [serviceResults, vectorResults] = await Promise.all([
          globalSearch(searchInput, { limit: Math.floor(maxResults / 2) }),
          searchForLegalEntities(searchInput)
        ]);

        results = [
          ...transformServiceResults(serviceResults),
          ...vectorResults
        ].slice(0, maxResults);
      } catch (fallbackError) {
        console.error('Fallback search also failed:', fallbackError);
        results = [];
      }
    } finally {
      isSearching = false;
    }
  }

  async function searchForLegalEntities(query: string): Promise<SearchResult[]> {
    try {
      // Mock legal entity search - would integrate with actual API
      const mockResults: SearchResult[] = [
        {
          id: 'case-001',
          title: `Case: ${query} Investigation`,
          type: 'case',
          content: `Legal case involving ${query} with multiple evidence items and witness testimonies.`,
          score: 0.9,
          metadata: {
            date: '2024-08-24',
            jurisdiction: 'Federal',
            status: 'Active',
            caseId: 'CASE-001',
            tags: ['investigation', 'active']
          }
        },
        {
          id: 'poi-001',
          title: `Person of Interest: Related to ${query}`,
          type: 'criminal',
          content: `Individual connected to ${query} case with documented criminal history.`,
          score: 0.8,
          metadata: {
            date: '2024-08-20',
            status: 'Under Investigation',
            tags: ['person-of-interest', 'suspect']
          }
        }
      ];

      return mockResults.filter(result =>
        selectedCategories.includes(result.type) ||
        selectedCategories.includes('criminals' as any)
      );
    } catch (error) {
      console.error('Legal entity search failed:', error);
      return [];
    }
  }

  function transformServiceResults(serviceResults: ServiceSearchResult[]): SearchResult[] {
    return serviceResults.map(result => ({
      id: result.id,
      title: result.title,
      type: mapCategoryToType(result.category),
      content: result.description,
      score: 1 - result.score, // Convert score to similarity
      metadata: {
        date: new Date().toISOString().split('T')[0],
        tags: result.tags,
        ...result.metadata
      }
    }));
  }

  function mapCategoryToType(category: string): SearchResult['type'] {
    const mapping: Record<string, SearchResult['type']> = {
      'service': 'document',
      'component': 'document',
      'api': 'document',
      'demo': 'document',
      'documentation': 'document'
    };
    return mapping[category] || 'document';
  }

  async function generateSearchSuggestions(query: string): Promise<SearchSuggestion[]> {
    // Generate contextual suggestions based on query
    const baseSuggestions = [
      { text: `${query} evidence`, category: 'evidence', score: 0.9 },
      { text: `${query} case files`, category: 'cases', score: 0.8 },
      { text: `${query} related persons`, category: 'criminals', score: 0.7 },
      { text: `${query} legal precedents`, category: 'precedents', score: 0.6 }
    ];

    return baseSuggestions.filter(s => s.text !== query);
  }

  function saveRecentSearch(query: string) {
    if (typeof window !== 'undefined') {
      recentSearches = [query, ...recentSearches.filter(s => s !== query)].slice(0, 10);
      localStorage.setItem('recentSearches', JSON.stringify(recentSearches));
    }
  }

  function selectResult(result: SearchResult) {
    if (onselect) {
      onselect(new CustomEvent('select', { detail: { result } }));
    }
    showResults = false;
    searchInput = '';
  }

  function clearSearch() {
    searchInput = '';
    results = [];
    showResults = false;
    if (onclear) {
      onclear(new CustomEvent('clear'));
    }
  }

  function toggleCategory(categoryId: string) {
    if (selectedCategories.includes(categoryId)) {
      selectedCategories = selectedCategories.filter(c => c !== categoryId);
    } else {
      selectedCategories = [...selectedCategories, categoryId];
    }

    if (searchInput) {
      performSearch();
    }
  }

  function selectSuggestion(suggestion: SearchSuggestion) {
    searchInput = suggestion.text;
    performSearch();
  }

  function selectTrendingSearch(trending: string) {
    searchInput = trending;
    performSearch();
  }

  // Theme classes
  let themeClasses = $derived({
    light: 'bg-white text-gray-900 border-gray-300',
    dark: 'bg-gray-800 text-white border-gray-600',
    yorha: 'bg-black/90 text-yellow-400 border-yellow-400/50 shadow-[0_0_10px_rgba(255,255,0,0.3)]'
  }[theme]);

  let inputClasses = $derived({
    light: 'bg-white text-gray-900 border-gray-300 focus:border-blue-500',
    dark: 'bg-gray-700 text-white border-gray-600 focus:border-blue-400',
    yorha: 'bg-black/80 text-yellow-400 border-yellow-400/50 focus:border-yellow-400 focus:shadow-[0_0_15px_rgba(255,255,0,0.5)]'
  }[theme]);
</script>

<div class="universal-search-container nes-search-bar w-full max-w-4xl mx-auto relative gpu-accelerated gpu-smooth-scroll">
  <!-- Main Search Bar -->
  <div class="search-bar-wrapper relative {themeClasses} rounded-lg shadow-lg">
    <div class="flex items-center p-2">
      <!-- Search Icon -->
      <div class="p-2">
        {#if isSearching}
          <div class="w-5 h-5 border-2 border-current border-t-transparent rounded-full animate-spin"></div>
        {:else}
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
          </svg>
        {/if}
      </div>

      <!-- Search Input -->
      <input
        bind:value={searchInput}
        {placeholder}
        class="flex-1 bg-transparent outline-none text-lg {inputClasses.includes('text-') ? '' : 'text-current'}"
        onfocus={() => showResults = true}
        on:keydown={(e) => {
          if (e.key === 'Escape') {
            showResults = false;
            searchInput = '';
          }
        }}
      />

      <!-- Clear Button -->
      {#if searchInput}
        <button
          onclick={clearSearch}
          class="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
          aria-label="Clear search"
        >
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
          </svg>
        </button>
      {/if}

      <!-- Filters Toggle -->
      {#if showFilters}
            <button
              class="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
              onclick={() => {/* Toggle filters panel */}}
              aria-label="Toggle search filters"
            >
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.207A1 1 0 013 6.5V4z"></path>
          </svg>
        </button>
      {/if}
    </div>

    <!-- Category Filters -->
    {#if showFilters}
      <div class="border-t border-current/20 p-3">
        <div class="flex flex-wrap gap-2">
          {#each searchCategories as category}
            <button
              class="category-chip flex items-center gap-1 px-3 py-1 rounded-full text-sm transition-all duration-200"
              class:selected={selectedCategories.includes(category.id)}
              class:bg-blue-100={selectedCategories.includes(category.id) && theme === 'light'}
              class:bg-blue-900={selectedCategories.includes(category.id) && theme === 'dark'}
              class:bg-yellow-400={selectedCategories.includes(category.id) && theme === 'yorha'}
              class:text-blue-800={selectedCategories.includes(category.id) && theme === 'light'}
              class:text-blue-200={selectedCategories.includes(category.id) && theme === 'dark'}
              class:text-black={selectedCategories.includes(category.id) && theme === 'yorha'}
              class:bg-gray-100={!selectedCategories.includes(category.id) && theme === 'light'}
              class:bg-gray-700={!selectedCategories.includes(category.id) && theme === 'dark'}
              class:bg-black={!selectedCategories.includes(category.id) && theme === 'yorha'}
              class:bg-opacity-50={!selectedCategories.includes(category.id) && theme === 'yorha'}
              onclick={() => toggleCategory(category.id)}
            >
              <span>{category.icon}</span>
              <span>{category.label}</span>
              {#if selectedCategories.includes(category.id)}
                <svg class="w-3 h-3 ml-1" fill="currentColor" viewBox="0 0 20 20">
                  <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"></path>
                </svg>
              {/if}
            </button>
          {/each}
        </div>
      </div>
    {/if}
  </div>

  <!-- Search Results / Suggestions -->
  {#if showResults}
    <div class="search-dropdown absolute top-full left-0 right-0 mt-2 {themeClasses} rounded-lg shadow-xl border z-50 max-h-96 overflow-y-auto">

      <!-- Recent Searches (shown when no input) -->
      {#if !searchInput && recentSearches.length > 0}
        <div class="p-4">
          <h3 class="font-medium mb-2 text-sm uppercase tracking-wide opacity-70">Recent Searches</h3>
          <div class="space-y-1">
            {#each recentSearches.slice(0, 5) as recent}
              <button
                class="w-full text-left p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded flex items-center gap-2"
                onclick={() => { searchInput = recent; performSearch(); }}
              >
                <svg class="w-4 h-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                </svg>
                {recent}
              </button>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Trending Searches -->
      {#if !searchInput && trendingSearches.length > 0}
        <div class="p-4 border-t border-current/20">
          <h3 class="font-medium mb-2 text-sm uppercase tracking-wide opacity-70">Trending</h3>
          <div class="flex flex-wrap gap-2">
            {#each trendingSearches as trending}
              <button
                class="px-3 py-1 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-full text-sm"
                onclick={() => selectTrendingSearch(trending)}
              >
                {trending}
              </button>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Search Results -->
      {#if results.length > 0}
        <div class="p-2">
          <div class="text-xs uppercase tracking-wide opacity-70 p-2">
            {results.length} Result{results.length !== 1 ? 's' : ''}
          </div>
          {#each results as result}
            <button
              class="w-full text-left p-3 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg border-b border-current/10 last:border-b-0 transition-colors"
              data-result-type={result.type}
              onclick={() => selectResult(result)}
            >
              <div class="flex items-start gap-3">
                <!-- Result Type Icon -->
                <div class="flex-shrink-0 w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white text-sm font-bold mt-1" data-result-type={result.type}>
                  {#if result.type === 'case'}📁
                  {:else if result.type === 'criminal'}👤
                  {:else if result.type === 'evidence'}🔍
                  {:else if result.type === 'precedent'}⚖️
                  {:else if result.type === 'statute'}📖
                  {:else}📄
                  {/if}
                </div>

                <!-- Result Content -->
                <div class="flex-1 min-w-0">
                  <h4 class="font-medium truncate">{result.title}</h4>
                  <p class="text-sm opacity-70 line-clamp-2 mt-1">{result.content}</p>

                  <!-- Result Metadata -->
                  <div class="flex items-center gap-2 mt-2 text-xs opacity-50">
                    <span class="bg-current/20 px-2 py-1 rounded">{result.type}</span>
                    {#if result.metadata.date}
                      <span>{result.metadata.date}</span>
                    {/if}
                    {#if result.metadata.jurisdiction}
                      <span>{result.metadata.jurisdiction}</span>
                    {/if}
                    <span class="ml-auto">{Math.round(result.score * 100)}% match</span>
                  </div>
                </div>
              </div>
            </button>
          {/each}
        </div>
      {/if}

      <!-- AI Suggestions -->
      {#if suggestions.length > 0}
        <div class="border-t border-current/20 p-4">
          <h3 class="font-medium mb-2 text-sm uppercase tracking-wide opacity-70">Suggested Searches</h3>
          <div class="space-y-1">
            {#each suggestions as suggestion}
              <button
                class="w-full text-left p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded flex items-center gap-2"
                onclick={() => selectSuggestion(suggestion)}
              >
                <svg class="w-4 h-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path>
                </svg>
                {suggestion.text}
                <span class="ml-auto text-xs opacity-50">{suggestion.category}</span>
              </button>
            {/each}
          </div>
        </div>
      {/if}

      <!-- No Results -->
      {#if searchInput && !isSearching && results.length === 0}
        <div class="p-6 text-center">
          <svg class="w-12 h-12 mx-auto opacity-30 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.172 16.172a4 4 0 015.656 0M9 12h6m-6-4h6m2 5.291A7.962 7.962 0 0112 15c-2.34 0-4.47-.881-6.08-2.33"></path>
          </svg>
          <p class="opacity-70">No results found for "{searchInput}"</p>
          <p class="text-sm opacity-50 mt-1">Try adjusting your search or filters</p>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  /* NES.css Legal AI Search Bar Integration */
  .nes-search-bar {
    font-family: 'Courier New', monospace;
  }

  /* NES-style Search Container */
  :global(.nes-search-bar .search-bar-wrapper) {
    border: 3px solid #000;
    box-shadow: 6px 6px 0px rgba(0, 0, 0, 0.3);
    background: linear-gradient(145deg, #f0f0f0, #e6e6e6);
    transition: all 0.2s ease;
  }

  :global(.nes-search-bar .search-bar-wrapper:focus-within) {
    box-shadow: 8px 8px 0px rgba(0, 100, 200, 0.4), 6px 6px 0px rgba(0, 0, 0, 0.3);
    transform: translateY(-2px);
  }

  /* NES-style Input */
  :global(.nes-search-bar input) {
    font-family: 'Courier New', monospace;
    font-weight: bold;
    font-size: 16px;
    background: transparent;
  }

  :global(.nes-search-bar input::placeholder) {
    color: #666;
    font-style: italic;
  }

  /* NES-style Category Chips */
  :global(.nes-search-bar .category-chip) {
    border: 2px solid #000;
    box-shadow: 3px 3px 0px rgba(0, 0, 0, 0.2);
    font-family: 'Courier New', monospace;
    font-weight: bold;
    font-size: 12px;
    text-transform: uppercase;
    transition: all 0.15s ease;
  }

  :global(.nes-search-bar .category-chip:hover) {
    transform: translateY(-2px);
    box-shadow: 4px 4px 0px rgba(0, 0, 0, 0.3);
  }

  :global(.nes-search-bar .category-chip:active) {
    transform: translateY(1px);
    box-shadow: 2px 2px 0px rgba(0, 0, 0, 0.2);
  }

  /* Legal Priority Color Coding for Categories */
  :global(.nes-search-bar .category-chip.selected) {
    background: linear-gradient(135deg, #4CAF50, #45a049) !important;
    color: white !important;
    border-color: #2E7D32;
    box-shadow: 3px 3px 0px rgba(46, 125, 50, 0.4), inset 0 0 10px rgba(255, 255, 255, 0.2);
  }

  /* NES-style Search Results Dropdown */
  :global(.nes-search-bar .search-dropdown) {
    border: 3px solid #000;
    box-shadow: 8px 8px 0px rgba(0, 0, 0, 0.3);
    background: linear-gradient(145deg, #fafafa, #f0f0f0);
    backdrop-filter: blur(8px);
  }

  /* NES-style Result Items */
  :global(.nes-search-bar .search-dropdown button) {
    font-family: 'Courier New', monospace;
    transition: all 0.2s ease;
  }

  :global(.nes-search-bar .search-dropdown button:hover) {
    background: linear-gradient(135deg, #e3f2fd, #bbdefb) !important;
    transform: translateX(4px);
    box-shadow: inset 3px 0 0 rgba(33, 150, 243, 0.5);
  }

  /* NES-style Search Result Type Icons */
  :global(.nes-search-bar .search-dropdown .w-8.h-8) {
    border: 2px solid #000;
    box-shadow: 2px 2px 0px rgba(0, 0, 0, 0.2);
    background: linear-gradient(135deg, #FF6B6B, #FF5252) !important;
  }

  /* Legal-specific Result Categories */
  :global(.nes-search-bar [data-result-type="case"]) {
    background: linear-gradient(135deg, #2196F3, #1976D2) !important;
  }

  :global(.nes-search-bar [data-result-type="criminal"]) {
    background: linear-gradient(135deg, #F44336, #D32F2F) !important;
  }

  :global(.nes-search-bar [data-result-type="evidence"]) {
    background: linear-gradient(135deg, #4CAF50, #388E3C) !important;
  }

  :global(.nes-search-bar [data-result-type="precedent"]) {
    background: linear-gradient(135deg, #FF9800, #F57C00) !important;
  }

  /* NES-style Loading Animation */
  :global(.nes-search-bar .animate-spin) {
    border-color: #000;
    border-top-color: transparent;
    box-shadow: 2px 2px 0px rgba(0, 0, 0, 0.2);
  }

  /* NES-style Clear/Action Buttons */
  :global(.nes-search-bar .p-2) {
    border-radius: 0;
    transition: all 0.15s ease;
  }

  :global(.nes-search-bar .p-2:hover) {
    background: linear-gradient(135deg, #ffecb3, #fff3c4) !important;
    transform: scale(1.1);
  }

  /* NES-style Trending/Recent Search Buttons */
  :global(.nes-search-bar .px-3.py-1) {
    border: 2px solid #000;
    box-shadow: 2px 2px 0px rgba(0, 0, 0, 0.2);
    font-family: 'Courier New', monospace;
    font-weight: bold;
    background: linear-gradient(135deg, #e8f5e8, #f1f8e9);
  }

  :global(.nes-search-bar .px-3.py-1:hover) {
    background: linear-gradient(135deg, #c8e6c9, #dcedc8);
    transform: translateY(-1px);
    box-shadow: 3px 3px 0px rgba(0, 0, 0, 0.3);
  }

  /* Custom Legal AI Color Scheme */
  :global(.nes-search-bar .text-blue-800) { color: #1565C0 !important; }
  :global(.nes-search-bar .text-blue-200) { color: #BBDEFB !important; }
  :global(.nes-search-bar .bg-blue-100) { background: linear-gradient(135deg, #E3F2FD, #BBDEFB) !important; }
  :global(.nes-search-bar .bg-blue-900) { background: linear-gradient(135deg, #0D47A1, #1565C0) !important; }

  /* Enhanced Visual Feedback */
  :global(.nes-search-bar .border-current) {
    border-width: 2px;
    border-style: solid;
  }

  /* GPU Performance Optimizations */
  :global(.nes-search-bar *) {
    transform-origin: center;
  }

  /* Original styles preserved */
  .line-clamp-2 {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  .category-chip {
    transition: all 0.2s ease-in-out;
  }

  .category-chip:hover {
    transform: translateY(-1px);
  }

  .search-dropdown {
    backdrop-filter: blur(8px);
  }
</style>