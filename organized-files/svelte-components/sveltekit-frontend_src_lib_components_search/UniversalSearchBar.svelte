<!--
  Universal Search Bar - Svelte 5 Component
  Modern search interface with AI-powered suggestions and vector search
-->
<script lang="ts">
  import { Search, Filter, X, Zap, Brain, Database } from 'lucide-svelte';
  import { createEventDispatcher } from 'svelte';
  import { writable } from 'svelte/store';
  
  // Props
  interface Props {
    placeholder?: string;
    searchType?: 'semantic' | 'vector' | 'legal' | 'universal';
    enableFilters?: boolean;
    enableAI?: boolean;
    initialQuery?: string;
    disabled?: boolean;
    class?: string;
  }
  
  let {
    placeholder = 'Search documents, cases, contracts...',
    searchType = 'universal',
    enableFilters = true,
    enableAI = true,
    initialQuery = '',
    disabled = false,
    class: className = '',
    ...restProps
  } = $props();
  
  // State using Svelte 5 runes
  let query = $state(initialQuery);
  let isSearching = $state(false);
  let showFilters = $state(false);
  let showSuggestions = $state(false);
  let searchInput: HTMLInputElement
  
  // Search suggestions
  let suggestions = $state<string[]>([]);
  let selectedSuggestionIndex = $state(-1);
  
  // Filters state
  let filters = $state({
    documentType: 'all',
    dateRange: 'all',
    jurisdiction: 'all',
    tags: [] as string[]
  });
  
  // Event dispatcher for Svelte 4/5 compatibility
  const dispatch = createEventDispatcher<{
    search: { query: string filters: typeof filters; type: string };
    clear: void
    filter: typeof filters;
    suggestion: { suggestion: string };
  }>();
  
  // Search types configuration
  const searchTypes = [
    { id: 'universal', name: 'Universal', icon: Search, description: 'Search all content' },
    { id: 'semantic', name: 'Semantic', icon: Brain, description: 'AI-powered contextual search' },
    { id: 'vector', name: 'Vector', icon: Zap, description: 'Similarity-based search' },
    { id: 'legal', name: 'Legal', icon: Database, description: 'Legal documents only' }
  ];
  
  // Mock suggestions for demo
  const mockSuggestions = [
    'contract liability clauses',
    'employment termination cases',
    'intellectual property disputes',
    'merger and acquisition agreements',
    'data privacy regulations',
    'corporate governance policies'
  ];
  
  // Reactive updates for suggestions
  $effect(() => {
    if (query.length >= 2) {
      // Simulate AI-powered suggestions
      const filtered = mockSuggestions.filter(s => 
        s.toLowerCase().includes(query.toLowerCase())
      );
      suggestions = filtered.slice(0, 5);
      showSuggestions = suggestions.length > 0 && !isSearching;
    } else {
      suggestions = [];
      showSuggestions = false;
    }
  });
  
  // Search handler
  async function handleSearch() {
    if (!query.trim() || isSearching || disabled) return;
    
    isSearching = true;
    showSuggestions = false;
    
    try {
      // Emit search event
      dispatch('search', {
        query: query.trim(),
        filters,
        type: searchType
      });
      
      // Simulate search delay
      await new Promise(resolve => setTimeout(resolve, 300));
      
    } catch (error) {
      console.error('Search error:', error);
    } finally {
      isSearching = false;
    }
  }
  
  // Clear search
  function handleClear() {
    query = '';
    suggestions = [];
    showSuggestions = false;
    selectedSuggestionIndex = -1;
    searchInput?.focus();
    dispatch('clear');
  }
  
  // Handle keyboard navigation
  function handleKeydown(event: KeyboardEvent) {
    if (!showSuggestions) return;
    
    switch (event.key) {
      case 'ArrowDown':
        event.preventDefault();
        selectedSuggestionIndex = Math.min(selectedSuggestionIndex + 1, suggestions.length - 1);
        break;
      case 'ArrowUp':
        event.preventDefault();
        selectedSuggestionIndex = Math.max(selectedSuggestionIndex - 1, -1);
        break;
      case 'Enter':
        if (selectedSuggestionIndex >= 0) {
          event.preventDefault();
          selectSuggestion(suggestions[selectedSuggestionIndex]);
        } else {
          handleSearch();
        }
        break;
      case 'Escape':
        showSuggestions = false;
        selectedSuggestionIndex = -1;
        break;
    }
  }
  
  // Select suggestion
  function selectSuggestion(suggestion: string) {
    query = suggestion;
    showSuggestions = false;
    selectedSuggestionIndex = -1;
    dispatch('suggestion', { suggestion });
    handleSearch();
  }
  
  // Toggle filters
  function toggleFilters() {
    showFilters = !showFilters;
  }
  
  // Update filters
  function updateFilters() {
    dispatch('filter', filters);
  }
  
  // Close suggestions when clicking outside
  function handleBlur() {
    setTimeout(() => {
      showSuggestions = false;
      selectedSuggestionIndex = -1;
    }, 150);
  }
</script>

<!-- Search container -->
<div class="universal-search-container {className}" {...restProps}>
  <!-- Search type selector -->
  <div class="search-type-tabs">
    {#each searchTypes as type}
      <button
        class="type-tab"
        class:active={searchType === type.id}
        onclick={() => searchType = type.id as any}
        title={type.description}
        {disabled}
      >
        <svelte:component this={type.icon} size="16" />
        <span>{type.name}</span>
      </button>
    {/each}
  </div>

  <!-- Main search bar -->
  <div class="search-bar-wrapper">
    <div class="search-bar" class:searching={isSearching} class:has-suggestions={showSuggestions}>
      <!-- Search icon -->
      <div class="search-icon">
        {#if isSearching}
          <div class="loading-spinner"></div>
        {:else}
          <Search size="20" />
        {/if}
      </div>
      
      <!-- Search input -->
      <input
        bind:this={searchInput}
        bind:value={query}
        type="text"
        {placeholder}
        class="search-input"
        autocomplete="off"
        spellcheck="false"
        {disabled}
        onkeydown={handleKeydown}
        onblur={handleBlur}
        onfocus={() => {
          if (query.length >= 2) showSuggestions = suggestions.length > 0;
        }}
      />
      
      <!-- Clear button -->
      {#if query}
        <button
          class="clear-btn"
          onclick={handleClear}
          title="Clear search"
          {disabled}
        >
          <X size="16" />
        </button>
      {/if}
      
      <!-- Search button -->
      <button
        class="search-btn"
        onclick={handleSearch}
        disabled={!query.trim() || isSearching || disabled}
        title="Search"
      >
        <Search size="18" />
      </button>
      
      <!-- Filters toggle -->
      {#if enableFilters}
        <button
          class="filters-btn"
          class:active={showFilters}
          onclick={toggleFilters}
          title="Filters"
          {disabled}
        >
          <Filter size="16" />
        </button>
      {/if}
    </div>
    
    <!-- Search suggestions -->
    {#if showSuggestions}
      <div class="suggestions-dropdown">
        {#each suggestions as suggestion, index}
          <button
            class="suggestion-item"
            class:selected={index === selectedSuggestionIndex}
            onclick={() => selectSuggestion(suggestion)}
          >
            <Search size="14" class="suggestion-icon" />
            <span class="suggestion-text">{suggestion}</span>
            {#if enableAI}
              <Brain size="12" class="ai-badge" title="AI suggestion" />
            {/if}
          </button>
        {/each}
      </div>
    {/if}
  </div>
  
  <!-- Filters panel -->
  {#if showFilters && enableFilters}
    <div class="filters-panel">
      <div class="filters-grid">
        <!-- Document type filter -->
        <div class="filter-group">
          <label for="document-type">Document Type</label>
          <select
            id="document-type"
            bind:value={filters.documentType}
            onchange={updateFilters}
            {disabled}
          >
            <option value="all">All Documents</option>
            <option value="contract">Contracts</option>
            <option value="case_law">Case Law</option>
            <option value="evidence">Evidence</option>
            <option value="statute">Statutes</option>
            <option value="memo">Memos</option>
          </select>
        </div>
        
        <!-- Date range filter -->
        <div class="filter-group">
          <label for="date-range">Date Range</label>
          <select
            id="date-range"
            bind:value={filters.dateRange}
            onchange={updateFilters}
            {disabled}
          >
            <option value="all">Any Time</option>
            <option value="recent">Last 30 days</option>
            <option value="quarter">Last 3 months</option>
            <option value="year">Last year</option>
            <option value="older">Older</option>
          </select>
        </div>
        
        <!-- Jurisdiction filter -->
        <div class="filter-group">
          <label for="jurisdiction">Jurisdiction</label>
          <select
            id="jurisdiction"
            bind:value={filters.jurisdiction}
            onchange={updateFilters}
            {disabled}
          >
            <option value="all">All Jurisdictions</option>
            <option value="federal">Federal</option>
            <option value="state">State</option>
            <option value="local">Local</option>
            <option value="international">International</option>
          </select>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .universal-search-container {
    width: 100%;
    background: rgba(0, 0, 0, 0.9);
    border: 2px solid #00ff88;
    border-radius: 12px;
    padding: 1rem;
    position: relative
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    color: #00ff88;
  }
  
  .search-type-tabs {
    display: flex
    gap: 0.5rem;
    margin-bottom: 1rem;
    flex-wrap: wrap
  }
  
  .type-tab {
    display: flex
    align-items: center
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background: rgba(0, 0, 0, 0.5);
    border: 1px solid #333;
    border-radius: 6px;
    color: #00ff88;
    font-family: inherit
    font-size: 0.8rem;
    cursor: pointer
    transition: all 0.3s ease;
  }
  
  .type-tab:hover {
    border-color: #00ff88;
    background: rgba(0, 255, 136, 0.1);
  }
  
  .type-tab.active {
    border-color: #00ff88;
    background: rgba(0, 255, 136, 0.2);
    box-shadow: 0 0 10px rgba(0, 255, 136, 0.3);
  }
  
  .search-bar-wrapper {
    position: relative
  }
  
  .search-bar {
    display: flex
    align-items: center
    background: rgba(0, 0, 0, 0.7);
    border: 1px solid #333;
    border-radius: 8px;
    padding: 0.75rem;
    gap: 0.75rem;
    transition: all 0.3s ease;
  }
  
  .search-bar:focus-within {
    border-color: #00ff88;
    box-shadow: 0 0 15px rgba(0, 255, 136, 0.3);
  }
  
  .search-bar.searching {
    border-color: #00aaff;
    box-shadow: 0 0 15px rgba(0, 170, 255, 0.3);
  }
  
  .search-bar.has-suggestions {
    border-bottom-left-radius: 0;
    border-bottom-right-radius: 0;
  }
  
  .search-icon {
    display: flex
    align-items: center
    color: #00ff88;
    min-width: 20px;
  }
  
  .loading-spinner {
    width: 20px;
    height: 20px;
    border: 2px solid #333;
    border-top: 2px solid #00ff88;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
  
  .search-input {
    flex: 1;
    background: transparent
    border: none
    outline: none
    color: #00ff88;
    font-family: inherit
    font-size: 1rem;
    padding: 0;
  }
  
  .search-input::placeholder {
    color: #666;
  }
  
  .search-input:disabled {
    color: #555;
    cursor: not-allowed;
  }
  
  .clear-btn,
  .search-btn,
  .filters-btn {
    display: flex
    align-items: center
    justify-content: center
    background: transparent
    border: 1px solid #333;
    border-radius: 4px;
    color: #00ff88;
    cursor: pointer
    padding: 0.5rem;
    transition: all 0.3s ease;
    min-width: 32px;
    height: 32px;
  }
  
  .clear-btn:hover {
    border-color: #ff4444;
    color: #ff4444;
  }
  
  .search-btn:hover:not(:disabled) {
    border-color: #00ff88;
    background: rgba(0, 255, 136, 0.1);
  }
  
  .search-btn:disabled {
    color: #555;
    border-color: #222;
    cursor: not-allowed;
  }
  
  .filters-btn:hover {
    border-color: #00aaff;
    color: #00aaff;
  }
  
  .filters-btn.active {
    border-color: #00aaff;
    background: rgba(0, 170, 255, 0.2);
    color: #00aaff;
  }
  
  .suggestions-dropdown {
    position: absolute
    top: 100%;
    left: 0;
    right: 0;
    background: rgba(0, 0, 0, 0.95);
    border: 1px solid #00ff88;
    border-top: none
    border-bottom-left-radius: 8px;
    border-bottom-right-radius: 8px;
    max-height: 200px;
    overflow-y: auto
    z-index: 1000;
  }
  
  .suggestion-item {
    display: flex
    align-items: center
    width: 100%;
    padding: 0.75rem;
    gap: 0.75rem;
    background: transparent
    border: none
    border-bottom: 1px solid #222;
    color: #00ff88;
    font-family: inherit
    cursor: pointer
    text-align: left
    transition: all 0.3s ease;
  }
  
  .suggestion-item:hover,
  .suggestion-item.selected {
    background: rgba(0, 255, 136, 0.1);
  }
  
  .suggestion-item:last-child {
    border-bottom: none
  }
  
  .suggestion-icon {
    color: #666;
  }
  
  .suggestion-text {
    flex: 1;
  }
  
  .ai-badge {
    color: #00aaff;
    opacity: 0.7;
  }
  
  .filters-panel {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid #333;
  }
  
  .filters-grid {
    display: grid
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
  }
  
  .filter-group {
    display: flex
    flex-direction: column
    gap: 0.5rem;
  }
  
  .filter-group label {
    font-size: 0.8rem;
    color: #00ff88;
    font-weight: bold
  }
  
  .filter-group select {
    background: rgba(0, 0, 0, 0.7);
    border: 1px solid #333;
    border-radius: 4px;
    padding: 0.5rem;
    color: #00ff88;
    font-family: inherit
    font-size: 0.9rem;
  }
  
  .filter-group select:focus {
    outline: none
    border-color: #00ff88;
    box-shadow: 0 0 5px rgba(0, 255, 136, 0.3);
  }
  
  .filter-group select:disabled {
    color: #555;
    cursor: not-allowed;
  }
  
  /* Responsive design */
  @media (max-width: 768px) {
    .search-type-tabs {
      justify-content: center
    }
    
    .type-tab span {
      display: none
    }
    
    .filters-grid {
      grid-template-columns: 1fr;
    }
  }
  
  /* Dark theme integration */
  @media (prefers-color-scheme: dark) {
    .universal-search-container {
      background: rgba(0, 0, 0, 0.95);
    }
    
    .search-bar {
      background: rgba(0, 0, 0, 0.8);
    }
    
    .suggestions-dropdown {
      background: rgba(0, 0, 0, 0.98);
    }
  }
</style>