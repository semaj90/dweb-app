<!--
Vector Search Widget
Compact searchable component for embedding in other interfaces
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { Button } from '$lib/components/ui/button';
  import { Badge } from "$lib/components/ui/badge";
  import {
    Search,
    Loader2,
    FileText,
    Users,
    MapPin,
    Calendar,
    Scale,
    Eye,
    X
  } from 'lucide-svelte';

  import { vectorIntelligenceService } from '$lib/services/vector-intelligence-service.js';
  import type { VectorSearchResult } from '$lib/services/vector-intelligence-service.js';

  interface Props {
    placeholder?: string;
    maxResults?: number;
    threshold?: number;
    contextFilter?: {
      caseId?: string;
      evidenceType?: string;
    };
    onResultSelect?: (result: VectorSearchResult) => void;
    compact?: boolean;
  }

  let {
    placeholder = 'Search documents, cases, evidence...',
    maxResults = 5,
    threshold = 0.7,
    contextFilter = {},
    onResultSelect = () => {},
    compact = false
  } = $props();

  let searchQuery = $state('');
  let searchResults = $state<VectorSearchResult[]>([]);
  let isSearching = $state(false);
  let isOpen = $state(false);
  let searchTimeout = $state<number | null>(null);
  let inputElement = $state<HTMLInputElement | null>(null);

  // Debounced search
  $effect(() => {
    if (searchQuery.length >= 2) {
      if (searchTimeout) clearTimeout(searchTimeout);
      searchTimeout = setTimeout(performSearch, 300);
    } else {
      searchResults = [];
      isOpen = false;
    }
  });

  async function performSearch() {
    if (!searchQuery.trim() || isSearching) return;

    isSearching = true;
    try {
      const results = await vectorIntelligenceService.semanticSearch({
        query: searchQuery,
        threshold,
        limit: maxResults,
        includeMetadata: true,
        contextFilter
      });

      searchResults = results;
      isOpen = results.length > 0;
    } catch (error) {
      console.error('Vector search failed:', error);
      searchResults = [];
    } finally {
      isSearching = false;
    }
  }

  function selectResult(result: VectorSearchResult) {
    onResultSelect(result);
    searchQuery = '';
    searchResults = [];
    isOpen = false;
    inputElement?.blur();
  }

  function clearSearch() {
    searchQuery = '';
    searchResults = [];
    isOpen = false;
    inputElement?.focus();
  }

  function getEntityIcon(type: string) {
    switch (type) {
      case 'person': return Users;
      case 'organization': return Users;
      case 'location': return MapPin;
      case 'date': return Calendar;
      case 'legal_concept': return Scale;
      default: return FileText;
    }
  }

  function getConfidenceColor(confidence: number) {
    if (confidence >= 0.8) return 'vector-confidence-high';
    if (confidence >= 0.6) return 'vector-confidence-medium';
    return 'vector-confidence-low';
  }

  onMount(() => {
    // Close dropdown when clicking outside
    function handleClickOutside(event: MouseEvent) {
      const target = event.target as Element;
      if (!target.closest('.vector-search-widget')) {
        isOpen = false;
      }
    }

    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  });
</script>

<div class="vector-search-widget relative w-full max-w-md">
  <!-- Search Input -->
  <div class="relative">
    <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
      {#if isSearching}
        <Loader2 class="h-4 w-4 animate-spin text-muted-foreground" />
      {:else}
        <Search class="h-4 w-4 text-muted-foreground" />
      {/if}
    </div>

    <input
      bind:this={inputElement}
      bind:value={searchQuery}
      type="text"
      {placeholder}
      class="vector-search-input pl-10 {searchQuery ? 'pr-10' : 'pr-3'} {compact ? 'h-8 text-sm' : 'h-10'}"
      onfocus={() => { if (searchResults.length > 0) isOpen = true; }}
    />

    {#if searchQuery}
      <button
        type="button"
        class="absolute inset-y-0 right-0 pr-3 flex items-center"
        on:click={clearSearch}
      >
        <X class="h-4 w-4 text-muted-foreground hover:text-foreground" />
      </button>
    {/if}
  </div>

  <!-- Search Results Dropdown -->
  {#if isOpen && searchResults.length > 0}
    <div class="absolute top-full left-0 right-0 z-50 mt-1 bg-popover border border-border rounded-md shadow-lg max-h-96 overflow-y-auto">
      <div class="p-2">
        <div class="text-xs text-muted-foreground mb-2 px-2">
          Found {searchResults.length} result{searchResults.length !== 1 ? 's' : ''}
        </div>

        <div class="space-y-1">
          {#each searchResults as result}
            <button
              type="button"
              class="w-full text-left p-3 rounded-md hover:bg-accent hover:text-accent-foreground transition-colors"
              on:click={() => selectResult(result)}
            >
              <div class="flex items-start justify-between mb-2">
                <div class="flex items-center gap-2">
                  <svelte:component this={getEntityIcon(result.source)} class="h-4 w-4 text-muted-foreground" />
                  <span class="text-sm font-medium truncate">{result.id}</span>
                </div>
                <div class="flex items-center gap-1">
                  <Badge class={`text-xs ${getConfidenceColor(result.similarity)}`}>
                    {Math.round(result.similarity * 100)}%
                  </Badge>
                  <Badge variant="outline" class="text-xs">{result.source}</Badge>
                </div>
              </div>

              <p class="text-xs text-muted-foreground line-clamp-2 mb-2">
                {result.content.substring(0, 120)}...
              </p>

              {#if result.highlights?.length > 0}
                <div class="text-xs">
                  <span class="vector-highlight">{result.highlights[0]}</span>
                </div>
              {/if}

              {#if !compact}
                <div class="flex items-center gap-3 mt-2 text-xs text-muted-foreground">
                  <span>Relevance: {result.relevanceScore.toFixed(2)}</span>
                  <span>•</span>
                  <span>Similarity: {result.similarity.toFixed(3)}</span>
                </div>
              {/if}
            </button>
          {/each}
        </div>
      </div>

      {#if searchResults.length === maxResults}
        <div class="border-t border-border p-2">
          <div class="text-xs text-muted-foreground text-center">
            Showing top {maxResults} results
          </div>
        </div>
      {/if}
    </div>
  {/if}

  <!-- No Results Message -->
  {#if isOpen && searchResults.length === 0 && !isSearching && searchQuery.length >= 2}
    <div class="absolute top-full left-0 right-0 z-50 mt-1 bg-popover border border-border rounded-md shadow-lg">
      <div class="p-4 text-center">
        <Search class="h-8 w-8 text-muted-foreground mx-auto mb-2" />
        <p class="text-sm text-muted-foreground">No results found for "{searchQuery}"</p>
        <p class="text-xs text-muted-foreground mt-1">Try adjusting your search terms</p>
      </div>
    </div>
  {/if}
</div>

<style>
  /* @unocss-include */
  .line-clamp-2 {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical
    overflow: hidden
  }
</style>