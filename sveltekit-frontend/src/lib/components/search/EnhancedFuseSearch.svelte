<script lang="ts">
  import Fuse from 'fuse.js';
  import { Card, CardContent, CardHeader, CardTitle } from "$lib/components/ui/card/index.js";
  import { Input } from "$lib/components/ui/input/index.js";
  import { Button } from "$lib/components/ui/button/index.js";
  import { Badge } from "$lib/components/ui/badge/index.js";
  import { Search, ExternalLink, Sparkles, FileText, Scale } from "lucide-svelte";
  import { legalDocuments, type LegalDocument } from "$lib/data/legal-documents";

  // Props
  let { 
    placeholder = "Search laws, cases, and legal documents...",
    maxResults = 10,
    showCategories = true,
    compact = false 
  } = $props();

  // State
  let searchQuery = $state('');
  let searchResults = $state<Fuse.FuseResult<LegalDocument>[]>([]);
  let isSearching = $state(false);

  // Fuse.js configuration for legal document search
  const fuseOptions = {
    keys: [
      { name: 'title', weight: 0.4 },
      { name: 'description', weight: 0.3 },
      { name: 'content', weight: 0.2 },
      { name: 'code', weight: 0.1 }
    ],
    threshold: 0.4, // More permissive for legal terms
    distance: 100,
    includeScore: true,
    includeMatches: true,
    minMatchCharLength: 2,
    shouldSort: true,
    findAllMatches: false
  };

  const fuse = new Fuse(legalDocuments, fuseOptions);

  // Perform search
  function performSearch() {
    if (!searchQuery.trim()) {
      searchResults = [];
      return;
    }

    isSearching = true;
    try {
      const results = fuse.search(searchQuery, { limit: maxResults });
      searchResults = results;
    } catch (error) {
      console.error('Fuse search error:', error);
      searchResults = [];
    } finally {
      isSearching = false;
    }
  }

  // Real-time search as user types (debounced)
  let debounceTimer: NodeJS.Timeout | null = null;
  $effect(() => {
    if (searchQuery.trim()) {
      if (debounceTimer) clearTimeout(debounceTimer);
      debounceTimer = setTimeout(performSearch, 300);
    } else {
      searchResults = [];
    }
    return () => {
      if (debounceTimer) clearTimeout(debounceTimer);
    };
  });

  function handleKeydown(event: KeyboardEvent) {
    if (event.key === 'Enter') {
      event.preventDefault();
      performSearch();
    }
  }

  function highlightMatches(text: string, matches?: readonly Fuse.FuseResultMatch[]): string {
    if (!matches) return text;
    
    let highlightedText = text;
    const sortedMatches = [...matches].sort((a, b) => b.indices[0][0] - a.indices[0][0]);
    
    for (const match of sortedMatches) {
      for (const [start, end] of match.indices.reverse()) {
        highlightedText = 
          highlightedText.slice(0, start) + 
          '<mark class="bg-yellow-200 dark:bg-yellow-800 px-1 rounded">' + 
          highlightedText.slice(start, end + 1) + 
          '</mark>' + 
          highlightedText.slice(end + 1);
      }
    }
    
    return highlightedText;
  }

  function getCategoryColor(category: string): string {
    const colors = {
      criminal: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
      civil: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
      contract: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
      evidence: 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200',
      corporate: 'bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-200',
      constitutional: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
      family: 'bg-pink-100 text-pink-800 dark:bg-pink-900 dark:text-pink-200',
      administrative: 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
    };
    return colors[category] || colors.administrative;
  }

  function getJurisdictionColor(jurisdiction: string): string {
    const colors = {
      california: 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200',
      federal: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
      state: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
    };
    return colors[jurisdiction] || colors.state;
  }

  function getConfidenceLabel(score?: number): string {
    if (!score) return 'Perfect';
    if (score < 0.1) return 'Perfect';
    if (score < 0.3) return 'Excellent';
    if (score < 0.5) return 'Good';
    return 'Fair';
  }
</script>

<div class="space-y-4">
  <!-- Search Input -->
  <Card class="border-primary/20">
    <CardHeader class="pb-3">
      <CardTitle class="flex items-center gap-2 text-lg">
        <Search class="h-5 w-5 text-primary" />
        Enhanced Legal Search
        <Sparkles class="h-4 w-4 text-yellow-500" />
      </CardTitle>
    </CardHeader>
    <CardContent>
      <div class="flex gap-2">
        <div class="relative flex-1">
          <Search class="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            {placeholder}
            bind:value={searchQuery}
            onkeydown={handleKeydown}
            class="pl-10"
          />
        </div>
        <Button 
          onclick={performSearch} 
          disabled={isSearching || !searchQuery.trim()}
          size="sm"
        >
          {#if isSearching}
            Searching...
          {:else}
            <Search class="h-4 w-4" />
          {/if}
        </Button>
      </div>
      
      {#if searchQuery && searchResults.length > 0}
        <div class="mt-2 text-sm text-muted-foreground">
          Found {searchResults.length} result{searchResults.length === 1 ? '' : 's'} for "{searchQuery}"
        </div>
      {/if}
    </CardContent>
  </Card>

  <!-- Search Results -->
  {#if searchResults.length > 0}
    <div class="space-y-3">
      {#each searchResults as result, index (result.item.id)}
        <Card class="hover:shadow-md transition-all duration-200 border-l-4 border-l-primary/30">
          <CardHeader class="pb-2">
            <div class="flex items-start justify-between">
              <div class="flex-1 space-y-2">
                <CardTitle class="text-base leading-tight">
                  {@html highlightMatches(result.item.title, result.matches?.filter(m => m.key === 'title'))}
                </CardTitle>
                
                <div class="flex flex-wrap gap-2">
                  <Badge class={getJurisdictionColor(result.item.jurisdiction)}>
                    {result.item.jurisdiction}
                  </Badge>
                  <Badge class={getCategoryColor(result.item.category)}>
                    {result.item.category}
                  </Badge>
                  <Badge variant="outline">
                    {result.item.code}
                  </Badge>
                  <Badge variant="secondary" class="text-xs">
                    <Scale class="h-3 w-3 mr-1" />
                    {getConfidenceLabel(result.score)}
                  </Badge>
                </div>
              </div>
              
              <div class="text-right text-xs text-muted-foreground">
                #{index + 1}
              </div>
            </div>
          </CardHeader>
          
          <CardContent class="pt-0">
            <p class="text-sm text-muted-foreground mb-3">
              {@html highlightMatches(result.item.description, result.matches?.filter(m => m.key === 'description'))}
            </p>
            
            {#if result.matches?.some(m => m.key === 'content')}
              <div class="text-xs bg-muted/50 p-2 rounded mb-3">
                <div class="font-medium mb-1">Content Match:</div>
                <div class="text-muted-foreground">
                  {@html highlightMatches(
                    result.item.content.substring(0, 200) + '...',
                    result.matches?.filter(m => m.key === 'content')
                  )}
                </div>
              </div>
            {/if}

            {#if result.item.sections && result.item.sections.length > 0}
              <div class="flex flex-wrap gap-1 mb-3">
                {#each result.item.sections.slice(0, 3) as section}
                  <Badge variant="outline" class="text-xs">
                    <FileText class="h-2 w-2 mr-1" />
                    {section}
                  </Badge>
                {/each}
                {#if result.item.sections.length > 3}
                  <Badge variant="outline" class="text-xs">
                    +{result.item.sections.length - 3} more
                  </Badge>
                {/if}
              </div>
            {/if}

            <div class="flex gap-2">
              <Button size="sm" variant="outline">
                <FileText class="h-3 w-3 mr-1" />
                AI Summary
              </Button>
              <Button size="sm" variant="outline">
                <Sparkles class="h-3 w-3 mr-1" />
                AI Analysis
              </Button>
              {#if result.item.url}
                <Button size="sm" variant="outline">
                  <a href={result.item.url} target="_blank" rel="noopener noreferrer" class="flex items-center gap-1">
                    <ExternalLink class="h-3 w-3" />
                    Full Text
                  </a>
                </Button>
              {/if}
            </div>
          </CardContent>
        </Card>
      {/each}
    </div>
  {:else if searchQuery && !isSearching}
    <Card>
      <CardContent class="py-8 text-center">
        <div class="space-y-3">
          <Search class="h-12 w-12 mx-auto text-muted-foreground" />
          <div>
            <p class="font-medium">No results found</p>
            <p class="text-sm text-muted-foreground">
              Try searching for terms like "murder", "contract", "evidence", or "robbery"
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  {/if}

  <!-- Quick Search Suggestions -->
  {#if !searchQuery}
    <Card>
      <CardHeader>
        <CardTitle class="text-sm">Quick Search Examples</CardTitle>
      </CardHeader>
      <CardContent>
        <div class="flex flex-wrap gap-2">
          {#each ['murder', 'contract liability', 'evidence rules', 'robbery', 'constitutional rights', 'family law'] as suggestion}
            <Button 
              variant="outline" 
              size="sm" 
              onclick={() => { searchQuery = suggestion; performSearch(); }}
            >
              {suggestion}
            </Button>
          {/each}
        </div>
      </CardContent>
    </Card>
  {/if}
</div>

<style>
  :global(mark) {
    background-color: rgb(254 240 138 / 0.5);
    padding: 0.125rem 0.25rem;
    border-radius: 0.25rem;
    font-weight: 500;
  }
  
  :global(.dark mark) {
    background-color: rgb(133 77 14 / 0.8);
    color: rgb(254 240 138);
  }
</style>