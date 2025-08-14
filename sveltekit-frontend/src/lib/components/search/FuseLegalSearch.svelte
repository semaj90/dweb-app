<script lang="ts">
  import Fuse from 'fuse.js';
  import { onMount } from 'svelte';
  import { Input } from "$lib/components/ui/input/index.js";
  import { Button } from "$lib/components/ui/button/index.js";
  import * as Card from "$lib/components/ui/card/index.js";
  import { Badge } from "$lib/components/ui/badge/index.js";
  import { Search, Loader2, ExternalLink, Bot } from "lucide-svelte";

  let { 
    data = [], 
    placeholder = "Search laws and regulations...",
    onResultSelect = null,
    showAIActions = true,
    maxResults = 10
  } = $props();

  let searchQuery = $state('');
  let searchResults = $state([]);
  let isSearching = $state(false);
  let fuse = $state(null);

  // Fuse.js configuration optimized for legal search
  const fuseOptions = {
    keys: [
      { name: 'title', weight: 0.4 },
      { name: 'description', weight: 0.3 },
      { name: 'code', weight: 0.2 },
      { name: 'keywords', weight: 0.1 }
    ],
    threshold: 0.3, // Lower = more strict matching
    distance: 100,   // How far to search for pattern
    minMatchCharLength: 2,
    includeScore: true,
    includeMatches: true,
    ignoreLocation: true, // Search anywhere in the text
    useExtendedSearch: true // Enable advanced search patterns
  };

  // Initialize Fuse.js when data changes
  $effect(() => {
    if (data && data.length > 0) {
      fuse = new Fuse(data, fuseOptions);
    }
  });

  // Perform search when query changes
  $effect(() => {
    if (searchQuery.trim() && fuse) {
      performFuseSearch();
    } else {
      searchResults = [];
    }
  });

  function performFuseSearch() {
    if (!fuse || !searchQuery.trim()) {
      searchResults = [];
      return;
    }

    isSearching = true;
    
    try {
      // Use Fuse.js extended search patterns
      let query = searchQuery;
      
      // Add smart query preprocessing
      if (query.includes('murder') || query.includes('homicide')) {
        query = `murder | homicide | killing`;
      } else if (query.includes('contract')) {
        query = `contract | agreement | "civil code"`;
      } else if (query.includes('search') || query.includes('warrant')) {
        query = `search | warrant | "fourth amendment" | seizure`;
      }

      const results = fuse.search(query).slice(0, maxResults);
      
      // Process results with highlighting and scoring
      searchResults = results.map(result => ({
        ...result.item,
        fuseScore: result.score,
        matches: result.matches || [],
        highlighted: highlightMatches(result.item, result.matches || [])
      }));

    } catch (error) {
      console.error('Fuse search error:', error);
      searchResults = [];
    } finally {
      isSearching = false;
    }
  }

  function highlightMatches(item, matches) {
    const highlighted = { ...item };
    
    matches.forEach(match => {
      if (match.key && highlighted[match.key]) {
        let text = highlighted[match.key];
        
        // Sort indices in reverse order to avoid offset issues
        const indices = [...match.indices].sort((a, b) => b[0] - a[0]);
        
        indices.forEach(([start, end]) => {
          const before = text.substring(0, start);
          const matched = text.substring(start, end + 1);
          const after = text.substring(end + 1);
          text = `${before}<mark class="bg-yellow-200 dark:bg-yellow-900 px-1 rounded">${matched}</mark>${after}`;
        });
        
        highlighted[match.key] = text;
      }
    });
    
    return highlighted;
  }

  function getScoreColor(score) {
    if (score < 0.2) return 'text-green-600 dark:text-green-400';
    if (score < 0.4) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  }

  function getScoreLabel(score) {
    if (score < 0.2) return 'Excellent Match';
    if (score < 0.4) return 'Good Match';
    return 'Fair Match';
  }

  async function handleAIAction(law, action) {
    if (onResultSelect) {
      onResultSelect(law, action);
    }
  }

  // Handle keyboard navigation
  function handleKeydown(event) {
    if (event.key === 'Enter' && searchResults.length > 0) {
      handleAIAction(searchResults[0], 'select');
    }
  }
</script>

<div class="space-y-4">
  <!-- Search Input -->
  <div class="relative">
    <Search class="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
    <Input
      bind:value={searchQuery}
      {placeholder}
      onkeydown={handleKeydown}
      class="pl-10"
    />
    {#if isSearching}
      <Loader2 class="absolute right-3 top-1/2 transform -translate-y-1/2 h-4 w-4 animate-spin text-muted-foreground" />
    {/if}
  </div>

  <!-- Search Info -->
  {#if searchQuery && searchResults.length > 0}
    <div class="text-sm text-muted-foreground">
      Found {searchResults.length} results for "{searchQuery}"
      {#if searchResults.length === maxResults}
        (showing top {maxResults})
      {/if}
    </div>
  {/if}

  <!-- Search Results -->
  {#if searchResults.length > 0}
    <div class="space-y-3">
      {#each searchResults as law}
        <Card.Root class="hover:shadow-md transition-shadow">
          <Card.Header class="pb-3">
            <div class="flex items-start justify-between">
              <Card.Title class="text-base leading-tight">
                {@html law.highlighted.title || law.title}
              </Card.Title>
              <div class="flex items-center gap-2 ml-2">
                <Badge variant="outline" class="text-xs {getScoreColor(law.fuseScore)}">
                  {getScoreLabel(law.fuseScore)}
                </Badge>
                <Badge variant="secondary" class="text-xs">
                  {law.jurisdiction}
                </Badge>
              </div>
            </div>
            <Card.Description class="text-sm">
              {@html law.highlighted.description || law.description}
            </Card.Description>
            <div class="flex gap-2 text-xs text-muted-foreground">
              <span>{law.code}</span>
              {#if law.category}
                <span>•</span>
                <span class="capitalize">{law.category}</span>
              {/if}
              {#if law.lastUpdated}
                <span>•</span>
                <span>Updated {new Date(law.lastUpdated).toLocaleDateString()}</span>
              {/if}
            </div>
          </Card.Header>
          
          {#if showAIActions}
            <Card.Content class="pt-0">
              <div class="flex gap-2 flex-wrap">
                <Button 
                  size="sm" 
                  onclick={() => handleAIAction(law, 'summary')}
                >
                  <Bot class="h-3 w-3 mr-1" />
                  AI Summary
                </Button>
                <Button 
                  variant="outline" 
                  size="sm"
                  onclick={() => handleAIAction(law, 'chat')}
                >
                  <Bot class="h-3 w-3 mr-1" />
                  Ask AI
                </Button>
                {#if law.fullTextUrl}
                  <Button 
                    variant="outline" 
                    size="sm" 
                    asChild
                  >
                    <a href={law.fullTextUrl} target="_blank" rel="noopener noreferrer">
                      <ExternalLink class="h-3 w-3 mr-1" />
                      Full Text
                    </a>
                  </Button>
                {/if}
              </div>
            </Card.Content>
          {/if}
        </Card.Root>
      {/each}
    </div>
  {:else if searchQuery && !isSearching}
    <Card.Root>
      <Card.Content class="py-8 text-center">
        <p class="text-muted-foreground">
          No results found for "{searchQuery}".
        </p>
        <p class="text-sm text-muted-foreground mt-1">
          Try adjusting your search terms or use more general keywords.
        </p>
      </Card.Content>
    </Card.Root>
  {/if}
</div>

<style>
  :global(mark) {
    background-color: theme(colors.yellow.200);
    padding: 0.125rem 0.25rem;
    border-radius: 0.25rem;
    font-weight: 500;
  }
  
  :global(.dark mark) {
    background-color: theme(colors.yellow.900);
    color: theme(colors.yellow.100);
  }
</style>