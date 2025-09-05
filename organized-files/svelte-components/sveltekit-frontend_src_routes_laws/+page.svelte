<script lang="ts">
  import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "$lib/components/ui/card/index.js";
  import { Input } from "$lib/components/ui/input/index.js";
  import { Button } from "$lib/components/ui/button/index.js";
  import { Badge } from "$lib/components/ui/badge/index.js";
  import { Search, BookOpen, ExternalLink, Bot, MessageSquare } from "lucide-svelte";
  import AIToolbar from "$lib/components/ai/AIToolbar.svelte";
  import EnhancedFuseSearch from "$lib/components/search/EnhancedFuseSearch.svelte";

  let { data } = $props();

  // Simple search state
  let searchQuery = $state('');
  let searchResults = $state([]);
  let isSearching = $state(false);

  async function performSearch() {
    if (!searchQuery.trim()) return;
    
    isSearching = true;
    try {
      const params = new URLSearchParams({
        q: searchQuery,
        jurisdiction: 'all',
        category: 'all'
      });

      const response = await fetch(`/api/laws/search?${params}`);
      const result = await response.json();
      
      if (result.success) {
        searchResults = result.laws || [];
      } else {
        searchResults = [];
        console.error('Search failed:', result.error);
      }
    } catch (error) {
      console.error('Search error:', error);
      searchResults = [];
    } finally {
      isSearching = false;
    }
  }

  function handleKeydown(event) {
    if (event.key === 'Enter') {
      performSearch();
    }
  }

  // AI toolbar event handlers
  function handleAISearchResult(result) {
    console.log('AI Search Result:', result);
    if (result.laws) {
      searchResults = result.laws;
    }
  }

  function handleAIChatResult(result) {
    console.log('AI Chat Result:', result);
    // Could show chat in a modal or side panel
  }

  function handleAISummarizeResult(result) {
    console.log('AI Summarization Result:', result);
    // Could show summary in a modal or notification
  }
</script>

<svelte:head>
  <title>Legal Resources - Laws & Regulations | YoRHa Legal AI</title>
  <meta name="description" content="Browse California and state laws with AI-powered search and summaries" />
</svelte:head>

<div class="container mx-auto py-8 space-y-8">
  <!-- Header -->
  <div class="text-center space-y-4">
    <h1 class="text-4xl font-bold tracking-tight">Legal Resources</h1>
    <p class="text-xl text-muted-foreground max-w-2xl mx-auto">
      Browse California and state laws with AI-powered search and summaries
    </p>
  </div>

  <!-- AI Toolbar -->
  <AIToolbar 
    onAISearch={handleAISearchResult}
    onAIChat={handleAIChatResult}
    onAISummarize={handleAISummarizeResult}
  />

  <!-- Enhanced Fuse.js Search -->
  <EnhancedFuseSearch maxResults={10} />

  <!-- Simple Search -->
  <Card>
    <CardHeader>
      <CardTitle class="flex items-center gap-2">
        <Search class="h-5 w-5" />
        Search Laws & Regulations
      </CardTitle>
    </CardHeader>
    <CardContent class="space-y-4">
      <div class="flex gap-2">
        <Input
          placeholder="Search laws, codes, regulations..."
          bind:value={searchQuery}
          onkeydown={handleKeydown}
          class="flex-1"
        />
        <Button onclick={performSearch} disabled={isSearching || !searchQuery.trim()}>
          {#if isSearching}
            Loading...
          {:else}
            <Search class="h-4 w-4 mr-2" />
            Search
          {/if}
        </Button>
      </div>
    </CardContent>
  </Card>

  <!-- Quick Links -->
  <div class="space-y-4">
    <h2 class="text-2xl font-semibold flex items-center gap-2">
      <BookOpen class="h-6 w-6" />
      Quick Access
    </h2>
    
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {#each data.quickLinks as link}
        <Card class="hover:shadow-lg transition-all duration-200">
          <CardHeader>
            <CardTitle class="text-lg">{link.title}</CardTitle>
            <CardDescription>{link.description}</CardDescription>
            <div class="flex gap-2">
              <Badge variant="secondary">{link.jurisdiction}</Badge>
              <Badge variant="outline">{link.category}</Badge>
            </div>
          </CardHeader>
          <CardContent>
            <Button class="w-full">
              <a href={link.url} target="_blank" rel="noopener noreferrer" class="flex items-center gap-2">
                <ExternalLink class="h-4 w-4" />
                Browse {link.title}
              </a>
            </Button>
          </CardContent>
        </Card>
      {/each}
    </div>
  </div>

  <!-- Search Results -->
  {#if searchResults.length > 0}
    <div class="space-y-4">
      <h2 class="text-2xl font-semibold">
        Search Results ({searchResults.length})
      </h2>
      
      <div class="space-y-4">
        {#each searchResults as law}
          <Card>
            <CardHeader>
              <CardTitle>{law.title}</CardTitle>
              <CardDescription>
                {law.jurisdiction} • {law.category}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p class="mb-4 text-sm">{law.description}</p>
              <div class="flex gap-2">
                <Button size="sm">
                  <Bot class="h-4 w-4 mr-2" />
                  AI Summary
                </Button>
                <Button variant="outline" size="sm">
                  <MessageSquare class="h-4 w-4 mr-2" />
                  AI Chat
                </Button>
                {#if law.fullTextUrl}
                  <Button variant="outline" size="sm">
                    <a href={law.fullTextUrl} target="_blank" rel="noopener noreferrer">
                      Full Text
                    </a>
                  </Button>
                {/if}
              </div>
            </CardContent>
          </Card>
        {/each}
      </div>
    </div>
  {:else if searchQuery && !isSearching}
    <Card>
      <CardContent class="py-8 text-center">
        <p class="text-muted-foreground">No results found for "{searchQuery}"</p>
      </CardContent>
    </Card>
  {/if}
</div>