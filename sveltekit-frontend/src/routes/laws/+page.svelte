<script lang="ts">
  import {
    Card,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle,
  } from '$lib/components/ui/card';
  import { Input } from '$lib/components/ui/input/index.js';
  import { Badge } from '$lib/components/ui/badge/index.js';
  import { Search, BookOpen, ExternalLink, Bot, MessageSquare } from 'lucide-svelte';
  import { onMount } from 'svelte';

  // In Svelte 5 (runes mode) don't use `export let` for page props — use $props()
  const { data } = $props() as { data: any };
let EnhancedFuseSearch = $state<any >(null);

  onMount(async () => {
    EnhancedFuseSearch = (await import('$lib/components/search/EnhancedFuseSearch.svelte')).default;
  });

  // Simple search state
let searchQuery = $state<string >('');
let searchResults = $state<any[] >([]);
let isSearching = $state<boolean >(false);

  async function performSearch() {
    if (!searchQuery.trim()) return;

    isSearching = true;
    try {
      const params = new URLSearchParams({
        q: searchQuery,
        jurisdiction: 'all',
        category: 'all',
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

  function handleKeydown(event: KeyboardEvent) {
    if (event.key === 'Enter') {
      performSearch();
    }
  }

  // AI toolbar event handlers (typed)
  function handleAISearchResult(result: any) {
    console.log('AI Search Result:', result);
    if (result?.laws) {
      searchResults = result.laws;
    }
  }

  function handleAIChatResult(result: any) {
    console.log('AI Chat Result:', result);
  }

  function handleAISummarizeResult(result: any) {
    console.log('AI Summarization Result:', result);
  }
</script>

<svelte:head>
  <title>Legal Resources - Laws & Regulations | YoRHa Legal AI</title>
  <meta
    name="description"
    content="Browse California and state laws with AI-powered search and summaries" />
  <!-- NES.css (optional) -->
  <link rel="stylesheet" href="https://unpkg.com/nes.css@2.3.0/css/nes.min.css" />
</svelte:head>

<div class="container mx-auto py-8 space-y-8 nes-container is-rounded">
  <!-- Header -->
  <div class="text-center space-y-4">
    <h1 class="text-4xl font-bold tracking-tight">Legal Resources</h1>
    <p class="text-xl text-muted-foreground max-w-2xl mx-auto">
      Browse California and state laws with AI-powered search and summaries
    </p>
  </div>

  <!-- Enhanced Fuse.js Search (client-only) -->
  {#if EnhancedFuseSearch}
    <${EnhancedFuseSearch}
      maxResults={10}
      bind:results={searchResults}
      class="mb-4"
      on:select={(e: CustomEvent) => {
        const selected = e.detail;
        if (selected?.title) {
          searchQuery = selected.title;
        }
      }}
    />
  {/if}
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
      <div class="flex gap-2">
        <Input
          placeholder="Search laws, codes, regulations..."
          bind:value={searchQuery}
          on:keydown={handleKeydown}
          class="flex-1" />
        <button on:onclick={performSearch} disabled={isSearching || !searchQuery.trim()} class="inline-flex items-center justify-center gap-2 rounded-md bg-primary text-primary-foreground py-2 px-3 hover:opacity-90 transition disabled:opacity-50">
          {#if isSearching}
            Loading...
          {:else}
            <Search class="h-4 w-4 mr-2" />
            Search
          {/if}
        </button>
      </div>

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
            <a
              href={link.url}
              target="_blank"
              rel="noopener noreferrer"
              class="w-full inline-flex items-center justify-center gap-2 rounded-md bg-primary text-primary-foreground py-2 px-3 hover:opacity-90 transition">
              <ExternalLink class="h-4 w-4" />
              Browse {link.title}
            </a>
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
              <div class="flex gap-2">
                <button on:onclick={() => handleAISummarizeResult(law)} class="inline-flex items-center gap-2 rounded-md bg-primary text-primary-foreground px-3 py-1 text-sm">
                  <Bot class="h-4 w-4 mr-2" />
                  <span>AI Summary</span>
                </button>
                <button on:onclick={() => handleAIChatResult(law)} class="inline-flex items-center gap-2 rounded-md border px-3 py-1 text-sm">
                  <MessageSquare class="h-4 w-4 mr-2" />
                  <span>AI Chat</span>
                </button>
                {#if law.fullTextUrl}
                  <a href={law.fullTextUrl} target="_blank" rel="noopener noreferrer" class="inline-flex items-center gap-2 rounded-md border px-3 py-1 text-sm">
                    Full Text
                  </a>
                {/if}
              </div>
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
