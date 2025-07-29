<!-- AI Search Bar: Svelte 5, Bits UI, UnoCSS, analytics logging -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { Input, Button } from './index.js';
  import { Search } from 'lucide-svelte';
  import { $state } from 'svelte';

  export let placeholder = 'Ask AI...';
  export let userContext: any = {};
  export let neo4jContext: any = {};
  export let analyticsLog: (event: any) => void = () => {};
  export let onResults: (results: any) => void = () => {};

  let query = $state('');
  let loading = $state(false);
  const dispatch = createEventDispatcher();

  async function handleSearch() {
    if (!query) return;
    loading = true;
    analyticsLog({ event: 'ai_search_submitted', query, userContext, timestamp: Date.now() });
    try {
      const res = await fetch('/api/semantic-search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, userContext, neo4jContext })
      });
      const data = await res.json();
      analyticsLog({ event: 'ai_search_result', query, resultCount: data.results?.length, timestamp: Date.now() });
      onResults(data.results);
      dispatch('results', { results: data.results });
    } catch (error) {
      analyticsLog({ event: 'ai_search_error', query, error: error.message, timestamp: Date.now() });
    } finally {
      loading = false;
    }
  }

  function handleKeyDown(e: KeyboardEvent) {
    if (e.key === 'Enter') handleSearch();
  }
</script>

<div class="flex gap-2 items-center w-full">
  <Input
    variant="search"
    bind:value={query}
    placeholder={placeholder}
    icon={Search}
    iconPosition="left"
    class="flex-1 vector-search-input"
    on:keydown={handleKeyDown}
    legal
    aiAssisted
  />
  <Button variant="yorha" on:click={handleSearch} loading={loading} legal>
    <Search class="w-4 h-4 mr-1" />
    Search
  </Button>
</div>
