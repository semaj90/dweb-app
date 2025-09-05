<script lang="ts">
  interface Props {
    data?: any;
  }
  let {
    data
  } = $props();



import { writable } from 'svelte/store';
import { browser } from '$app/environment';
import { semanticSearch } from '$lib/ai/mcp-helpers';

// Simple debounce utility
function debounce<T extends (...args: any[]) => void>(fn: T, ms: number) {
  let timeout: ReturnType<typeof setTimeout>;
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => fn(...args), ms);
  };
}

const query = writable(data?.initialQuery || '');
const results = writable(data?.initialResults || []);
const loading = writable(false);

if (browser) {
  const debouncedSearch = debounce(async (q: string) => {
    if (!q) { results.set([]); return; }
    loading.set(true);
    const res = await semanticSearch(q);
    results.set(res);
    loading.set(false);
  }, 400);

  query.subscribe((q) => {
    debouncedSearch(q);
  });
}
</script>

<h1>Semantic Search Demo (SSR + Hydration + Debounce)</h1>
<input
  type="text"
  bind:value={$query}
  placeholder="Type your legal query..."
  autocomplete="off"
/>
{#if $loading}
  <p>Searching...</p>
{/if}
<ul>
  {#each $results as result}
    <li>{result.text || JSON.stringify(result)}</li>
  {/each}
</ul>

<style>
input {
  width: 100%;
  padding: 0.5em;
  font-size: 1.1em;
  margin-bottom: 1em;
}
</style>
