<!--
  AiAssistant.svelte
  - Production-ready, context7-compliant, SvelteKit 5, XState, Loki.js, and global store integration
  - Handles: streaming, memoization, global state, evidence source highlighting, and save-to-DB
  - Backend: expects /api/ai/process-evidence (LangChain, Ollama, pg_vector, Neo4j, Redis, Docker)
-->
<script lang="ts">
  import { getContext, onMount } from 'svelte';

  // UI components
  import { Button } from '$lib/components/ui';
  import { Card } from '$lib/components/ui';
  import { useAIGlobalStore } from '$lib/stores/ai';

  // Get user from context (SSR-safe)
  const getUser = getContext('user');
  const user = typeof getUser === 'function' ? getUser() : undefined;

  export let contextItems: any[] = [];
  export let caseId: string = '';

  // Use the global AI store (XState-based, memoized, streaming-ready)
  const { state, send } = useAIGlobalStore();

  // Example: Move getSummaryCache() into onMount to avoid SSR issues
  onMount(() => {
    // getSummaryCache(); // Uncomment and use this if you need to initialize cache on client
  });

  // Trigger summary
  function handleSummarize() {
    if (!user?.id) return;
    send({ type: 'SUMMARIZE', caseId, evidence: contextItems, userId: user.id });
  }

  // Save summary to DB
  async function saveSummary() {
    if (!$state.context.summary || !caseId) return;
    await fetch('/api/summary/save', {
      method: 'POST',
      body: JSON.stringify({ caseId, summary: $state.context.summary }),
      headers: { 'Content-Type': 'application/json' }
    });
    // Optionally show a notification here
  }
</script>

<Card>
  <div class="flex items-center justify-between mb-2">
    <span class="font-bold text-lg">AI Evidence Summary</span>
    <Button on:click={handleSummarize} disabled={!user || $state.context.loading} class="uno-bg-blue-600 uno-text-white uno-px-3 uno-py-1 uno-rounded">
      {!user ? 'Sign in to Summarize' : ($state.context.loading ? 'Summarizing...' : 'Summarize Evidence')}
    </Button>
    <Button on:click={saveSummary} disabled={!$state.context.summary || $state.context.loading} class="uno-bg-green-600 uno-text-white uno-px-3 uno-py-1 uno-rounded ml-2">
      Save Summary
    </Button>
  </div>
  <div class="mt-2 text-gray-700 text-sm">
    {#if $state.context.loading}
      <span class="text-gray-400">Summarizing evidence...</span>
      <!-- Streaming output (if supported) -->
      {#if $state.context.stream}
        <pre>{$state.context.stream}</pre>
      {/if}
    {:else if $state.context.error}
      <span class="text-red-500">{$state.context.error}</span>
    {:else if $state.context.summary}
      <pre>{$state.context.summary}</pre>
      <!-- Top 3 evidence sources (if available) -->
      {#if $state.context.sources && $state.context.sources.length > 0}
        <div class="mt-2">
          <span class="font-semibold">Top Evidence Used:</span>
          <ol class="list-decimal ml-6">
            {#each $state.context.sources.slice(0, 3) as item, i}
              <li>{item.title || item.id || `Evidence #${i+1}`}</li>
            {/each}
          </ol>
        </div>
      {/if}
    {:else}
      <span class="text-gray-400">No summary yet.</span>
    {/if}
  </div>
</Card>

<style>
  /* @unocss-include */
  .uno-shadow {
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  }
</style>
