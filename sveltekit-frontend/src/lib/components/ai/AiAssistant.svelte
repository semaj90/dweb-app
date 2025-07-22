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
  const { snapshot, send, actorRef } = useAIGlobalStore();

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
    if (!$snapshot.context.summary || !caseId) return;
    await fetch('/api/summary/save', {
      method: 'POST',
      body: JSON.stringify({ caseId, summary: $snapshot.context.summary }),
      headers: { 'Content-Type': 'application/json' }
    });
    // Optionally show a notification here
  }
</script>

<Card class="nier-card p-6">
  <div class="flex items-center justify-between mb-4">
    <h3 class="nier-title text-xl font-bold">AI Evidence Summary</h3>
    <div class="flex gap-2">
      <Button
        onclick={handleSummarize}
        disabled={!user || $snapshot.context.loading}
        variant="primary"
        class="nier-button"
      >
        {!user ? 'Sign in to Summarize' : ($snapshot.context.loading ? 'Summarizing...' : 'Summarize Evidence')}
      </Button>
      <Button
        onclick={saveSummary}
        disabled={!$snapshot.context.summary || $snapshot.context.loading}
        variant="success"
        class="nier-button"
      >
        Save Summary
      </Button>
    </div>
  </div>

  <div class="nier-content">
    {#if $snapshot.context.loading}
      <div class="nier-loading">
        <span class="nier-text-muted">Summarizing evidence...</span>
        <!-- Streaming output (if supported) -->
        {#if $snapshot.context.stream}
          <pre class="nier-code mt-2">{$snapshot.context.stream}</pre>
        {/if}
      </div>
    {:else if $snapshot.context.error}
      <div class="nier-error p-3 rounded">
        <span class="text-red-600">{$snapshot.context.error}</span>
      </div>
    {:else if $snapshot.context.summary}
      <div class="nier-summary">
        <pre class="nier-code whitespace-pre-wrap">{$snapshot.context.summary}</pre>
        <!-- Top 3 evidence sources (if available) -->
        {#if $snapshot.context.sources && $snapshot.context.sources.length > 0}
          <div class="nier-sources mt-4 pt-4 border-t border-gray-200">
            <h4 class="nier-subtitle font-semibold mb-2">Top Evidence Used:</h4>
            <ol class="nier-list space-y-1">
              {#each $snapshot.context.sources.slice(0, 3) as item, i}
                <li class="nier-list-item">
                  <span class="nier-badge">{i + 1}</span>
                  {item.title || item.id || `Evidence #${i+1}`}
                </li>
              {/each}
            </ol>
          </div>
        {/if}
      </div>
    {:else}
      <div class="nier-empty">
        <span class="nier-text-muted">No summary yet.</span>
      </div>
    {/if}
  </div>
</Card>

<style>
  /* Nier.css inspired styles */
  :global(.nier-card) {
    background: rgba(255, 255, 255, 0.95);
    border: 2px solid #000;
    box-shadow: 4px 4px 0 rgba(0, 0, 0, 0.1);
  }

  :global(.nier-title) {
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }

  :global(.nier-button) {
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
  }

  :global(.nier-button:hover) {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
  }

  :global(.nier-code) {
    background: #f4f4f4;
    border: 1px solid #ddd;
    padding: 1rem;
    font-family: 'Courier New', monospace;
    font-size: 0.875rem;
  }

  :global(.nier-error) {
    background: rgba(255, 0, 0, 0.05);
    border: 2px solid #ff0000;
  }

  :global(.nier-badge) {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: #000;
    color: #fff;
    border-radius: 50%;
    font-size: 0.75rem;
    margin-right: 0.5rem;
  }

  :global(.nier-text-muted) {
    color: #666;
    font-style: italic;
  }

  :global(.nier-list-item) {
    display: flex;
    align-items: center;
    padding: 0.5rem 0;
  }
</style>
