<!--
  AiAssistant.svelte
  - Production-ready, context7-compliant, SvelteKit 5, XState, Loki.js, and global store integration
  - Handles: streaming, memoization, global state, evidence source highlighting, and save-to-DB
  - Backend: expects /api/ai/process-evidence (LangChain, Ollama, pg_vector, Neo4j, Redis, Docker)
-->
<script lang="ts">
  import { getContext, onMount } from 'svelte';

  // UI components (Svelte 5 + melt v0.39.0 compatible)
  import Button from '$lib/components/ui/button/Button.svelte';
  import { Card } from '$lib/components/ui/card';
  import { aiGlobalStore, aiGlobalActions } from '$lib/stores/ai';

  // Type definition for AI store context
  interface AIStoreContext {
    loading?: boolean;
    error?: string;
    summary?: string;
    stream?: string;
    sources?: Array<{
      id?: string;
      title?: string;
    }>;
  }

  interface AIStore {
    context: AIStoreContext
  }

  // Get user from context (SSR-safe)
  const getUser = getContext('user');
  const user = typeof getUser === 'function' ? getUser() : undefined;

  interface Props {
    contextItems?: any[];
    caseId?: string;
  }

  let {
    contextItems = [],
    caseId = ''
  } = $props();

  // Use the global AI store (XState-based, memoized, streaming-ready)
  // Access store state via $aiGlobalStore, send actions via aiGlobalActions.send
  // The actorRef is not directly used in the component's script, but can be accessed via aiGlobalStore if needed.
  // const { snapshot, send, actorRef } = useAIGlobalStore(); // Old usage

  onMount(() => {
    // getSummaryCache(); // Uncomment and use this if you need to initialize cache on client
  });

  // Trigger summary
  function handleSummarize() {
    if (!user?.id) return;
    aiGlobalActions.summarize(caseId, contextItems, user?.id || '');
  }

    // Save summary to DB using the comprehensive summaries API
    async function saveSummary() {
      if (!(($aiGlobalStore as AIStore).context.summary) || !caseId || !user?.id) return;
      try {
        const response = await fetch('/api/summaries', {
          method: 'POST',
          body: JSON.stringify({
            type: 'case',
            targetId: caseId,
            depth: 'comprehensive',
            includeRAG: true,
            includeUserActivity: false,
            enableStreaming: false,
            userId: user.id
          }),
          headers: { 'Content-Type': 'application/json' }
        });

        const result = await response.json();
        if (result.success) {
          // Optionally show a success notification here
          console.log('Summary saved successfully');
        } else {
          console.error('Failed to save summary:', result.error);
        }
      } catch (error) {
        console.error('Error saving summary:', error);
      }
    }
  </script>

  <Card class="nier-card p-6">
    <div class="nier-header mb-4">
      <h3 class="nier-title text-lg font-bold mb-2">AI Evidence Summary</h3>
    <div class="flex gap-2">
      <Button
        onclick={handleSummarize}
        disabled={!user || $aiGlobalStore.context.loading}
        variant="primary"
        class="relative overflow-hidden transition-all duration-300 hover:translate-y--0.5 hover:shadow-lg"
      >
        {!user ? 'Sign in to Summarize' : ($aiGlobalStore.context.loading ? 'Summarizing...' : 'Summarize Evidence')}
      </Button>
      <Button
        onclick={saveSummary}
        disabled={!$aiGlobalStore.context.summary || $aiGlobalStore.context.loading}
        variant="primary"
        class="relative overflow-hidden transition-all duration-300 hover:translate-y--0.5 hover:shadow-lg"
      >
        Save Summary
      </Button>
    </div>
  </div>

  <div class="nier-content">
    {#if $aiGlobalStore.context.loading}
      <div class="nier-loading">
        <span class="nier-text-muted">Summarizing evidence...</span>
        <!-- Streaming output (if supported) -->
        {#if $aiGlobalStore.context.stream}
          <pre class="nier-code mt-2">{$aiGlobalStore.context.stream}</pre>
        {/if}
      </div>
    {:else if $aiGlobalStore.context.error}
      <div class="nier-error p-3 rounded">
        <span class="text-red-600">{$aiGlobalStore.context.error}</span>
      </div>
    {:else if $aiGlobalStore.context.summary}
      <div class="nier-summary">
        <pre class="nier-code whitespace-pre-wrap">{$aiGlobalStore.context.summary}</pre>
        <!-- Top 3 evidence sources (if available) -->
        {#if $aiGlobalStore.context.sources && $aiGlobalStore.context.sources.length > 0}
          <div class="nier-sources mt-4 pt-4 border-t border-gray-200">
            <h4 class="nier-subtitle font-semibold mb-2">Top Evidence Used:</h4>
            <ol class="nier-list space-y-1">
              {#each $aiGlobalStore.context.sources.slice(0, 3) as item, i}
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
    text-transform: uppercase
  }

  :global(.nier-button) {
    position: relative
    overflow: hidden
    transition: all 0.3s ease;
  }

  :global(.nier-button\:hover) {
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
    align-items: center
    justify-content: center
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
    font-style: italic
  }

  :global(.nier-list-item) {
    display: flex
    align-items: center
    padding: 0.5rem 0;
  }
</style>
