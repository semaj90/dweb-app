<!-- Phase 10: Audit Results UI Scaffold (Context7)
This Svelte component displays semantic audit results and TODOs from the backend.
TODO: After initial test, wire up real Context7 audit API, agent triggers, and live updates. -->
<script lang="ts">
  import { onMount } from 'svelte';
  import type { SemanticAuditResult } from '$lib/ai/types';

  let auditResults: SemanticAuditResult[] = $state([]);
  let loading = $state(true);
  let error: string | null = $state(null);

  // Fetch audit results from backend
  async function fetchAuditResults() {
    loading = true;
    error = null;
    try {
      const res = await fetch('/api/audit/semantic', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: 'Context7 pipeline audit' })
      });
      if (!res.ok) throw new Error('Failed to fetch audit results');
      const data = await res.json();
      auditResults = data.results || [];
    } catch (e: any) {
      error = e.message || 'Unknown error';
    } finally {
      loading = false;
    }
  }

  onMount(fetchAuditResults);

  // TODO: Add actions to trigger agent fixes, mark TODOs as resolved, and live update from backend
</script>

<div class="space-y-6">
  <h2 class="text-xl font-bold">Pipeline Audit Results</h2>
  {#if loading}
    <div>Loading audit results...</div>
  {:else if error}
    <div class="text-red-600">{error}</div>
  {:else if auditResults.length === 0}
    <div>No audit results found.</div>
  {:else}
    <ul class="space-y-2">
      {#each auditResults as result}
        <li class="border rounded p-3 flex flex-col gap-1">
          <div class="font-semibold">{result.step}</div>
          <div class="text-sm">{result.message}</div>
          {#if result.suggestedFix}
            <div class="text-amber-700 text-xs">Suggested fix: {result.suggestedFix}</div>
          {/if}
          <div class="flex gap-2 mt-1">
            <span class="text-xs px-2 py-1 rounded bg-gray-100">{result.status}</span>
            {#if result.agentTriggered}
              <span class="text-xs px-2 py-1 rounded bg-blue-100">Agent triggered</span>
            {/if}
            <!-- TODO: Add button to trigger agent action for this TODO -->
          </div>
        </li>
      {/each}
    </ul>
  {/if}
</div>

<!-- TODO: After initial test, wire up agent action buttons, live updates, and best practice docs. -->
<style>
/* Uses Yorha/Phase10/Context7 design system classes */
</style>
<!-- #context7 #Phase10 #todo: Wire up agent trigger, improve UI, connect to real backend after test -->


