<script lang="ts">
  import { Card, Modal, Button } from 'bits-ui';
  import { onMount } from 'svelte';
  export let contextItems = [];
  export let caseId = '';
  let summary = '';
  let loading = false;

  async function summarizeEvidence() {
    loading = true;
    try {
      const res = await fetch('/api/ai-summary', {
        method: 'POST',
        body: JSON.stringify({ caseId, evidence: contextItems }),
        headers: { 'Content-Type': 'application/json' }
      });
      const data = await res.json();
      summary = data.summary;
    } catch (err) {
      summary = 'Error generating summary.';
    }
    loading = false;
  }
</script>

<Card class="uno-p-4 uno-bg-white uno-shadow">
  <div class="flex items-center justify-between mb-2">
    <span class="font-bold text-lg">AI Evidence Summary</span>
    <Button on:click={summarizeEvidence} disabled={loading} class="uno-bg-blue-600 uno-text-white uno-px-3 uno-py-1 uno-rounded">
      {loading ? 'Summarizing...' : 'Summarize Evidence'}
    </Button>
  </div>
  <div class="mt-2 text-gray-700 text-sm">
    {#if summary}
      <pre>{summary}</pre>
    {:else}
      <span class="text-gray-400">No summary yet.</span>
    {/if}
  </div>
</Card>

<style>
  .uno-shadow {
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  }
</style>
