<script lang="ts">
  import { $state } from 'svelte';
  import { apiClient } from '$lib/services/api-client';
  import { notificationStore as notify } from '$lib/stores/notifications';
  import { evidenceStore } from '$lib/stores/evidence-unified-fixed';
  import { createEventDispatcher } from 'svelte';

  // Props
  export let caseId: string | undefined = undefined; // Optional explicit case context
  export let autoClear: boolean = true;              // Clear IDs after successful queue

  const dispatch = createEventDispatcher<{ refreshed: void; queued: { queued: number; ids: string[]; mode: 'full' | 'incremental' } }>();
  // Minimal bits-ui placeholder (replace with real import if available)
  // import { Button } from 'bits-ui';

  let selectedIds = $state<string[]>([]);
  let inputValue = $state('');
  let loading = $state(false);
  let resultMessage = $state<string | null>(null);
  let addDebounce: NodeJS.Timeout | null = null;
  let lastIngestAt = 0;

  function scheduleAdd() {
    if (addDebounce) clearTimeout(addDebounce);
    addDebounce = setTimeout(() => {
      const raw = inputValue.trim();
      if (!raw) return;
      // Support comma / whitespace separated batch entry
      const parts = raw.split(/[\s,]+/).map(p => p.trim()).filter(Boolean);
      let added = 0;
      for (const p of parts) {
        if (p && !selectedIds.includes(p)) {
          selectedIds = [...selectedIds, p];
          added++;
        }
      }
      if (added > 0) {
        notify.info(`Added ${added} id${added>1?'s':''}` , { duration: 2500 });
        inputValue = '';
      }
    }, 250);
  }

  function addIdImmediate() {
    if (addDebounce) { clearTimeout(addDebounce); addDebounce = null; }
    const raw = inputValue.trim();
    if (!raw) return;
    const parts = raw.split(/[\s,]+/).map(p => p.trim()).filter(Boolean);
    let added = 0;
    for (const p of parts) {
      if (p && !selectedIds.includes(p)) {
        selectedIds = [...selectedIds, p];
        added++;
      }
    }
    if (added > 0) {
      notify.info(`Added ${added} id${added>1?'s':''}` , { duration: 2500 });
      inputValue = '';
    }
  }

  async function triggerIngestion(mode: 'full' | 'incremental' = 'incremental') {
    const now = Date.now();
    if (now - lastIngestAt < 1200) { // simple throttle
      notify.warning('Please wait a moment before triggering again');
      return;
    }
    if (selectedIds.length === 0) {
      resultMessage = 'Add at least one ID';
      notify.warning('Add at least one ID before ingestion');
      return;
    }
    lastIngestAt = now;
    loading = true;
    resultMessage = null;
    const queuedIds = [...selectedIds];
    notify.info(`Queuing ${queuedIds.length} item(s) for ${mode} ingestion...`, { duration: 3000 });
    try {
      const response = await apiClient.triggerEnhancedRagIngestion(queuedIds, mode);
      resultMessage = `Queued ${response.queued} item(s) (${mode})`;
      dispatch('queued', { queued: response.queued, ids: queuedIds, mode });
      notify.success(resultMessage, { duration: 4000 });
      try {
        // Refresh evidence if we have a caseId (preferred) otherwise let store subscription handle updates
        await evidenceStore.fetchEvidence(caseId ?? null as any);
        dispatch('refreshed');
      } catch (e) {
        console.warn('Evidence refresh failed (non-fatal):', e);
      }
      if (autoClear) {
        selectedIds = [];
      }
    } catch (e: any) {
      resultMessage = e.message || 'Network error';
      notify.error(`Ingestion failed: ${resultMessage}`);
    } finally {
      loading = false;
    }
  }
</script>

<div class="evidence-uploader space-y-3 p-4 border rounded bg-white/50">
  <h3 class="font-semibold text-sm">Enhanced RAG Ingestion</h3>
  <div class="flex gap-2 items-center">
  <input class="border px-2 py-1 text-sm flex-1" bind:value={inputValue} placeholder="Enter evidence/document ID(s)" input={scheduleAdd} keydown={(e)=> e.key==='Enter' && addIdImmediate()} />
  <button class="px-3 py-1 text-xs bg-gray-200 rounded hover:bg-gray-300" click={addIdImmediate}>Add</button>
  </div>
  {#if selectedIds.length > 0}
    <ul class="text-xs list-disc ml-4 space-y-1 max-h-24 overflow-auto pr-2">
      {#each selectedIds as id}
        <li class="flex items-center justify-between gap-2">
          <span class="truncate" title={id}>{id}</span>
          <button class="text-red-500 hover:text-red-700" click={() => selectedIds = selectedIds.filter(x => x !== id)}>✕</button>
        </li>
      {/each}
    </ul>
  {/if}
  <div class="flex gap-2">
    <button class="px-3 py-1 text-xs rounded bg-blue-600 text-white disabled:opacity-50" disabled={loading} click={() => triggerIngestion('incremental')}>Ingest Incremental</button>
    <button class="px-3 py-1 text-xs rounded bg-indigo-600 text-white disabled:opacity-50" disabled={loading} click={() => triggerIngestion('full')}>Ingest Full</button>
  </div>
  {#if loading}
    <p class="text-xs text-gray-500">Processing...</p>
  {/if}
  {#if resultMessage}
    <p class="text-xs" class:text-green-600={resultMessage.startsWith('Queued')} class:text-red-600={!resultMessage.startsWith('Queued')}>{resultMessage}</p>
  {/if}
</div>

<style>
  .evidence-uploader { font-family: system-ui, sans-serif; }
</style>
