<script lang="ts">
import Badge from "$lib/components/ui/Badge.svelte";
  import { $derived } from 'svelte';
import Button from "$lib/components/ui/Button.svelte";
import Input from "$lib/components/ui/Input.svelte";
import Fuse from "fuse.js";
import { createEventDispatcher, onMount } from "svelte";

    const dispatch = createEventDispatcher();

  // Keep selectedNode untyped externally; use sn (any) internally for a safe quick-pass
  export let selectedNode: any = null;
  // Use derived form to avoid duplicate declarations and ensure reactive evaluation
  let sn: any = $derived(() => selectedNode);

    let isProcessing = false;
    let processingStatus = "";

    let searchQuery = "";
    let searchResults: any[] = [];
    let fuse: Fuse<any> | null = null;

    let allEvidence: any[] = [];

    let aiInsights: any = { connections: [], similarEvidence: [], timeline: [], suggestedActions: [] };

  // template-friendly aliases (derived values)
  let ic = $derived(aiInsights.connections || []);
  let isim = $derived(aiInsights.similarEvidence || []);
  let iactions = $derived(aiInsights.suggestedActions || []);

    onMount(() => {
      // placeholder: load evidence later
      allEvidence = [];
      if (allEvidence.length > 0) {
        fuse = new Fuse(allEvidence, { keys: ["name", "tags", "title"], threshold: 0.4 });
      }
    });

    function performSearch() {
      if (!fuse || !searchQuery.trim()) {
        searchResults = [];
        return;
      }
      const results = fuse.search(searchQuery);
      searchResults = results.map(r => ({ ...r.item, score: r.score }));
    }

    function clearSearch() {
      searchQuery = "";
      searchResults = [];
    }

    async function reprocessWithAI() {
      if (!sn || isProcessing) return;
      isProcessing = true;
      processingStatus = "Analyzing with AI...";
      try {
        const response = await fetch("/api/ai/tag", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ content: sn.content, fileName: sn.name }) });
        if (response.ok) {
          const newTags = await response.json();
          sn.aiTags = newTags;
          dispatch("tagsUpdate", newTags);
          processingStatus = "Analysis complete!";
          aiInsights = { connections: [], similarEvidence: [], timeline: [], suggestedActions: [] };
        } else {
          processingStatus = "Analysis failed";
        }
      } catch (err) {
        console.error(err);
        processingStatus = "Analysis failed";
      } finally {
        isProcessing = false;
        setTimeout(() => (processingStatus = ""), 2500);
      }
    }
  </script>

  <div class="ai-assistant">
    <h2>AI Assistant</h2>

    <div class="search">
      <Input bind:value={searchQuery} placeholder="Search evidence..." />
      {#if searchQuery}
        <Button onclick={clearSearch} variant="outline" size="sm">Clear</Button>
      {/if}

      {#if searchResults.length > 0}
        <div>{searchResults.length} results</div>
        {#each searchResults.slice(0,5) as r}
          <div>
            <div>{r.name}</div>
            <div>Score: {(1 - (r.score ?? 0)).toFixed(2)}</div>
          </div>
        {/each}
      {/if}
    </div>

    {#if sn}
      {#if processingStatus}
        <div>{processingStatus}</div>
      {/if}

      {#if sn.aiTags}
        <div>
          <h3>AI Analysis</h3>
          <Button onclick={reprocessWithAI} disabled={isProcessing}>{isProcessing ? 'Processing...' : 'Re-analyze'}</Button>
          {#if sn.aiTags.summary}
            <div><strong>Summary</strong><div>{sn.aiTags.summary}</div></div>
          {/if}
          {#if sn.aiTags.tags?.length}
            <div>{#each sn.aiTags.tags as tag}<Badge>{tag}</Badge>{/each}</div>
          {/if}
        </div>
      {:else}
        <div>
          <div>🤖 No AI analysis available</div>
          <Button onclick={reprocessWithAI} disabled={isProcessing}>{isProcessing ? 'Processing...' : 'Analyze with AI'}</Button>
        </div>
      {/if}

      {#if ic.length || isim.length || iactions.length}
        <div>
          <h3>Insights</h3>
          {#if ic.length}
            <div>
              <strong>Connections</strong>
              {#each ic as c}<div>{c.entity} — {c.description}</div>{/each}
            </div>
          {/if}
          {#if isim.length}
            <div>
              <strong>Similar Evidence</strong>
              {#each isim as s}<div>{s.name} — {s.reason}</div>{/each}
            </div>
          {/if}
        </div>
      {/if}
    {/if}
  </div>

  <!-- Quick-pass: no style block to avoid @apply CSS warnings; reintroduce styles later -->
