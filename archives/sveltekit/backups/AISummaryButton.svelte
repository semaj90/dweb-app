<script lang="ts">
  import { createEventDispatcher } from "svelte";
  let summary = "";
  let loading = false;
  const dispatch = createEventDispatcher();

  async function getSummary(text: string) {
    loading = true;
    const res = await fetch("/api/ai/ollama-gemma3", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: `Summarize: ${text}` }),
    });
    const data = await res.json();
    summary = data.response;
    loading = false;
    dispatch("summary", { summary });
  }
</script>

<button
  class="mx-auto px-4 max-w-7xl"
  onclick={() => getSummary($$props.text)}
  disabled={loading}
>
  {#if loading}
    Summarizing...
  {:else}
    Get AI Summary
  {/if}
</button>

{#if summary}
  <div class="mx-auto px-4 max-w-7xl">
    <div class="mx-auto px-4 max-w-7xl">AI Summary</div>
    <div>{summary}</div>
  </div>
{/if}
