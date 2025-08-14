<script lang="ts">
  interface Props {
    onsummary?: (event?: any) => void;
  }


    let summary = "";
  let loading = false;
  
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
    onsummary?.();
}
</script>

<button
  class="space-y-4"
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
  <div class="space-y-4">
    <div class="space-y-4">AI Summary</div>
    <div>{summary}</div>
  </div>
{/if}
