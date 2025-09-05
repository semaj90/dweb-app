<script lang="ts">
  import { aiHistory } from "$lib/stores/aiHistoryStore";
  import Fuse from "fuse.js";
  import { onMount } from "svelte";

  let recommendations: any[] = [];
  let fuse: Fuse<any>

  $: history = $aiHistory;

  onMount(() => {
    fuse = new Fuse(history, {
      keys: ["prompt", "response"],
      threshold: 0.2,
    });
    // Example: recommend based on last prompt
    if (history.length > 0) {
      const lastPrompt = history[history.length - 1].prompt;
      recommendations = fuse
        .search(lastPrompt)
        .map((r) => r.item)
        .slice(0, 3);
}
  });
</script>

<div class="container mx-auto px-4">
  <h3 class="container mx-auto px-4">Recommended Next Actions</h3>
  <ul class="container mx-auto px-4">
    {#each recommendations as item}
      <li class="container mx-auto px-4">
        <div class="container mx-auto px-4">{item.prompt}</div>
        <div class="container mx-auto px-4">{item.response}</div>
      </li>
    {/each}
  </ul>
</div>
