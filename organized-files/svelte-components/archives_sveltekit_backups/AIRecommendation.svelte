<script lang="ts">
  import { aiHistory } from "$lib/stores/aiHistoryStore";
  import Fuse from "fuse.js";
  import { onMount } from "svelte";

  let recommendations: unknown[] = [];
  let fuse: Fuse<any>;

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

<div class="mx-auto px-4 max-w-7xl">
  <h3 class="mx-auto px-4 max-w-7xl">Recommended Next Actions</h3>
  <ul class="mx-auto px-4 max-w-7xl">
    {#each recommendations as item}
      <li class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">{item.prompt}</div>
        <div class="mx-auto px-4 max-w-7xl">{item.response}</div>
      </li>
    {/each}
  </ul>
</div>
