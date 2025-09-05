<script lang="ts">
  import { aiHistory } from "$lib/stores/aiHistoryStore";
  import Fuse from "fuse.js";
  import { onMount } from "svelte";

  let query = "";
  let results: any[] = [];
  let fuse: Fuse<any>

  $: history = $aiHistory;

  onMount(() => {
    fuse = new Fuse(history, {
      keys: ["prompt", "response"],
      threshold: 0.3,
    });
  });

  $: results = query && fuse ? fuse.search(query).map((r) => r.item) : history;
</script>

<div class="container mx-auto px-4">
  <input
    type="text"
    bind:value={query}
    placeholder="Search AI history..."
    class="container mx-auto px-4"
  />
  <ul class="container mx-auto px-4">
    {#each results as item}
      <li class="container mx-auto px-4">
        <div class="container mx-auto px-4">{item.prompt}</div>
        <div class="container mx-auto px-4">{item.response}</div>
        <div class="container mx-auto px-4">{item.timestamp}</div>
      </li>
    {/each}
  </ul>
</div>
