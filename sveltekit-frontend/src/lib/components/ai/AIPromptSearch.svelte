<script lang="ts">
  import { $derived } from 'svelte';

  import { aiHistory } from "$lib/stores/aiHistoryStore";
  import Fuse from "fuse.js";
  import { onMount } from "svelte";
let query = $state("");
  let fuse: Fuse<any>;

  let history = $derived($aiHistory);

  onMount(() => {
    fuse = new Fuse(history, {
      keys: ["prompt", "response"],
      threshold: 0.3,
    });
  });

  let results = $derived(query && fuse ? fuse.search(query).map((r) => r.item) : history);
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


