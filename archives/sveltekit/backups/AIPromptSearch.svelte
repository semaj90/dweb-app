<script lang="ts">
  import { aiHistory } from "$lib/stores/aiHistoryStore";
  import Fuse from "fuse.js";
  import { onMount } from "svelte";

  let query = "";
  let results: unknown[] = [];
  let fuse: Fuse<any>;

  $: history = $aiHistory;

  onMount(() => {
    fuse = new Fuse(history, {
      keys: ["prompt", "response"],
      threshold: 0.3,
    });
  });

  $: results = query && fuse ? fuse.search(query).map((r) => r.item) : history;
</script>

<div className="${1}">
  <input
    type="text"
    bind:value={query}
    placeholder="Search AI history..."
    className="${1}"
  />
  <ul className="${1}">
    {#each results as item}
      <li className="${1}">
        <div className="${1}">{item.prompt}</div>
        <div className="${1}">{item.response}</div>
        <div className="${1}">{item.timestamp}</div>
      </li>
    {/each}
  </ul>
</div>
