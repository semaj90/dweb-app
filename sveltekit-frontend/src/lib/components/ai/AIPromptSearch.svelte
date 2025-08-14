<!-- @migration-task Error while migrating Svelte code: Unexpected token
https://svelte.dev/e/js_parse_error -->
<script lang="ts">
  import { aiHistory } from "$lib/stores/aiHistoryStore";
  import Fuse from "fuse.js";
  import { onMount } from "svelte";

  let query = "";
  let results: any[] = [];
  let fuse: Fuse<any>;

  let history = $derived($aiHistory;);

  onMount(() => {
    fuse = new Fuse(history, {
      keys: ["prompt", "response"],
      threshold: 0.3,
    });
  });

  let results = $derived(query && fuse ? fuse.search(query).map((r) => r.item) : history;);
</script>

<div class="space-y-4">
  <input
    type="text"
    bind:value={query}
    placeholder="Search AI history..."
    class="space-y-4"
  />
  <ul class="space-y-4">
    {#each results as item}
      <li class="space-y-4">
        <div class="space-y-4">{item.prompt}</div>
        <div class="space-y-4">{item.response}</div>
        <div class="space-y-4">{item.timestamp}</div>
      </li>
    {/each}
  </ul>
</div>
