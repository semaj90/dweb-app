<script lang="ts">
  import { onMount } from "svelte";
  let items: any[] = $state([]);
  let count = $state(0);
  onMount(async () => {
    try {
      const res = await fetch("/logs/svelte-suggestions.json");
      const data = await res.json();
      count = data.count;
      items = data.items;
    } catch {}
  });
</script>

<h1>Svelte Suggestions</h1>
<p>Total parsed: {count}</p>
<ul>
  {#each items as it}
    <li>
      <strong>{it.file}:{it.line}</strong>
      <div>{it.suggestion}</div>
    </li>
  {/each}
  {#if items.length === 0}
    <li>
      No suggestions file found yet. Run the VS Code task "Svelte: Generate
      Error Suggestions".
    </li>
  {/if}
</ul>

<style>
  li {
    margin: 8px 0;
  }
  strong {
    display: block;
  }
  div {
    color: #6cf;
  }
  ul {
    list-style: none;
    padding: 0;
  }
</style>
