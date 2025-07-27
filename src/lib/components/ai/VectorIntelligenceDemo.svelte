<script lang="ts">
  import { onMount } from 'svelte';
  import { writable } from 'svelte/store';

  const query = writable('');
  const results = writable([]);
  const loading = writable(false);

  async function search() {
    loading.set(true);
    const res = await fetch('/api/semantic-search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: $query })
    });
    const data = await res.json();
    results.set(data.results || []);
    loading.set(false);
  }
</script>

<div class="vector-demo">
  <input type="text" bind:value={$query} placeholder="Ask a question..." />
  <button on:click={search} disabled={$loading}>Search</button>
  {#if $loading}
    <p>Loading...</p>
  {/if}
  <ul>
    {#each $results as result}
      <li>{result.content}</li>
    {/each}
  </ul>
</div>

<style>
.vector-demo {
  max-width: 500px;
  margin: 2rem auto;
  padding: 1rem;
  border: 1px solid #ccc;
  border-radius: 8px;
}
input {
  width: 70%;
  padding: 0.5rem;
  margin-right: 0.5rem;
}
button {
  padding: 0.5rem 1rem;
}
ul {
  margin-top: 1rem;
}
</style>
