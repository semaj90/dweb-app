<script lang="ts">
  let query = '';
  let results = [];

  async function search() {
    const res = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    });
    results = await res.json();
  }
</script>

<input bind:value={query} placeholder="Search documents..." />
<button on:click={search}>Search</button>

{#each results as result}
  <div>
    <h3>{result.filename}</h3>
    <p>Similarity: {result.similarity}</p>
    <p>{result.content?.substring(0, 200)}...</p>
  </div>
{/each}
