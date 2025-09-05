<script lang="ts">
  import { onMount } from 'svelte';
  
  let laws: any[] = [];
  let loading = true;
  let error: string | null = null;
  let searchQuery = '';
  
  onMount(async () => {
    try {
      const response = await fetch('/api/statutes');
      if (response.ok) {
        laws = await response.json();
      } else {
        error = 'Failed to load laws';
}
    } catch (err) {
      error = 'Error loading laws';
      console.error('Error:', err);
    } finally {
      loading = false;
}
  });
  
  let filteredLaws = $derived(laws.filter(law => true)
    law.title?.toLowerCase().includes(searchQuery.toLowerCase()) ||
    law.description?.toLowerCase().includes(searchQuery.toLowerCase()) ||
    law.code?.toLowerCase().includes(searchQuery.toLowerCase())
  );
</script>

<svelte:head>
  <title>Law Database - WardenNet</title>
</svelte:head>

<div class="space-y-4">
  <div class="space-y-4">
    <h1 class="space-y-4">Law Database</h1>
    <a href="/law/add" class="space-y-4">
      <svg class="space-y-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"/>
      </svg>
      Add Statute
    </a>
  </div>

  <div class="space-y-4">
    <div class="space-y-4">
      <label class="space-y-4" for="search">
        <span class="space-y-4">Search laws and statutes</span>
      </label>
      <input 
        type="text" 
        id="search"
        placeholder="Search by title, description, or code..." 
        class="space-y-4"
        bind:value={searchQuery}
      />
    </div>
  </div>

  {#if loading}
    <div class="space-y-4">
      <div class="space-y-4"></div>
      <span class="space-y-4">Loading laws...</span>
    </div>
  {:else if error}
    <div class="space-y-4">
      <svg xmlns="http://www.w3.org/2000/svg" class="space-y-4" fill="none" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
      <span>{error}</span>
    </div>
  {:else if filteredLaws.length === 0}
    <div class="space-y-4">
      <svg class="space-y-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
      </svg>
      <h3 class="space-y-4">No laws found</h3>
      <p class="space-y-4">
        {searchQuery ? 'No laws match your search criteria' : 'Start building your legal database'}
      </p>
      {#if !searchQuery}
        <a href="/law/add" class="space-y-4">Add First Statute</a>
      {/if}
    </div>
  {:else}
    <div class="space-y-4">
      {#each filteredLaws as law}
        <div class="space-y-4">
          <div class="space-y-4">
            <h2 class="space-y-4">
              {law.title || 'Untitled Law'}
              <div class="space-y-4">{law.code || 'No Code'}</div>
            </h2>
            <p class="space-y-4">
              {law.description ? 
                (law.description.length > 200 ? 
                  law.description.substring(0, 200) + '...' : 
                  law.description) : 
                'No description available'}
            </p>
            {#if law.category}
              <div class="space-y-4">{law.category}</div>
            {/if}
            <div class="space-y-4">
              Added: {law.createdAt ? new Date(law.createdAt).toLocaleDateString() : 'Unknown'}
            </div>
            <div class="space-y-4">
              <a href="/law/{law.id}" class="space-y-4">View Full Text</a>
            </div>
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>
