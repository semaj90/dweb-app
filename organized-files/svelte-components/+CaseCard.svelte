<script lang="ts">
  import type { Case } from '$lib/data/types';

  export let caseItem: Case;

</script>

<div class="card h-100 shadow-sm">
  <div class="card-body d-flex flex-column">
    <h5 class="card-title text-primary">{caseItem.title}</h5>
    {#if caseItem.createdAt}
      <p class="card-text text-muted mb-2"><small>Created: {new Date(caseItem.createdAt).toLocaleDateString()}</small></p>
    {/if}
    <p class="card-text flex-grow-1">{caseItem.description ? caseItem.description.substring(0, 100) : 'No description.'}{caseItem.description && caseItem.description.length > 100 ? '...' : ''}</p>
    <a href="/case/{caseItem.id}" class="btn btn-outline-primary mt-auto" aria-label="View details for {caseItem.title}">View Details &rarr;</a>
  </div>
</div>
