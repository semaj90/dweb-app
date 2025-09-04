<script lang="ts">
  import { page } from '$app/stores';
  import { goto } from '$app/navigation';
  import { writable, derived } from 'svelte/store';
  import { enhance } from '$app/forms';
  import { browser } from '$app/environment';
  import CaseListItem from '$lib/components/cases/CaseListItem.svelte';
  import CaseFilters from '$lib/components/cases/CaseFilters.svelte';
  import CaseStats from '$lib/components/cases/CaseStats.svelte';
  import { casesStore } from "$lib/stores/casesStore";
  import { Search, Filter, SortAsc, SortDesc, Plus, RefreshCw } from 'lucide-svelte';
  import type { LayoutData } from "./$types";

  export let data: LayoutData;

  // Sync server data with client store
  $: if (browser) {
    casesStore.set({
      cases: data.userCases,
      stats: data.caseStats,
      filters: {
        search: data.searchQuery,
        status: data.statusFilter,
        priority: data.priorityFilter,
        sort: data.sortBy,
        order: data.sortOrder
      }
    });
  }

  // Reactive derived stores for UI state
  $: activeCaseId = $page.url.searchParams.get('view');
  $: isModalOpen = $page.url.searchParams.has('view');
  $: selectedCase = data.userCases.find(c => c.id === activeCaseId);

  // Loading state for AJAX operations
  const isLoading = writable(false);
  const isFiltering = writable(false);

  // Enhanced form submission with loading state and partial updates
  const handleFilterSubmit = () => {
    isFiltering.set(true);
    return async ({ result, update }) => {
      if (result.type === 'success' && result.data) {
        // Update only the cases data without full page reload
        casesStore.update(store => ({
          ...store,
          cases: result.data.cases,
          filters: result.data.filters
        }));
        
        // Update URL params to reflect filter state
        const url = new URL($page.url);
        const filters = result.data.filters;
        
        if (filters.search) url.searchParams.set('search', filters.search);
        else url.searchParams.delete('search');
        
        if (filters.status !== 'all') url.searchParams.set('status', filters.status);
        else url.searchParams.delete('status');
        
        if (filters.priority !== 'all') url.searchParams.set('priority', filters.priority);
        else url.searchParams.delete('priority');
        
        if (filters.sort !== 'openedAt') url.searchParams.set('sort', filters.sort);
        else url.searchParams.delete('sort');
        
        if (filters.order !== 'desc') url.searchParams.set('order', filters.order);
        else url.searchParams.delete('order');
        
        goto(url.toString(), { replaceState: true, keepFocus: true, noScroll: true });
      } else {
        await update();
      }
      isFiltering.set(false);
    };
  };

  // Close modal/case view
  function closeCase() {
    const url = new URL($page.url);
    url.searchParams.delete('view');
    goto(url.toString(), { keepFocus: true, noScroll: true });
  }

  // Open case view
  function openCase(caseId: string) {
    const url = new URL($page.url);
    url.searchParams.set('view', caseId);
    goto(url.toString(), { keepFocus: true, noScroll: true });
  }

  // Quick case actions
  async function quickAction(caseId: string, action: string) {
    isLoading.set(true);
    try {
      const response = await fetch(`/api/cases/${caseId}/${action}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (response.ok) {
        // Refresh the layout data
        goto($page.url.pathname + $page.url.search, { invalidateAll: true });
      }
    } catch (error) {
      console.error('Action failed:', error);
    } finally {
      isLoading.set(false);
    }
  }

  // Handle quick status update
  async function updateCaseStatus(caseId: string, status: string) {
    isLoading.set(true);
    try {
      const formData = new FormData();
      formData.append('caseId', caseId);
      formData.append('status', status);
      
      const response = await fetch('?/updateStatus', {
        method: 'POST',
        body: formData
      });
      
      if (response.ok) {
        // Update the case in the store
        casesStore.update(store => ({
          ...store,
          cases: store.cases.map(c => 
            c.id === caseId ? { ...c, status } : c
          )
        }));
      }
    } catch (error) {
      console.error('Status update failed:', error);
    } finally {
      isLoading.set(false);
    }
  }

  // Handle keyboard shortcuts
  function handleKeydown(event: KeyboardEvent) {
    if (event.key === 'Escape' && isModalOpen) {
      closeCase();
    }
  }
</script>

<svelte:window onkeydown={handleKeydown} />

<div class="mx-auto px-4 max-w-7xl">
  <div class="mx-auto px-4 max-w-7xl">
    
    <!-- Left Column: Case List & Filters -->
    <aside class="mx-auto px-4 max-w-7xl">
      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">
          <div>
            <h1 class="mx-auto px-4 max-w-7xl">Cases</h1>
            <p class="mx-auto px-4 max-w-7xl">{data.userCases.length} cases</p>
          </div>
          <button
            onclick={() => goto('/cases/new')}
            class="mx-auto px-4 max-w-7xl"
          >
            <Plus class="mx-auto px-4 max-w-7xl" />
            New
          </button>
        </div>
      </div>

      <!-- Search & Filters -->
      <div class="mx-auto px-4 max-w-7xl">
        <form 
          method="POST" 
          action="?/filter"
          use:enhance={handleFilterSubmit}
          class="mx-auto px-4 max-w-7xl"
        >
          <div class="mx-auto px-4 max-w-7xl">
            <Search class="mx-auto px-4 max-w-7xl" />
            <input
              type="search"
              name="search"
              placeholder="Search cases..."
              value={data.searchQuery}
              class="mx-auto px-4 max-w-7xl"
            />
          </div>
          
          <div class="mx-auto px-4 max-w-7xl">
            <select
              name="status"
              class="mx-auto px-4 max-w-7xl"
            >
              <option value="all" selected={data.statusFilter === 'all'}>All Status</option>
              <option value="open" selected={data.statusFilter === 'open'}>Open</option>
              <option value="in_progress" selected={data.statusFilter === 'in_progress'}>In Progress</option>
              <option value="closed" selected={data.statusFilter === 'closed'}>Closed</option>
              <option value="archived" selected={data.statusFilter === 'archived'}>Archived</option>
            </select>
            
            <select
              name="priority"
              class="mx-auto px-4 max-w-7xl"
            >
              <option value="all" selected={data.priorityFilter === 'all'}>All Priority</option>
              <option value="low" selected={data.priorityFilter === 'low'}>Low</option>
              <option value="medium" selected={data.priorityFilter === 'medium'}>Medium</option>
              <option value="high" selected={data.priorityFilter === 'high'}>High</option>
              <option value="urgent" selected={data.priorityFilter === 'urgent'}>Urgent</option>
            </select>
          </div>

          <div class="mx-auto px-4 max-w-7xl">
            <select
              name="sort"
              class="mx-auto px-4 max-w-7xl"
            >
              <option value="openedAt" selected={data.sortBy === 'openedAt'}>Date Opened</option>
              <option value="title" selected={data.sortBy === 'title'}>Title</option>
              <option value="status" selected={data.sortBy === 'status'}>Status</option>
              <option value="priority" selected={data.sortBy === 'priority'}>Priority</option>
              <option value="courtDate" selected={data.sortBy === 'courtDate'}>Court Date</option>
            </select>
            
            <button
              type="submit"
              name="order"
              value={data.sortOrder === 'asc' ? 'desc' : 'asc'}
              class="mx-auto px-4 max-w-7xl"
            >
              {#if data.sortOrder === 'asc'}
                <SortAsc class="mx-auto px-4 max-w-7xl" />
              {:else}
                <SortDesc class="mx-auto px-4 max-w-7xl" />
              {/if}
            </button>
          </div>

          <button
            type="submit"
            disabled={$isFiltering}
            class="mx-auto px-4 max-w-7xl"
          >
            {#if $isFiltering}
              <RefreshCw class="mx-auto px-4 max-w-7xl" />
              Filtering...
            {:else}
              <Filter class="mx-auto px-4 max-w-7xl" />
              Apply Filters
            {/if}
          </button>
        </form>
      </div>

      <!-- Case Stats -->
      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">
          {#each data.caseStats as stat}
            <div class="mx-auto px-4 max-w-7xl">
              <span class="mx-auto px-4 max-w-7xl">{stat.status}</span>
              <span class="mx-auto px-4 max-w-7xl">{stat.count}</span>
            </div>
          {/each}
        </div>
      </div>

      <!-- Cases List -->
      <div class="mx-auto px-4 max-w-7xl">
        {#if $isFiltering}
          <div class="mx-auto px-4 max-w-7xl">
            <RefreshCw class="mx-auto px-4 max-w-7xl" />
            Filtering cases...
          </div>
        {:else if data.userCases.length === 0}
          <div class="mx-auto px-4 max-w-7xl">
            <p>No cases found.</p>
            <button
              onclick={() => goto('/cases/new')}
              class="mx-auto px-4 max-w-7xl"
            >
              Create your first case
            </button>
          </div>
        {:else}
          {#each data.userCases as caseItem}
            <CaseListItem 
              caseData={caseItem}
              isActive={caseItem.id === activeCaseId}
              onclick={() => openCase(caseItem.id)}
              on:statusChange={(event) => updateCaseStatus(caseItem.id, event.detail)}
              disabled={$isLoading}
            />
          {/each}
        {/if}
      </div>
    </aside>

    <!-- Main Content Area -->
    <main class="mx-auto px-4 max-w-7xl">
      <slot />
    </main>

    <!-- Right Column: Case Details/Properties (when case is selected) -->
    {#if selectedCase}
      <aside class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">
          <h2 class="mx-auto px-4 max-w-7xl">Case Details</h2>
          <button 
            onclick={() => closeCase()}
            class="mx-auto px-4 max-w-7xl"
            aria-label="Close case details"
          >
            <svg class="mx-auto px-4 max-w-7xl" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
            </svg>
          </button>
        </div>

        <div class="mx-auto px-4 max-w-7xl">
          <div class="mx-auto px-4 max-w-7xl">
            <div>
              <h3 class="mx-auto px-4 max-w-7xl">Case Number</h3>
              <p class="mx-auto px-4 max-w-7xl">{selectedCase?.${1"</p>
            </div>

            <div>
              <h3 class="mx-auto px-4 max-w-7xl">Status</h3>
              <span class="mx-auto px-4 max-w-7xl">
                {selectedCase?.${1"
              </span>
            </div>

            <div>
              <h3 class="mx-auto px-4 max-w-7xl">Priority</h3>
              <span class="mx-auto px-4 max-w-7xl">
                {selectedCase?.${1"
              </span>
            </div>

            <div>
              <h3 class="mx-auto px-4 max-w-7xl">Opened</h3>
              <p class="mx-auto px-4 max-w-7xl">
                {new Date(selectedCase?.${1}).toLocaleDateString()}
              </p>
            </div>

            <div>
              <h3 class="mx-auto px-4 max-w-7xl">Jurisdiction</h3>
              <p class="mx-auto px-4 max-w-7xl">{selectedCase?.${1"</p>
            </div>

            <div>
              <h3 class="mx-auto px-4 max-w-7xl">Description</h3>
              <p class="mx-auto px-4 max-w-7xl">{selectedCase?.${1"</p>
            </div>
          </div>
        </div>

        <div class="mx-auto px-4 max-w-7xl">
          <div class="mx-auto px-4 max-w-7xl">
            <a 
              href="/cases/{selectedCase?.${1"/edit"
              class="mx-auto px-4 max-w-7xl"
            >
              Edit
            </a>
            <button 
              onclick={() => quickAction(selectedCase?.${1}, 'archive')}
              class="mx-auto px-4 max-w-7xl"
            >
              Archive
            </button>
          </div>
        </div>
      </aside>
    {/if}
  </div>
</div>

<style lang="ts">
  /* Ensure smooth transitions */
  :global(.case-list-item) {
    transition: all 0.2s ease;
  }

  :global(.case-list-item:hover) {
    background-color: #f9fafb;
  }

  :global(.case-list-item.active) {
    background-color: #dbeafe;
    border-left: 4px solid #3b82f6;
  }

  /* Custom scrollbar for case list */
  .overflow-y-auto::-webkit-scrollbar {
    width: 6px;
  }
  
  .overflow-y-auto::-webkit-scrollbar-track {
    background: #f1f1f1;
  }
  
  .overflow-y-auto::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 3px;
  }
  
  .overflow-y-auto::-webkit-scrollbar-thumb:hover {
    background: #a8a8a8;
  }
</style>
