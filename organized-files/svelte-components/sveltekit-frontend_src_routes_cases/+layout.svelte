import type { Case } from '$lib/types';




<script lang="ts">
  interface Props {
    data: LayoutData
  }
  let {
    data
  } = $props();



  import { browser } from '$app/environment';
  import { enhance } from '$app/forms';
  import { goto } from '$app/navigation';
  import { page } from '$app/stores';
  import CaseListItem from '$lib/components/cases/CaseListItem.svelte';
  import { casesStore } from "$lib/stores/casesStore";
  import { Filter, Plus, RefreshCw, Search, SortAsc, SortDesc } from 'lucide-svelte';
  import { writable } from 'svelte/store';
  import type { LayoutData } from "./$types";

  
  // Sync server data with client store
  $effect(() => {
    if (browser) {
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
  });
  // Reactive derived stores for UI state
  let activeCaseId = $derived($page.url.searchParams.get('view'));
  let isModalOpen = $derived($page.url.searchParams.has('view'));
  let selectedCase = $derived(data.userCases.find(c => c.id === activeCaseId));

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

<div class="space-y-4">
  <div class="space-y-4">


    <!-- Left Column: CaseList & Filters -->
    <aside class="space-y-4">
      <div class="space-y-4">
        <div class="space-y-4">
          <div>
            <h1 class="space-y-4">Cases</h1>
            <p class="space-y-4">{data.userCases.length} cases</p>
          </div>
          <button
            onclick={() => goto('/cases/new')}
            class="space-y-4"
          >
            <Plus class="space-y-4" />
            New
          </button>
        </div>
      </div>

      <!-- Search & Filters -->
      <div class="space-y-4">
        <form
          method="POST"
          action="?/filter"
          use:enhance={handleFilterSubmit}
          class="space-y-4"
        >
          <div class="space-y-4">
            <Search class="space-y-4" />
            <input
              type="search"
              name="search"
              placeholder="Search cases..."
              value={data.searchQuery}
              class="space-y-4"
            />
          </div>

          <div class="space-y-4">
            <select
              name="status"
              class="space-y-4"
            >
              <option value="all" selected={data.statusFilter === 'all'}>All Status</option>
              <option value="open" selected={data.statusFilter === 'open'}>Open</option>
              <option value="in_progress" selected={data.statusFilter === 'in_progress'}>In Progress</option>
              <option value="closed" selected={data.statusFilter === 'closed'}>Closed</option>
              <option value="archived" selected={data.statusFilter === 'archived'}>Archived</option>
            </select>

            <select
              name="priority"
              class="space-y-4"
            >
              <option value="all" selected={data.priorityFilter === 'all'}>All Priority</option>
              <option value="low" selected={data.priorityFilter === 'low'}>Low</option>
              <option value="medium" selected={data.priorityFilter === 'medium'}>Medium</option>
              <option value="high" selected={data.priorityFilter === 'high'}>High</option>
              <option value="urgent" selected={data.priorityFilter === 'urgent'}>Urgent</option>
            </select>
          </div>

          <div class="space-y-4">
            <select
              name="sort"
              class="space-y-4"
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
              class="space-y-4"
            >
              {#if data.sortOrder === 'asc'}
                <SortAsc class="space-y-4" />
              {:else}
                <SortDesc class="space-y-4" />
              {/if}
            </button>
          </div>

          <button
            type="submit"
            disabled={$isFiltering}
            class="space-y-4"
          >
            {#if $isFiltering}
              <RefreshCw class="space-y-4" />
              Filtering...
            {:else}
              <Filter class="space-y-4" />
              Apply Filters
            {/if}
          </button>
        </form>
      </div>

      <!-- Case Stats -->
      <div class="space-y-4">
        <div class="space-y-4">
          {#each data.caseStats as stat}
            <div class="space-y-4">
              <span class="space-y-4">{stat.status}</span>
              <span class="space-y-4">{stat.count}</span>
            </div>
          {/each}
        </div>
      </div>

      <!-- Cases List -->
      <div class="space-y-4">
        {#if $isFiltering}
          <div class="space-y-4">
            <RefreshCw class="space-y-4" />
            Filtering cases...
          </div>
        {:else if data.userCases.length === 0}
          <div class="space-y-4">
            <p>No cases found.</p>
            <button
              onclick={() => goto('/cases/new')}
              class="space-y-4"
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
    <main class="space-y-4">
      <slot></slot>
    </main>


    <!-- Right Column: CaseDetails/Properties (when case is selected) -->
    {#if selectedCase}
      <aside class="space-y-4">
        <div class="space-y-4">
          <h2 class="space-y-4">Case Details</h2>
          <button
            onclick={() => closeCase()}
            class="space-y-4"
            aria-label="Close case details"
          >
            <svg class="space-y-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
            </svg>
          </button>
        </div>

        <div class="space-y-4">
          <div class="space-y-4">
            <div>
              <h3 class="space-y-4">Case Number</h3>
              <p class="space-y-4">{selectedCase?.caseNumber}</p>
            </div>

            <div>
              <h3 class="space-y-4">Status</h3>
              <span class="space-y-4">
                {selectedCase?.status}
              </span>
            </div>

            <div>
              <h3 class="space-y-4">Priority</h3>
              <span class="space-y-4">
                {selectedCase?.priority}
              </span>
            </div>

            <div>
              <h3 class="space-y-4">Opened</h3>
              <p class="space-y-4">
                {selectedCase?.createdAt ? new Date(selectedCase.createdAt).toLocaleDateString() : ''}
              </p>
            </div>

            <div>
              <!-- Jurisdiction property not found in Case type, remove or update if needed -->
              <!-- <h3 class="space-y-4">Jurisdiction</h3> -->
              <!-- <p class="space-y-4">{selectedCase?.jurisdiction}</p> -->
            </div>

            <div>
              <h3 class="space-y-4">Description</h3>
              <p class="space-y-4">{selectedCase?.description}</p>
            </div>
          </div>
        </div>

        <div class="space-y-4">
          <div class="space-y-4">
            <a
              href={`/cases/${selectedCase?.id}/edit`}
              class="space-y-4"
            >
              Edit
            </a>
            <button
              onclick={() => quickAction(selectedCase?.id, 'archive')}
              class="space-y-4"
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
  /* Removed unused .overflow-y-auto selectors to resolve Svelte warnings */
</style>
