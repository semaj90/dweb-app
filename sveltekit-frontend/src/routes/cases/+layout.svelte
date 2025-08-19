import type { Case } from '$lib/types';




<script lang="ts">
  interface Props {
    data: LayoutData;
  }
  let {
    data
  }: Props = $props();



  import { browser } from '$app/environment';
  import { enhance } from '$app/forms';
  import { goto } from '$app/navigation';
  import { page } from '$app/state';
  import CaseListItem from '$lib/components/cases/CaseListItem.svelte';
  import { casesStore } from "$lib/stores/casesStore";
  import { Filter, Plus, RefreshCw, Search, SortAsc, SortDesc } from 'lucide-svelte';
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
  let isLoading = $state(false);
  let isFiltering = $state(false);

  // Enhanced form submission with loading state and partial updates
  const handleFilterSubmit = () => {
    isFiltering = true;
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
      isFiltering = false;
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
    isLoading = true;
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
      isLoading = false;
    }
}
  // Handle quick status update
  async function updateCaseStatus(caseId: string, status: string) {
    isLoading = true;
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
      isLoading = false;
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

<div class="flex min-h-screen bg-gray-50">
  <div class="flex w-full max-w-7xl mx-auto gap-6 p-6">

    <!-- Left Column: CaseList & Filters -->
    <aside class="w-80 flex-shrink-0 space-y-6">
      <div class="bg-white rounded-lg shadow-sm p-6">
        <div class="flex items-center justify-between mb-4">
          <div>
            <h1 class="text-2xl font-bold text-gray-900">Cases</h1>
            <p class="text-sm text-gray-600">{data.userCases.length} cases</p>
          </div>
          <button
            onclick={() => goto('/cases/new')}
            class="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <Plus class="w-4 h-4 mr-2" />
            New
          </button>
        </div>
      </div>

      <!-- Search & Filters -->
      <div class="bg-white rounded-lg shadow-sm p-6">
        <form
          method="POST"
          action="?/filter"
          use:enhance={handleFilterSubmit}
          class="space-y-4"
        >
          <div class="relative">
            <Search class="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="search"
              name="search"
              placeholder="Search cases..."
              value={data.searchQuery}
              class="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          <div class="grid grid-cols-2 gap-4">
            <select
              name="status"
              class="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="all" selected={data.statusFilter === 'all'}>All Status</option>
              <option value="open" selected={data.statusFilter === 'open'}>Open</option>
              <option value="in_progress" selected={data.statusFilter === 'in_progress'}>In Progress</option>
              <option value="closed" selected={data.statusFilter === 'closed'}>Closed</option>
              <option value="archived" selected={data.statusFilter === 'archived'}>Archived</option>
            </select>

            <select
              name="priority"
              class="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="all" selected={data.priorityFilter === 'all'}>All Priority</option>
              <option value="low" selected={data.priorityFilter === 'low'}>Low</option>
              <option value="medium" selected={data.priorityFilter === 'medium'}>Medium</option>
              <option value="high" selected={data.priorityFilter === 'high'}>High</option>
              <option value="urgent" selected={data.priorityFilter === 'urgent'}>Urgent</option>
            </select>
          </div>

          <div class="flex gap-2">
            <select
              name="sort"
              class="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
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
              class="p-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
            >
              {#if data.sortOrder === 'asc'}
                <SortAsc class="w-4 h-4" />
              {:else}
                <SortDesc class="w-4 h-4" />
              {/if}
            </button>
          </div>

          <button
            type="submit"
            disabled={isFiltering}
            class="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            {#if isFiltering}
              <RefreshCw class="inline w-4 h-4 mr-2 animate-spin" />
              Filtering...
            {:else}
              <Filter class="inline w-4 h-4 mr-2" />
              Apply Filters
            {/if}
          </button>
        </form>
      </div>

      <!-- Case Stats -->
      <div class="bg-white rounded-lg shadow-sm p-6">
        <h3 class="text-lg font-semibold text-gray-900 mb-4">Case Statistics</h3>
        <div class="space-y-2">
          {#each data.caseStats as stat}
            <div class="flex justify-between items-center py-2 border-b border-gray-100 last:border-0">
              <span class="text-sm font-medium text-gray-600 capitalize">{stat.status}</span>
              <span class="text-sm font-bold text-gray-900">{stat.count}</span>
            </div>
          {/each}
        </div>
      </div>

      <!-- Cases List -->
      <div class="bg-white rounded-lg shadow-sm">
        {#if isFiltering}
          <div class="flex items-center justify-center p-8">
            <RefreshCw class="w-5 h-5 animate-spin mr-3" />
            <span class="text-gray-600">Filtering cases...</span>
          </div>
        {:else if data.userCases.length === 0}
          <div class="text-center p-8">
            <p class="text-gray-500 mb-4">No cases found.</p>
            <button
              onclick={() => goto('/cases/new')}
              class="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              Create your first case
            </button>
          </div>
        {:else}
          <div class="divide-y divide-gray-100">
            {#each data.userCases as caseItem}
              <CaseListItem
                caseData={caseItem}
                isActive={caseItem.id === activeCaseId}
                on:click={() => openCase(caseItem.id)}
                onstatuschange={(event) => updateCaseStatus(caseItem.id, event.detail)}
                disabled={isLoading}
              />
            {/each}
          </div>
        {/if}
      </div>
    </aside>

    <!-- Main Content Area -->
    <main class="flex-1 bg-white rounded-lg shadow-sm">
      {@render children?.()}
    </main>

    <!-- Right Column: CaseDetails/Properties (when case is selected) -->
    {#if selectedCase}
      <aside class="w-80 flex-shrink-0 bg-white rounded-lg shadow-sm p-6">
        <div class="flex items-center justify-between mb-6">
          <h2 class="text-lg font-semibold text-gray-900">Case Details</h2>
          <button
            onclick={() => closeCase()}
            class="p-1 text-gray-400 hover:text-gray-600 transition-colors"
            aria-label="Close case details"
          >
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
            </svg>
          </button>
        </div>

        <div class="space-y-4">
          <div class="space-y-4">
            <div>
              <h3 class="text-sm font-medium text-gray-500">Case Number</h3>
              <p class="text-sm font-semibold text-gray-900">{selectedCase?.caseNumber}</p>
            </div>

            <div>
              <h3 class="text-sm font-medium text-gray-500">Status</h3>
              <span class="inline-flex items-center px-2 py-1 text-xs font-medium rounded-full bg-gray-100 text-gray-800 capitalize">
                {selectedCase?.status}
              </span>
            </div>

            <div>
              <h3 class="text-sm font-medium text-gray-500">Priority</h3>
              <span class="inline-flex items-center px-2 py-1 text-xs font-medium rounded-full bg-gray-100 text-gray-800 capitalize">
                {selectedCase?.priority}
              </span>
            </div>

            <div>
              <h3 class="text-sm font-medium text-gray-500">Opened</h3>
              <p class="text-sm text-gray-900">
                {selectedCase?.createdAt ? new Date(selectedCase.createdAt).toLocaleDateString() : ''}
              </p>
            </div>

            <div>
              <h3 class="text-sm font-medium text-gray-500">Description</h3>
              <p class="text-sm text-gray-900">{selectedCase?.description}</p>
            </div>
          </div>
        </div>

        <div class="mt-6 pt-6 border-t border-gray-100">
          <div class="flex gap-2">
            <a
              href={`/cases/${selectedCase?.id}/edit`}
              class="flex-1 px-3 py-2 text-center text-sm font-medium text-blue-600 border border-blue-600 rounded-lg hover:bg-blue-50 transition-colors"
            >
              Edit
            </a>
            <button
              onclick={() => quickAction(selectedCase?.id, 'archive')}
              class="flex-1 px-3 py-2 text-sm font-medium text-gray-600 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
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
