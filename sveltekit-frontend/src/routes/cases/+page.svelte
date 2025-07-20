<script lang="ts">
  import type { Evidence } from '$lib/types';
  import { browser } from '$app/environment';
  import { enhance } from '$app/forms';
  import { goto } from '$app/navigation';
  import { page } from '$app/stores';
  import EvidenceCard from '$lib/components/cases/EvidenceCard.svelte';
  import { Badge, Button, Input, Modal } from "$lib/components/ui/index";
  import { Textarea } from "$lib/components/ui/textarea/index";
  import { notifications } from "$lib/stores/notification";
  import { formatDistanceToNow } from 'date-fns';
  import { Archive, Calendar, Download, Edit, FileText, Plus, Scale, Search, X } from "lucide-svelte";
  import { writable } from 'svelte/store';
  import type { PageData } from "./$types";

  export let data: PageData;

  // Reactive state
  $: activeCase = data.activeCase;
  $: caseEvidence = data.caseEvidence;
  $: isModalOpen = !!activeCase;

  // Local state
  const isLoading = writable(false);
  const evidenceFilter = writable('');
  const isAddingEvidence = writable(false);
  const editingEvidence = writable<any>(null);

  // Additional state management
  let showNewCaseModal = false;
  let retryCount = 0;
  let maxRetries = 3;
  let autoRefreshInterval: NodeJS.Timeout | null = null;
  let lastFetch = Date.now();

  // Pagination
  let currentPage = 1;
  let itemsPerPage = 12;
  let totalPages = 1;

  // Bulk operations
  let bulkOperationLoading = false;

  // Evidence form data
  let evidenceForm = {
    title: '',
    description: '',
    type: 'document'
  };

  // Accessibility
  let focusManager = {
    setFocus: (selector: string) => {
      if (browser) {
        const element = document.querySelector(selector);
        if (element instanceof HTMLElement) {
          element.focus();
}
}
}
  };

  // Statistics
  let caseStats = {
    total: 0,
    active: 0,
    closed: 0,
    urgent: 0,
    overdue: 0,
  };

  // Status options for filtering and creation
  const statusOptions = [
    { value: "open", label: "Open", color: "bg-green-100 text-green-800" },
    { value: "in_progress", label: "In Progress", color: "bg-blue-100 text-blue-800" },
    { value: "pending", label: "Pending", color: "bg-yellow-100 text-yellow-800" },
    { value: "closed", label: "Closed", color: "bg-gray-100 text-gray-800" },
    { value: "archived", label: "Archived", color: "bg-gray-100 text-gray-500" },
  ];

  // Close modal
  function closeModal() {
    const url = new URL($page.url);
    url.searchParams.delete('view');
    goto(url.toString(), { keepFocus: true, noScroll: true });
}
  // Handle evidence form submission
  const handleEvidenceSubmit = () => {
    isLoading.set(true);
    return async ({ result, update }) => {
      if (result.type === 'success') {
        evidenceForm = { title: '', description: '', type: 'document' };
        isAddingEvidence.set(false);
        editingEvidence.set(null);
        notifications.add({
          type: 'success',
          title: 'Success',
          message: 'Evidence saved successfully'
        });
        // Refresh page data
        goto($page.url.pathname + $page.url.search, { invalidateAll: true });
      } else if (result.type === 'failure') {
        notifications.add({
          type: 'error',
          title: 'Error',
          message: result.data?.message || 'Failed to save evidence'
        });
}
      await update();
      isLoading.set(false);
    };
  };

  // Handle evidence deletion
  async function deleteEvidence(evidenceId: string) {
    if (!confirm('Are you sure you want to delete this evidence?')) return;

    isLoading.set(true);
    try {
      const formData = new FormData();
      formData.append('evidenceId', evidenceId);

      const response = await fetch('?/deleteEvidence', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        notifications.add({
          type: 'success',
          title: 'Success',
          message: 'Evidence deleted successfully'
        });
        // Refresh page data
        goto($page.url.pathname + $page.url.search, { invalidateAll: true });
      } else {
        notifications.add({
          type: 'error',
          title: 'Error',
          message: 'Failed to delete evidence'
        });
}
    } catch (error) {
      notifications.add({
        type: 'error',
        title: 'Error',
        message: 'Failed to delete evidence'
      });
    } finally {
      isLoading.set(false);
}
}
  // Start editing evidence
  function startEditEvidence(evidence: any) {
    evidenceForm = {
      title: evidence.title,
      description: evidence.description || '',
      type: evidence.evidenceType || evidence.type
    };
    editingEvidence.set(evidence);
    isAddingEvidence.set(true);
}
  // Filter evidence
  $: filteredEvidence = caseEvidence.filter(e =>
    !$evidenceFilter ||
    e.title.toLowerCase().includes($evidenceFilter.toLowerCase()) ||
    e.description?.toLowerCase().includes($evidenceFilter.toLowerCase())
  );

  // Handle keyboard shortcuts
  function handleKeydown(event: KeyboardEvent) {
    if (event.key === 'Escape') {
      if ($isAddingEvidence) {
        isAddingEvidence.set(false);
        editingEvidence.set(null);
      } else if (isModalOpen) {
        closeModal();
}
}
}
  // Case status colors
  function getStatusColor(status: string) {
    switch (status) {
      case 'open': return 'bg-green-100 text-green-800';
      case 'in_progress': return 'bg-yellow-100 text-yellow-800';
      case 'closed': return 'bg-blue-100 text-blue-800';
      case 'archived': return 'bg-gray-100 text-gray-800';
      default: return 'bg-gray-100 text-gray-800';
}
}
  function getPriorityColor(priority: string) {
    switch (priority) {
      case 'low': return 'bg-green-100 text-green-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'high': return 'bg-orange-100 text-orange-800';
      case 'urgent': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
}
}
</script>

<svelte:window on:keydown={handleKeydown} />

{#if !isModalOpen}
  <!-- Empty State: No case selected -->
  <div class="container mx-auto px-4">
    <div class="container mx-auto px-4">
      <Scale class="container mx-auto px-4" />
      <h2 class="container mx-auto px-4">No Case Selected</h2>
      <p class="container mx-auto px-4">Select a case from the sidebar to view details and manage evidence</p>
      <Button on:click={() => goto('/cases/new')} class="container mx-auto px-4">
        <Plus class="container mx-auto px-4" />
        Create New Case
      </Button>
    </div>
  </div>
{:else}
  <!-- Case Details View -->
  <div class="container mx-auto px-4">
import type { Case } from '$lib/types';

    <!-- Center Column: CaseDetails -->
    <div class="container mx-auto px-4">
      <!-- Case Header -->
      <div class="container mx-auto px-4">
        <div class="container mx-auto px-4">
          <div class="container mx-auto px-4">
            <Scale class="container mx-auto px-4" />
            <div>
              <h1 class="container mx-auto px-4">{activeCase?.title}</h1>
              <p class="container mx-auto px-4">Case #{activeCase?.caseNumber}</p>
            </div>
          </div>
          <div class="container mx-auto px-4">
            <Badge class={getStatusColor(activeCase?.status)}>
              {activeCase?.status.replace('_', ' ')}
            </Badge>
            <Badge class={getPriorityColor(activeCase?.priority)}>
              {activeCase?.priority}
            </Badge>
            <Button variant="ghost" size="sm" on:click={() => closeModal()}>
              <X class="container mx-auto px-4" />
            </Button>
          </div>
        </div>
      </div>

      <!-- Case Information -->
      <div class="container mx-auto px-4">
        <div class="container mx-auto px-4">
          <div class="container mx-auto px-4">
            <Calendar class="container mx-auto px-4" />
            <span>Opened {activeCase?.createdAt ? formatDistanceToNow(new Date(activeCase.createdAt), { addSuffix: true }) : ''}</span>
          </div>
          <!-- User, Jurisdiction, and Court Date fields not present in Case type. Add if needed. -->
        </div>
      </div>

      <!-- Case Description -->
      {#if activeCase?.description}
        <div class="container mx-auto px-4">
          <h3 class="container mx-auto px-4">Description</h3>
          <p class="container mx-auto px-4">{activeCase.description}</p>
        </div>
      {/if}

      <!-- Case Actions -->
      <div class="container mx-auto px-4">
        <div class="container mx-auto px-4">
          <Button variant="outline" size="sm">
            <Edit class="container mx-auto px-4" />
            Edit Case
          </Button>
          <Button variant="outline" size="sm">
            <Download class="container mx-auto px-4" />
            Export
          </Button>
          <Button variant="outline" size="sm">
            <Archive class="container mx-auto px-4" />
            Archive
          </Button>
        </div>
      </div>

      <!-- Evidence Summary -->
      <div class="container mx-auto px-4">
        <h3 class="container mx-auto px-4">Evidence Summary</h3>

        {#if caseEvidence.length === 0}
          <div class="container mx-auto px-4">
            <FileText class="container mx-auto px-4" />
            <p class="container mx-auto px-4">No evidence added yet</p>
          </div>
        {:else}
          <div class="container mx-auto px-4">
            {#each caseEvidence.slice(0, 5) as evidence}
              <div class="container mx-auto px-4">
                <div class="container mx-auto px-4">
                  <FileText class="container mx-auto px-4" />
                  <div>
                    <p class="container mx-auto px-4">{evidence.title}</p>
                    <p class="container mx-auto px-4">{evidence.evidenceType || evidence.type}</p>
                  </div>
                </div>
                <div class="container mx-auto px-4">
                  {evidence.uploadedAt ? formatDistanceToNow(new Date(evidence.uploadedAt), { addSuffix: true }) : ''}
                </div>
              </div>
            {/each}

            {#if caseEvidence.length > 5}
              <p class="container mx-auto px-4">
                and {caseEvidence.length - 5} more evidence items
              </p>
            {/if}
          </div>
        {/if}
      </div>
    </div>

    <!-- Right Column: Evidence Management -->
    <div class="container mx-auto px-4">
      <!-- Evidence Header -->
      <div class="container mx-auto px-4">
        <div class="container mx-auto px-4">
          <h2 class="container mx-auto px-4">Evidence</h2>
          <Button
            size="sm"
            on:click={() => isAddingEvidence.set(true)}
            disabled={$isLoading}
          >
            <Plus class="container mx-auto px-4" />
            Add
          </Button>
        </div>
      </div>

      <!-- Evidence Filter -->
      <div class="container mx-auto px-4">
        <div class="container mx-auto px-4">
          <Search class="container mx-auto px-4" />
          <Input
            type="search"
            placeholder="Search evidence..."
            bind:value={$evidenceFilter}
            class="container mx-auto px-4"
          />
        </div>
      </div>

      <!-- Evidence List -->
      <div class="container mx-auto px-4">
        {#if filteredEvidence.length === 0}
          <div class="container mx-auto px-4">
            <FileText class="container mx-auto px-4" />
            <p>No evidence found</p>
          </div>
        {:else}
          <div class="container mx-auto px-4">
            {#each filteredEvidence as evidence}
              <EvidenceCard
                {evidence}
                on:edit={() => startEditEvidence(evidence)}
                on:delete={() => deleteEvidence(evidence.id)}
                disabled={$isLoading}
              />
            {/each}
          </div>
        {/if}
      </div>
    </div>
  </div>

  <!-- Add/Edit Evidence Modal -->
  {#if $isAddingEvidence}
    <Modal
      open={$isAddingEvidence}
      on:close={() => {
        isAddingEvidence.set(false);
        editingEvidence.set(null);
      }}
    >
      <div class="container mx-auto px-4">
        <h2 class="container mx-auto px-4">
          {$editingEvidence ? 'Edit Evidence' : 'Add Evidence'}
        </h2>

        <form
          method="POST"
          action={$editingEvidence ? '?/updateEvidence' : '?/addEvidence'}
          use:enhance={handleEvidenceSubmit}
          class="container mx-auto px-4"
        >
          <input type="hidden" name="caseId" value={activeCase?.id} />
          {#if $editingEvidence}
            <input type="hidden" name="evidenceId" value={$editingEvidence.id} />
          {/if}

          <div>
            <label for="title" class="container mx-auto px-4">
              Title *
            </label>
            <Input
              id="title"
              name="title"
              bind:value={evidenceForm.title}
              placeholder="Evidence title"
              required
            />
          </div>

          <div>
            <label for="type" class="container mx-auto px-4">
              Type *
            </label>
            <select
              id="type"
              name="type"
              bind:value={evidenceForm.type}
              class="container mx-auto px-4"
              required
            >
              <option value="document">Document</option>
              <option value="photo">Photo</option>
              <option value="video">Video</option>
              <option value="audio">Audio</option>
              <option value="physical">Physical Evidence</option>
              <option value="digital">Digital Evidence</option>
              <option value="testimony">Testimony</option>
              <option value="other">Other</option>
            </select>
          </div>

          <div>
            <label for="description" class="container mx-auto px-4">
              Description
            </label>
            <Textarea
              id="description"
              name="description"
              bind:value={evidenceForm.description}
              placeholder="Evidence description"
              rows={4}
            />
          </div>

          <div class="container mx-auto px-4">
            <Button
              type="submit"
              disabled={$isLoading}
              class="container mx-auto px-4"
            >
              {$isLoading ? 'Saving...' : ($editingEvidence ? 'Update' : 'Add')} Evidence
            </Button>
            <Button
              type="button"
              variant="outline"
              on:click={() => {
                isAddingEvidence.set(false);
                editingEvidence.set(null);
              }}
            >
              Cancel
            </Button>
          </div>
        </form>
      </div>
    </Modal>
  {/if}
{/if}

<style lang="ts">
  :global(.modal-overlay) {
    background-color: rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(2px);
}
  .case-card {
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
  .case-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}
  /* High contrast mode support */
  @media (prefers-contrast: high) {
    .case-card {
      border: 2px solid;
}
    .case-card:hover,
    .case-card:focus-within {
      border-width: 3px;
}
}
  /* Reduced motion support */
  @media (prefers-reduced-motion: reduce) {
    .case-card,
    .animate-spin {
      animation: none;
      transition: none;
}
    .case-card:hover {
      transform: none;
}
}
  /* Print styles */
  @media print {
    .case-grid {
      display: block !important;
}
    .case-card {
      break-inside: avoid;
      margin-bottom: 1rem;
      box-shadow: none;
      border: 1px solid #000;
}
}
  /* Mobile optimizations */
  @media (max-width: 640px) {
    .case-card {
      padding: 1rem;
}
}
</style>
