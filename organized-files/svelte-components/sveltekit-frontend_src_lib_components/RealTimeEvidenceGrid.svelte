<!-- Real-time Evidence Grid with WebSocket and local sync -->
<script lang="ts">
  interface Props {
    caseId: string | undefined ;
    searchQuery: string
    selectedTypes: string[] ;
  }
  let {
    caseId = undefined,
    searchQuery = "",
    selectedTypes = []
  } = $props();



  import type { Evidence } from '$lib/types';
  import Button from "$lib/components/ui/button";
  import { evidenceStore } from "$lib/stores/evidenceStore";
  import { lokiEvidenceService } from "$lib/utils/loki-evidence";
  import {
    Archive,
    Calendar,
    Eye,
    File,
    FileText,
    Grid,
    Image,
    List,
    Music,
    Redo2,
    RefreshCw,
    Search,
    SortAsc,
    SortDesc,
    Tag,
    Trash2,
    Undo2,
    Video,
    Wifi,
    WifiOff,
  } from "lucide-svelte";
  import { onMount } from "svelte";

  // Props
        export const showAdvancedFilters: boolean = false;

  // Store subscriptions
  let evidence = $derived($evidenceStore.evidence || [])
  let isLoading = $derived($evidenceStore.isLoading || false)
  let isConnected = $derived($evidenceStore.isConnected || false)
  let error = $derived($evidenceStore.error || null)

  // Reactive filtering and sorting
  $effect(() => { {
    filteredEvidence = evidence
      .filter((item) => {
        if (caseId && item.caseId !== caseId) return false;

        if (searchQuery) {
          const query = searchQuery.toLowerCase();
          const searchableText = [
            item.title,
            item.description,
            item.type,
            ...(item.tags || []),
          ]
            .join(" ")
            .toLowerCase();

          if (!searchableText.includes(query)) return false;
}
        if (selectedTypes.length > 0 && !selectedTypes.includes(item.type)) {
          return false;
}
        return true;
      })
      .sort((a, b) => {
        let aVal: any, bVal: any

        switch (sortBy) {
          case "date":
            aVal = new Date(a.timeline?.updatedAt || 0);
            bVal = new Date(b.timeline?.updatedAt || 0);
            break;
          case "title":
            aVal = a.title.toLowerCase();
            bVal = b.title.toLowerCase();
            break;
          case "type":
            aVal = a.type.toLowerCase();
            bVal = b.type.toLowerCase();
            break;
          case "relevance":
            aVal = a.classification?.relevance || 0;
            bVal = b.classification?.relevance || 0;
            break;
          default:
            return 0;
}
        const comparison = aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
        return sortOrder === "asc" ? comparison : -comparison;
      });

    totalPages = Math.ceil(filteredEvidence.length / pageSize);
    paginatedEvidence = filteredEvidence.slice(
      currentPage * pageSize,
      (currentPage + 1) * pageSize
    );
}
  // Initialize on mount
  onMount(() => {
    const init = async () => {
      await initializeRealTimeEvidence();
    };
    init();

    // Update connection status
    const unsubscribe = evidenceStore.isConnected.subscribe((connected) => {
      connectionStatus = connected ? "connected" : "disconnected";
      if (connected) {
        lastUpdateTime = new Date().toISOString();
}
    });

    // Monitor sync status
    const syncInterval = setInterval(updateSyncStatus, 2000);

    return () => {
      unsubscribe();
      clearInterval(syncInterval);
      evidenceStore.disconnect();
    };
  });

  async function initializeRealTimeEvidence() {
    try {
      // Load from local cache first
      if (lokiEvidenceService.isReady()) {
        loadFromLocal();
      } else {
        // Wait for Loki to initialize
        setTimeout(() => {
          if (lokiEvidenceService.isReady()) {
            loadFromLocal();
}
        }, 1000);
}
      // Then sync with server
      await syncWithServer();
    } catch (err) {
      console.error("Failed to initialize real-time evidence:", err);
      error = err instanceof Error ? err.message : "Initialization failed";
}
}
  function loadFromLocal() {
    try {
      const localEvidence = caseId
        ? lokiEvidenceService.getEvidenceByCase(caseId)
        : lokiEvidenceService.getAllEvidence();

      evidenceStore.evidence.set(localEvidence);
    } catch (err) {
      console.error("Failed to load from local:", err);
}
}
  async function syncWithServer() {
    try {
      evidenceStore.isLoading.set(true);

      const endpoint = caseId
        ? `/api/evidence?caseId=${caseId}`
        : "/api/evidence";
      const response = await fetch(endpoint);

      if (response.ok) {
        const serverEvidence = await response.json();
        await lokiEvidenceService.syncWithServer(serverEvidence);
        evidenceStore.evidence.set(serverEvidence);
        lastUpdateTime = new Date().toISOString();
}
    } catch (err) {
      console.error("Sync failed:", err);
      error = "Failed to sync with server";
    } finally {
      evidenceStore.isLoading.set(false);
}
}
  function updateSyncStatus() {
    if (lokiEvidenceService.isReady()) {
      syncStatus = lokiEvidenceService.getSyncStatus();
}
}
  // Evidence operations
  async function createEvidence() {
    try {
      const newEvidence = {
        title: "New Evidence",
        description: "",
        type: "document",
        caseId: caseId || "default-case",
        tags: [],
      };

      const evidenceId = await evidenceStore.createEvidence(newEvidence);
      editingEvidence = evidenceId;
    } catch (err) {
      console.error("Failed to create evidence:", err);
      error = err instanceof Error ? err.message : "Failed to create evidence";
}
}
  async function updateEvidence(
    evidenceId: string,
    changes: Partial<Evidence>
  ) {
    try {
      await evidenceStore.updateEvidence(evidenceId, changes);
      editingEvidence = null;
    } catch (err) {
      console.error("Failed to update evidence:", err);
      error = err instanceof Error ? err.message : "Failed to update evidence";
}
}
  async function deleteEvidence(evidenceId: string) {
    if (!confirm("Are you sure you want to delete this evidence?")) return;

    try {
      await evidenceStore.deleteEvidence(evidenceId);
      selectedEvidence.delete(evidenceId);
      selectedEvidence = selectedEvidence;
    } catch (err) {
      console.error("Failed to delete evidence:", err);
      error = err instanceof Error ? err.message : "Failed to delete evidence";
}
}
  // UI interactions
  function toggleSelection(evidenceId: string) {
    if (selectedEvidence.has(evidenceId)) {
      selectedEvidence.delete(evidenceId);
    } else {
      selectedEvidence.add(evidenceId);
}
    selectedEvidence = selectedEvidence;
}
  function selectAll() {
    selectedEvidence = new Set(paginatedEvidence.map((e) => e.id));
}
  function clearSelection() {
    selectedEvidence.clear();
    selectedEvidence = selectedEvidence;
}
  function getTypeIcon(type: string) {
    switch (type) {
      case "document":
        return FileText;
      case "image":
        return Image;
      case "video":
        return Video;
      case "audio":
        return Music;
      default:
        return File;
}
}
  function formatDate(dateString: string): string {
    return new Date(dateString).toLocaleDateString();
}
  function getRelevanceColor(relevance: number): string {
    if (relevance >= 0.8) return "text-green-600";
    if (relevance >= 0.6) return "text-yellow-600";
    if (relevance >= 0.4) return "text-orange-600";
    return "text-red-600";
}
</script>

<!-- Connection Status Bar -->
<div
  class="space-y-4"
>
  <div class="space-y-4">
    <!-- Connection Status -->
    <div class="space-y-4">
      {#if connectionStatus === "connected"}
        <Wifi class="space-y-4" />
        <span class="space-y-4">Connected</span>
      {:else}
        <WifiOff class="space-y-4" />
        <span class="space-y-4">Offline</span>
      {/if}
    </div>

    <!-- Sync Status -->
    {#if syncStatus.pending > 0}
      <div class="space-y-4">
        <RefreshCw class="space-y-4" />
        <span>Syncing ({syncStatus.pending} pending)</span>
      </div>
    {/if}

    {#if syncStatus.failed > 0}
      <span class="space-y-4">{syncStatus.failed} failed</span>
    {/if}

    <!-- Stats -->
    <span class="space-y-4">Total: {filteredEvidence.length}</span>

    {#if selectedEvidence.size > 0}
      <span class="space-y-4">Selected: {selectedEvidence.size}</span>
    {/if}

    {#if lastUpdateTime}
      <span class="space-y-4">Updated: {formatDate(lastUpdateTime)}</span>
    {/if}
  </div>

  <!-- Action Buttons -->
  <div class="space-y-4">
    <Button
      variant="ghost"
      size="sm"
      onclick={() => evidenceStore.undo()}
      disabled={!evidenceStore.canUndo()}
      title="Undo (Ctrl+Z)"
    >
      <Undo2 class="space-y-4" />
    </Button>

    <Button
      variant="ghost"
      size="sm"
      onclick={() => evidenceStore.redo()}
      disabled={!evidenceStore.canRedo()}
      title="Redo (Ctrl+Y)"
    >
      <Redo2 class="space-y-4" />
    </Button>

    <Button
      variant="ghost"
      size="sm"
      onclick={() => syncWithServer()}
      disabled={isLoading}
      title="Sync with server"
    >
      <RefreshCw class="space-y-4" />
    </Button>
  </div>
</div>

<!-- Error Banner -->
{#if error}
  <div class="space-y-4">
    <div class="space-y-4">
      <div class="space-y-4">
        <p class="space-y-4">{error}</p>
      </div>
      <div class="space-y-4">
        <button
          class="space-y-4"
          onclick={() => (error = null)}
        >
          <span class="space-y-4">Dismiss</span>
          ✕
        </button>
      </div>
    </div>
  </div>
{/if}

<!-- Toolbar -->
<div class="space-y-4">
  <div class="space-y-4">
    <!-- Left: Search and Filters -->
    <div class="space-y-4">
      <!-- Search -->
      <div class="space-y-4">
        <Search
          class="space-y-4"
        />
        <input
          type="text"
          placeholder="Search evidence..."
          bind:value={searchQuery}
          class="space-y-4"
        />
      </div>

      <!-- Type Filter -->
      <select
        multiple
        bind:value={selectedTypes}
        class="space-y-4"
      >
        <option value="">All Types</option>
        <option value="document">Documents</option>
        <option value="image">Images</option>
        <option value="video">Videos</option>
        <option value="audio">Audio</option>
        <option value="testimony">Testimony</option>
        <option value="physical">Physical</option>
        <option value="digital">Digital</option>
      </select>

      <!-- Sort -->
      <div class="space-y-4">
        <select
          bind:value={sortBy}
          class="space-y-4"
        >
          <option value="date">Date</option>
          <option value="title">Title</option>
          <option value="type">Type</option>
          <option value="relevance">Relevance</option>
        </select>

        <Button
          variant="ghost"
          size="sm"
          onclick={() => (sortOrder = sortOrder === "asc" ? "desc" : "asc")}
        >
          {#if sortOrder === "asc"}
            <SortAsc class="space-y-4" />
          {:else}
            <SortDesc class="space-y-4" />
          {/if}
        </Button>
      </div>
    </div>

    <!-- Right: View and Actions -->
    <div class="space-y-4">
      <!-- View Mode Toggle -->
      <Button
        variant="ghost"
        size="sm"
        onclick={() => (viewMode = viewMode === "grid" ? "list" : "grid")}
      >
        {#if viewMode === "grid"}
          <List class="space-y-4" />
        {:else}
          <Grid class="space-y-4" />
        {/if}
      </Button>

      <!-- Selection Actions -->
      {#if selectedEvidence.size > 0}
        <Button variant="outline" size="sm" onclick={() => clearSelection()}>
          Clear ({selectedEvidence.size})
        </Button>

        <Button
          variant="danger"
          size="sm"
          onclick={() => {
            if (confirm(`Delete ${selectedEvidence.size} selected items?`)) {
              selectedEvidence.forEach((id) => deleteEvidence(id));
            }
          }
        >
          <Trash2 class="space-y-4" />
          Delete
        </Button>
      {:else}
        <Button variant="ghost" size="sm" onclick={() => selectAll()}>
          Select All
        </Button>
      {/if}

      <!-- Add Evidence -->
      <Button onclick={() => createEvidence()}>
        <span class="space-y-4">+</span>
        Add Evidence
      </Button>
    </div>
  </div>
</div>

<!-- Content Area -->
<div class="space-y-4">
  {#if isLoading && evidence.length === 0}
    <!-- Loading State -->
    <div class="space-y-4">
      <div class="space-y-4">
        <RefreshCw class="space-y-4" />
        <p class="space-y-4">Loading evidence...</p>
      </div>
    </div>
  {:else if paginatedEvidence.length === 0}
    <!-- Empty State -->
    <div class="space-y-4">
      <div class="space-y-4">
        <Archive class="space-y-4" />
        <h3 class="space-y-4">
          No Evidence Found
        </h3>
        <p class="space-y-4">
          {filteredEvidence.length === 0 && evidence.length > 0
            ? "No evidence matches your current filters."
            : "No evidence has been added yet."}
        </p>
        <Button onclick={() => createEvidence()}>Add First Evidence</Button>
      </div>
    </div>
  {:else}
    <!-- Evidence Grid/List -->
    {#if viewMode === "grid"}
      <div
        class="space-y-4"
      >
        {#each paginatedEvidence as item (item.id)}
          <div
            class="space-y-4"
          >
            <!-- Header -->
            <div class="space-y-4">
              <div class="space-y-4">
                <input
                  type="checkbox"
                  checked={selectedEvidence.has(item.id)}
                  onchange={() => toggleSelection(item.id)}
                  class="space-y-4"
                />
                <svelte:component
                  this={getTypeIcon(item.type)}
                  class="space-y-4"
                />
              </div>

              <div class="space-y-4">
                {#if item.classification?.relevance !== undefined}
                  <span
                    class="space-y-4"
                  >
                    {Math.round(item.classification.relevance * 100)}%
                  </span>
                {/if}

                <Button
                  variant="ghost"
                  size="sm"
                  onclick={() => (editingEvidence = item.id)}
                >
                  <Eye class="space-y-4" />
                </Button>

                <Button
                  variant="ghost"
                  size="sm"
                  onclick={() => deleteEvidence(item.id)}
                >
                  <Trash2 class="space-y-4" />
                </Button>
              </div>
            </div>

            <!-- Content -->
            {#if editingEvidence === item.id}
              <div class="space-y-4">
                <input
                  type="text"
                  bind:value={item.title}
                  class="space-y-4"
                  placeholder="Evidence title"
                />
                <textarea
                  bind:value={item.description}
                  class="space-y-4"
                  placeholder="Description"
                ></textarea>
                <div class="space-y-4">
                  <Button
                    size="sm"
                    onclick={() =>
                      updateEvidence(item.id, {
                        title: item.title,
                        description: item.description,
                      })}
                  >
                    Save
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onclick={() => (editingEvidence = null)}
                  >
                    Cancel
                  </Button>
                </div>
              </div>
            {:else}
              <h3 class="space-y-4">
                {item.title}
              </h3>

              {#if item.description}
                <p class="space-y-4">
                  {item.description}
                </p>
              {/if}

              <!-- Metadata -->
              <div class="space-y-4">
                <div class="space-y-4">
                  <Tag class="space-y-4" />
                  <span>{item.type}</span>
                </div>
                {#if item.timeline?.createdAt}
                  <div class="space-y-4">
                    <Calendar class="space-y-4" />
                    <span>{formatDate(item.timeline.createdAt)}</span>
                  </div>
                {/if}
              </div>

              <!-- Tags -->
              {#if item.tags?.length}
                <div class="space-y-4">
                  {#each item.tags as tag}
                    <span
                      class="space-y-4"
                      >{tag}</span
                    >
                  {/each}
                </div>
              {/if}
            {/if}
          </div>
        {/each}
      </div>
    {:else}
      <!-- List View -->
      <div class="space-y-4">
        <table class="space-y-4">
          <thead class="space-y-4">
            <tr>
              <th
                class="space-y-4"
              >
                <input
                  type="checkbox"
                  onchange={(e) =>
                    (e.target as HTMLInputElement).checked
                      ? selectAll()
                      : clearSelection()}
                  class="space-y-4"
                />
              </th>
              <th
                class="space-y-4"
                >Evidence</th
              >
              <th
                class="space-y-4"
                >Type</th
              >
              <th
                class="space-y-4"
                >Date</th
              >
              <th
                class="space-y-4"
                >Relevance</th
              >
              <th class="space-y-4"
                ><span class="space-y-4">Actions</span></th
              >
            </tr>
          </thead>
          <tbody class="space-y-4">
            {#each paginatedEvidence as item (item.id)}
              <tr
                class="space-y-4"
              >
                <td class="space-y-4">
                  <input
                    type="checkbox"
                    checked={selectedEvidence.has(item.id)}
                    onchange={() => toggleSelection(item.id)}
                    class="space-y-4"
                  />
                </td>
                <td class="space-y-4">
                  <div class="space-y-4">
                    <svelte:component
                      this={getTypeIcon(item.type)}
                      class="space-y-4"
                    />
                    <div>
                      <div class="space-y-4">
                        {item.title}
                      </div>
                      {#if item.description}
                        <div class="space-y-4">
                          {item.description}
                        </div>
                      {/if}
                    </div>
                  </div>
                </td>
                <td class="space-y-4"
                  >{item.type}</td
                >
                <td class="space-y-4">
                  {item.timeline?.createdAt
                    ? formatDate(item.timeline.createdAt)
                    : "-"}
                </td>
                <td class="space-y-4">
                  {#if item.classification?.relevance !== undefined}
                    <span
                      class="space-y-4"
                    >
                      {Math.round(item.classification.relevance * 100)}%
                    </span>
                  {:else}
                    <span class="space-y-4">-</span>
                  {/if}
                </td>
                <td
                  class="space-y-4"
                >
                  <div class="space-y-4">
                    <Button
                      variant="ghost"
                      size="sm"
                      onclick={() => (editingEvidence = item.id)}
                    >
                      <Eye class="space-y-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onclick={() => deleteEvidence(item.id)}
                    >
                      <Trash2 class="space-y-4" />
                    </Button>
                  </div>
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}

    <!-- Pagination -->
    {#if totalPages > 1}
      <div class="space-y-4">
        <div class="space-y-4">
          Showing {currentPage * pageSize + 1} to {Math.min(
            (currentPage + 1) * pageSize,
            filteredEvidence.length
          )} of {filteredEvidence.length} results
        </div>

        <div class="space-y-4">
          <Button
            variant="outline"
            size="sm"
            disabled={currentPage === 0}
            onclick={() => currentPage--}
          >
            Previous
          </Button>

          <span class="space-y-4">
            Page {currentPage + 1} of {totalPages}
          </span>

          <Button
            variant="outline"
            size="sm"
            disabled={currentPage >= totalPages - 1}
            onclick={() => currentPage++}
          >
            Next
          </Button>
        </div>
      </div>
    {/if}
  {/if}
</div>

<style>
  /* @unocss-include */
  .line-clamp-2 {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    -webkit-box-orient: vertical
    overflow: hidden
}
  .line-clamp-3 {
    display: -webkit-box;
    -webkit-line-clamp: 3;
    line-clamp: 3;
    -webkit-box-orient: vertical
    overflow: hidden
}
</style>
