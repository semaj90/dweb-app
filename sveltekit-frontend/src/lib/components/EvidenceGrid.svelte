<script lang="ts">
  import type { Evidence } from '$lib/types';
  import { Button } from "$lib/components/ui/button";
  import {
    evidenceActions,
    evidenceGrid,
    filteredEvidence,
    type EvidenceGridState,
  } from "$lib/stores/evidence-store";
  import {
    formatFileSize,
    getFileCategory,
    isImageFile,
  } from "$lib/utils/file-utils";
  // Note: file-saver package would need to be installed
  // import { saveAs } from "file-saver";
  import {
    Archive,
    Download,
    Eye,
    File,
    FileText,
    Grid,
    Image,
    List,
    MoreHorizontal,
    Music,
    Search,
    SortAsc,
    SortDesc,
    Tag,
    Trash2,
    Video,
  } from "lucide-svelte";
  import { onMount } from "svelte";

  interface Props {
    caseId?: string;
    showHeader?: boolean;
    columns?: number;
  }

  let {
    caseId = undefined,
    showHeader = true,
    columns = 3
  }: Props = $props();

  let searchInput: HTMLInputElement;
  let selectedItem: Evidence | null = null;

  // In Svelte 5, access store values directly
  let gridData = $state<EvidenceGridState | undefined>(undefined);
  let filteredData = $state<Evidence[]>([]);

  // Subscribe to store changes
  $effect(() => {
    const unsubscribe = evidenceGrid.subscribe(value => {
      gridData = value;
    });
    const unsubscribeFiltered = filteredEvidence.subscribe(value => {
      filteredData = value;
    });
    return () => {
      unsubscribe();
      unsubscribeFiltered();
    };
  });

  // Derived values
  let items = $derived(gridData?.items || []);
  let searchQuery = $derived(gridData?.searchQuery || '');
  let sortBy = $derived(gridData?.sortBy || 'uploadedAt');
  let sortOrder = $derived(gridData?.sortOrder || 'desc');
  let selectedItems = $derived(gridData?.selectedItems || new Set());
  let viewMode = $derived(gridData?.viewMode || 'grid');
  let isLoading = $derived(gridData?.isLoading || false);
  let error = $derived(gridData?.error);

  // Load evidence on mount
  onMount(() => {
    evidenceActions.loadEvidence(caseId);
  });

  function handleSearch(event: Event) {
    const target = event.target as HTMLInputElement;
    evidenceActions.setSearchQuery(target.value);
}
  function toggleViewMode() {
    evidenceActions.setViewMode(viewMode === "grid" ? "list" : "grid");
}
  function toggleSort(field: typeof sortBy) {
    if (sortBy === field) {
      evidenceActions.setSorting(field, sortOrder === "asc" ? "desc" : "asc");
    } else {
      evidenceActions.setSorting(field, "desc");
}}
  function toggleSelection(item: Evidence) {
    evidenceActions.toggleSelection(item.id);
}
  function selectAll() {
    filteredData.forEach((item) => {
      if (!selectedItems.has(item.id)) {
        evidenceActions.toggleSelection(item.id);
}
    });
}
  function clearSelection() {
    evidenceActions.clearSelection();
}
  function getFileIcon(evidenceType: string, mimeType?: string) {
    if (mimeType) {
      if (isImageFile(mimeType)) return Image;
      if (mimeType.startsWith("video/")) return Video;
      if (mimeType.startsWith("audio/")) return Music;
      if (mimeType.includes("pdf")) return FileText;
}
    switch (evidenceType.toLowerCase()) {
      case "image":
        return Image;
      case "video":
        return Video;
      case "audio":
        return Music;
      case "document":
      case "pdf":
        return FileText;
      default:
        return File;
}}  function formatDate(date: string | Date | undefined): string {
    if (!date) return 'Unknown';
    const dateObj = typeof date === 'string' ? new Date(date) : date;
    return new Intl.DateTimeFormat("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }).format(dateObj);
  }
  async function downloadEvidence(item: Evidence) {
    if (!item.fileUrl) return;

    try {
      const response = await fetch(item.fileUrl);
      const blob = await response.blob();

      // Native browser download without file-saver library
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = item.fileName || item.title || 'evidence-file';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error("Download failed:", error);
    }
  }
  async function deleteEvidence(item: Evidence) {
    if (confirm(`Are you sure you want to delete "${item.title}"?`)) {
      try {
        await evidenceActions.deleteEvidence(item.id);
      } catch (error) {
        console.error("Delete failed:", error);
        alert("Failed to delete evidence. Please try again.");
}}}
  function openPreview(item: Evidence) {
    selectedItem = item;
    // You can implement a preview modal here
}
  function showContextMenu(event: MouseEvent, item: Evidence) {
    // Simple context menu implementation - could be enhanced with a proper context menu component
    event.preventDefault();
    selectedItem = item;
    // For now, just select the item - can be enhanced later with actual context menu
    if (!selectedItems.has(item.id)) {
      toggleSelection(item);
}}
  // Context menu actions
  const contextMenuItems = [
    { label: "Preview", icon: Eye, action: "preview" },
    { label: "Download", icon: Download, action: "download" },
    { label: "Save for Later", icon: Archive, action: "save" },
    { label: "Add Tags", icon: Tag, action: "tag" },
    { label: "Delete", icon: Trash2, action: "delete", destructive: true },
  ];

  function handleContextAction(action: string, item: Evidence) {
    switch (action) {
      case "preview":
        openPreview(item);
        break;
      case "download":
        downloadEvidence(item);
        break;
      case "save":
        // Implement save for later functionality
        console.log("Save for later:", item);
        break;
      case "tag":
        // Implement tagging modal
        console.log("Add tags:", item);
        break;
      case "delete":
        deleteEvidence(item);
        break;
}}
</script>

<div class="container mx-auto px-4">
  {#if showHeader}
    <!-- Header with search and controls -->
    <div
      class="container mx-auto px-4"
    >
      <div class="container mx-auto px-4">
        <div class="container mx-auto px-4">
          <Search
            class="container mx-auto px-4"
          />
          <input
            bind:this={searchInput}
            type="text"
            placeholder="Search evidence..."
            value={searchQuery}
            oninput={handleSearch}
            class="container mx-auto px-4"
          />
        </div>
      </div>

      <div class="container mx-auto px-4">
        <!-- Sort dropdown -->
        <select
          value={sortBy}
          onchange={(e) => {
            const value = (e.target as HTMLSelectElement)?.value;
            if (value === 'title' || value === 'evidenceType' || value === 'fileSize' || value === 'uploadedAt') {
              toggleSort(value);
            }
          }}
          class="container mx-auto px-4"
        >
          <option value="uploadedAt">Date</option>
          <option value="title">Title</option>
          <option value="evidenceType">Type</option>
          <option value="fileSize">Size</option>
        </select>

        <!-- Sort direction -->
        <Button
          variant="secondary"
          size="sm"
          onclick={() => toggleSort(sortBy)}
          class="container mx-auto px-4"
        >
          {#if sortOrder === "asc"}
            <SortAsc class="container mx-auto px-4" />
          {:else}
            <SortDesc class="container mx-auto px-4" />
          {/if}
        </Button>

        <!-- View mode toggle -->
        <Button
          variant="secondary"
          size="sm"
          onclick={() => toggleViewMode()}
          class="container mx-auto px-4"
        >
          {#if viewMode === "grid"}
            <List class="container mx-auto px-4" />
          {:else}
            <Grid class="container mx-auto px-4" />
          {/if}
        </Button>
      </div>
    </div>

    <!-- Selection controls -->
    {#if selectedItems.size > 0}
      <div
        class="container mx-auto px-4"
      >
        <span class="container mx-auto px-4">
          {selectedItems.size} item{selectedItems.size !== 1 ? "s" : ""} selected
        </span>
        <div class="container mx-auto px-4">
          <Button
            variant="secondary"
            size="sm"
            onclick={() => clearSelection()}
          >
            Clear
          </Button>
          <Button variant="secondary" size="sm">
            <Download class="container mx-auto px-4" />
            Download
          </Button>
          <Button variant="secondary" size="sm">
            <Archive class="container mx-auto px-4" />
            Archive
          </Button>
        </div>
      </div>
    {/if}
  {/if}

  <!-- Loading state -->
  {#if isLoading}
    <div class="container mx-auto px-4">
      <div
        class="container mx-auto px-4"
      ></div>
      <span class="container mx-auto px-4">Loading evidence...</span>
    </div>
  {:else if error}
    <!-- Error state -->
    <div class="container mx-auto px-4">
      <div class="container mx-auto px-4">Error loading evidence</div>
      <p class="container mx-auto px-4">{error}</p>
      <Button
        variant="secondary"
        size="sm"
        class="container mx-auto px-4"
        onclick={() => evidenceActions.loadEvidence(caseId)}
      >
        Try Again
      </Button>
    </div>
  {:else if filteredData.length === 0}
    <!-- Empty state -->
    <div class="container mx-auto px-4">
      <File class="container mx-auto px-4" />
      <h3 class="container mx-auto px-4">No evidence found</h3>
      <p class="container mx-auto px-4">
        {searchQuery
          ? "Try adjusting your search terms."
          : "Upload some evidence to get started."}
      </p>
    </div>
  {:else}
    <!-- Evidence grid/list -->
    <div class="container mx-auto px-4">
      {#if viewMode === "grid"}
        <!-- Grid view -->
        <div
          class="container mx-auto px-4"
          style="grid-template-columns: repeat({columns}, minmax(0, 1fr))"
        >
          {#each filteredData as item (item.id)}
            <div
              class="container mx-auto px-4"
              onclick={() => toggleSelection(item)}
              oncontextmenu={(e) => { e.preventDefault(); showContextMenu(e, item); }}
            >
              <!-- Preview/Thumbnail -->
              <div
                class="container mx-auto px-4"
              >
                {#if item.fileUrl && isImageFile(item.mimeType || "")}
                  <img
                    src={item.fileUrl}
                    alt={item.title}
                    class="container mx-auto px-4"
                    loading="lazy"
                  />
                {:else}
                  <svelte:component
                    this={getFileIcon(item.evidenceType, item.mimeType)}
                    class="container mx-auto px-4"
                  />
                {/if}

                <!-- Overlay with selection checkbox -->
                <div class="container mx-auto px-4">
                  <input
                    type="checkbox"
                    checked={selectedItems.has(item.id)}
                    onchange={() => toggleSelection(item)}
                    class="container mx-auto px-4"
                  />
                </div>

                <!-- File type badge -->
                <div class="container mx-auto px-4">
                  <span
                    class="container mx-auto px-4"
                  >
                    {getFileCategory(item.mimeType || item.evidenceType)}
                  </span>
                </div>
              </div>

              <!-- Content -->
              <div class="container mx-auto px-4">
                <h3 class="container mx-auto px-4">
                  {item.title}
                </h3>

                {#if item.description}
                  <p class="container mx-auto px-4">
                    {item.description}
                  </p>
                {/if}

                <!-- Metadata -->
                <div class="container mx-auto px-4">
                  <div class="container mx-auto px-4">
                    <span>{formatDate(item.uploadedAt)}</span>
                    {#if item.fileSize}
                      <span>{formatFileSize(item.fileSize)}</span>
                    {/if}
                  </div>

                  {#if item.tags && item.tags.length > 0}
                    <div class="container mx-auto px-4">
                      {#each item.tags.slice(0, 3) as tag}
                        <span
                          class="container mx-auto px-4"
                        >
                          {tag}
                        </span>
                      {/each}
                      {#if item.tags.length > 3}
                        <span
                          class="container mx-auto px-4"
                        >
                          +{item.tags.length - 3}
                        </span>
                      {/if}
                    </div>
                  {/if}
                </div>
              </div>
            </div>
          {/each}
        </div>
      {:else}
        <!-- List view -->
        <div class="container mx-auto px-4">
          {#each filteredData as item (item.id)}
            <div
              class="container mx-auto px-4"
              onclick={() => toggleSelection(item)}
              oncontextmenu={(e) => { e.preventDefault(); showContextMenu(e, item); }}
            >
              <!-- Selection checkbox -->
              <input
                type="checkbox"
                checked={selectedItems.has(item.id)}
                on:change={() => toggleSelection(item)}
                class="container mx-auto px-4"
              />

              <!-- File icon -->
              <div class="container mx-auto px-4">
                <svelte:component
                  this={getFileIcon(item.evidenceType, item.mimeType)}
                  class="container mx-auto px-4"
                />
              </div>

              <!-- Content -->
              <div class="container mx-auto px-4">
                <div class="container mx-auto px-4">
                  <div class="container mx-auto px-4">
                    <h3 class="container mx-auto px-4">
                      {item.title}
                    </h3>
                    {#if item.description}
                      <p class="container mx-auto px-4">
                        {item.description}
                      </p>
                    {/if}
                  </div>

                  <div class="container mx-auto px-4">
                    <p class="container mx-auto px-4">
                      {formatDate(item.uploadedAt)}
                    </p>
                    {#if item.fileSize}
                      <p class="container mx-auto px-4">
                        {formatFileSize(item.fileSize)}
                      </p>
                    {/if}
                  </div>
                </div>

                <!-- Tags -->
                {#if item.tags && item.tags.length > 0}
                  <div class="container mx-auto px-4">
                    {#each item.tags.slice(0, 5) as tag}
                      <span
                        class="container mx-auto px-4"
                      >
                        {tag}
                      </span>
                    {/each}
                    {#if item.tags.length > 5}
                      <span
                        class="container mx-auto px-4"
                      >
                        +{item.tags.length - 5}
                      </span>
                    {/if}
                  </div>
                {/if}
              </div>

              <!-- Actions -->
              <div class="container mx-auto px-4">
                <Button variant="ghost" size="sm" class="container mx-auto px-4">
                  <MoreHorizontal class="container mx-auto px-4" />
                </Button>
              </div>
            </div>
          {/each}
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  /* @unocss-include */
  .line-clamp-2 {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
  .line-clamp-3 {
    display: -webkit-box;
    -webkit-line-clamp: 3;
    line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
</style>
