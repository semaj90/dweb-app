<script lang="ts">
  import { Button } from "$lib/components/ui/button";
  import {
    evidenceActions,
    evidenceGrid,
    filteredEvidence,
    type Evidence,
  } from "$lib/stores/evidence-store";
  import {
    formatFileSize,
    getFileCategory,
    isImageFile,
  } from "$lib/utils/file-utils";
  import { saveAs } from "file-saver";
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

  export let caseId: string | undefined = undefined;
  export let showHeader: boolean = true;
  export let columns: number = 3;

  let searchInput: HTMLInputElement;
  let selectedItem: Evidence | null = null;

  // Reactive values from store
  $: ({
    items,
    searchQuery,
    sortBy,
    sortOrder,
    selectedItems,
    viewMode,
    isLoading,
    error,
  } = $evidenceGrid);

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
    }
  }

  function toggleSelection(item: Evidence) {
    evidenceActions.toggleSelection(item.id);
  }

  function selectAll() {
    $filteredEvidence.forEach((item) => {
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
    }
  }

  function formatDate(dateString: string): string {
    return new Intl.DateTimeFormat("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }).format(new Date(dateString));
  }

  async function downloadEvidence(item: Evidence) {
    if (!item.fileUrl) return;

    try {
      const response = await fetch(item.fileUrl);
      const blob = await response.blob();
      saveAs(blob, item.fileName || item.title);
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
      }
    }
  }

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
    }
  }

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
    }
  }
</script>

<div class="mx-auto px-4 max-w-7xl">
  {#if showHeader}
    <!-- Header with search and controls -->
    <div
      class="mx-auto px-4 max-w-7xl"
    >
      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">
          <Search
            class="mx-auto px-4 max-w-7xl"
          />
          <input
            bind:this={searchInput}
            type="text"
            placeholder="Search evidence..."
            value={searchQuery}
            oninput={handleSearch}
            class="mx-auto px-4 max-w-7xl"
          />
        </div>
      </div>

      <div class="mx-auto px-4 max-w-7xl">
        <!-- Sort dropdown -->
        <select
          value={sortBy}
          onchange={(e) => toggleSort((e.target as HTMLSelectElement)?.value)}
          class="mx-auto px-4 max-w-7xl"
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
          class="mx-auto px-4 max-w-7xl"
        >
          {#if sortOrder === "asc"}
            <SortAsc class="mx-auto px-4 max-w-7xl" />
          {:else}
            <SortDesc class="mx-auto px-4 max-w-7xl" />
          {/if}
        </Button>

        <!-- View mode toggle -->
        <Button
          variant="secondary"
          size="sm"
          onclick={() => toggleViewMode()}
          class="mx-auto px-4 max-w-7xl"
        >
          {#if viewMode === "grid"}
            <List class="mx-auto px-4 max-w-7xl" />
          {:else}
            <Grid class="mx-auto px-4 max-w-7xl" />
          {/if}
        </Button>
      </div>
    </div>

    <!-- Selection controls -->
    {#if selectedItems.size > 0}
      <div
        class="mx-auto px-4 max-w-7xl"
      >
        <span class="mx-auto px-4 max-w-7xl">
          {selectedItems.size} item{selectedItems.size !== 1 ? "s" : ""} selected
        </span>
        <div class="mx-auto px-4 max-w-7xl">
          <Button
            variant="secondary"
            size="sm"
            onclick={() => clearSelection()}
          >
            Clear
          </Button>
          <Button variant="secondary" size="sm">
            <Download class="mx-auto px-4 max-w-7xl" />
            Download
          </Button>
          <Button variant="secondary" size="sm">
            <Archive class="mx-auto px-4 max-w-7xl" />
            Archive
          </Button>
        </div>
      </div>
    {/if}
  {/if}

  <!-- Loading state -->
  {#if isLoading}
    <div class="mx-auto px-4 max-w-7xl">
      <div
        class="mx-auto px-4 max-w-7xl"
      ></div>
      <span class="mx-auto px-4 max-w-7xl">Loading evidence...</span>
    </div>
  {:else if error}
    <!-- Error state -->
    <div class="mx-auto px-4 max-w-7xl">
      <div class="mx-auto px-4 max-w-7xl">Error loading evidence</div>
      <p class="mx-auto px-4 max-w-7xl">{error}</p>
      <Button
        variant="secondary"
        size="sm"
        class="mx-auto px-4 max-w-7xl"
        onclick={() => evidenceActions.loadEvidence(caseId)}
      >
        Try Again
      </Button>
    </div>
  {:else if $filteredEvidence.length === 0}
    <!-- Empty state -->
    <div class="mx-auto px-4 max-w-7xl">
      <File class="mx-auto px-4 max-w-7xl" />
      <h3 class="mx-auto px-4 max-w-7xl">No evidence found</h3>
      <p class="mx-auto px-4 max-w-7xl">
        {searchQuery
          ? "Try adjusting your search terms."
          : "Upload some evidence to get started."}
      </p>
    </div>
  {:else}
    <!-- Evidence grid/list -->
    <div class="mx-auto px-4 max-w-7xl">
      {#if viewMode === "grid"}
        <!-- Grid view -->
        <div
          class="mx-auto px-4 max-w-7xl"
          style="grid-template-columns: repeat({columns}, minmax(0, 1fr))"
        >
          {#each $filteredEvidence as item (item.id)}
            <div
              class="mx-auto px-4 max-w-7xl"
              onclick={() => toggleSelection(item)}
              on:contextmenu|preventDefault={(e) => showContextMenu(e, item)}
            >
              <!-- Preview/Thumbnail -->
              <div
                class="mx-auto px-4 max-w-7xl"
              >
                {#if item.fileUrl && isImageFile(item.mimeType || "")}
                  <img
                    src={item.fileUrl}
                    alt={item.title}
                    class="mx-auto px-4 max-w-7xl"
                    loading="lazy"
                  />
                {:else}
                  <svelte:component
                    this={getFileIcon(item.evidenceType, item.mimeType)}
                    class="mx-auto px-4 max-w-7xl"
                  />
                {/if}

                <!-- Overlay with selection checkbox -->
                <div class="mx-auto px-4 max-w-7xl">
                  <input
                    type="checkbox"
                    checked={selectedItems.has(item.id)}
                    onchange={() => toggleSelection(item)}
                    class="mx-auto px-4 max-w-7xl"
                  />
                </div>

                <!-- File type badge -->
                <div class="mx-auto px-4 max-w-7xl">
                  <span
                    class="mx-auto px-4 max-w-7xl"
                  >
                    {getFileCategory(item.mimeType || item.evidenceType)}
                  </span>
                </div>
              </div>

              <!-- Content -->
              <div class="mx-auto px-4 max-w-7xl">
                <h3 class="mx-auto px-4 max-w-7xl">
                  {item.title}
                </h3>

                {#if item.description}
                  <p class="mx-auto px-4 max-w-7xl">
                    {item.description}
                  </p>
                {/if}

                <!-- Metadata -->
                <div class="mx-auto px-4 max-w-7xl">
                  <div class="mx-auto px-4 max-w-7xl">
                    <span>{formatDate(item.uploadedAt)}</span>
                    {#if item.fileSize}
                      <span>{formatFileSize(item.fileSize)}</span>
                    {/if}
                  </div>

                  {#if item.tags && item.tags.length > 0}
                    <div class="mx-auto px-4 max-w-7xl">
                      {#each item.tags.slice(0, 3) as tag}
                        <span
                          class="mx-auto px-4 max-w-7xl"
                        >
                          {tag}
                        </span>
                      {/each}
                      {#if item.tags.length > 3}
                        <span
                          class="mx-auto px-4 max-w-7xl"
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
        <div class="mx-auto px-4 max-w-7xl">
          {#each $filteredEvidence as item (item.id)}
            <div
              class="mx-auto px-4 max-w-7xl"
              onclick={() => toggleSelection(item)}
              on:contextmenu|preventDefault={(e) => showContextMenu(e, item)}
            >
              <!-- Selection checkbox -->
              <input
                type="checkbox"
                checked={selectedItems.has(item.id)}
                onchange={() => toggleSelection(item)}
                class="mx-auto px-4 max-w-7xl"
              />

              <!-- File icon -->
              <div class="mx-auto px-4 max-w-7xl">
                <svelte:component
                  this={getFileIcon(item.evidenceType, item.mimeType)}
                  class="mx-auto px-4 max-w-7xl"
                />
              </div>

              <!-- Content -->
              <div class="mx-auto px-4 max-w-7xl">
                <div class="mx-auto px-4 max-w-7xl">
                  <div class="mx-auto px-4 max-w-7xl">
                    <h3 class="mx-auto px-4 max-w-7xl">
                      {item.title}
                    </h3>
                    {#if item.description}
                      <p class="mx-auto px-4 max-w-7xl">
                        {item.description}
                      </p>
                    {/if}
                  </div>

                  <div class="mx-auto px-4 max-w-7xl">
                    <p class="mx-auto px-4 max-w-7xl">
                      {formatDate(item.uploadedAt)}
                    </p>
                    {#if item.fileSize}
                      <p class="mx-auto px-4 max-w-7xl">
                        {formatFileSize(item.fileSize)}
                      </p>
                    {/if}
                  </div>
                </div>

                <!-- Tags -->
                {#if item.tags && item.tags.length > 0}
                  <div class="mx-auto px-4 max-w-7xl">
                    {#each item.tags.slice(0, 5) as tag}
                      <span
                        class="mx-auto px-4 max-w-7xl"
                      >
                        {tag}
                      </span>
                    {/each}
                    {#if item.tags.length > 5}
                      <span
                        class="mx-auto px-4 max-w-7xl"
                      >
                        +{item.tags.length - 5}
                      </span>
                    {/if}
                  </div>
                {/if}
              </div>

              <!-- Actions -->
              <div class="mx-auto px-4 max-w-7xl">
                <Button variant="ghost" size="sm" class="mx-auto px-4 max-w-7xl">
                  <MoreHorizontal class="mx-auto px-4 max-w-7xl" />
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