<script lang="ts">
  import Fuse from "fuse.js";
  import { onMount } from "svelte";
  import { quintOut } from "svelte/easing";
  import { slide } from "svelte/transition";
  import { sidebarStore } from "../stores/canvas";
  import { loki, lokiStore } from "../stores/lokiStore";
  import InfiniteScrollList from "./InfiniteScrollList.svelte";
  import SearchBar from "./SearchBar.svelte";
  import TagList from "./TagList.svelte";

  import { FileText, Folder, Tag, X } from "lucide-svelte";

  let sidebarElement: HTMLElement;
  let isHovered = false;
  let isPinned = false;
  let searchQuery = "";
  let activeTab = "evidence";
  let fuse: Fuse<any>;

  // Reactive data
  $: sidebarOpen = $sidebarStore.open || isHovered || isPinned;
  $: evidenceItems = $lokiStore.evidence || [];
  $: notesItems = $lokiStore.notes || [];
  $: canvasStates = $lokiStore.canvasStates || [];

  // Initialize Fuse search
  $: if (activeTab === "evidence" && evidenceItems.length > 0) {
    fuse = new Fuse(evidenceItems, {
      keys: ["fileName", "description", "tags"],
      threshold: 0.3,
    });
  } else if (activeTab === "notes" && notesItems.length > 0) {
    fuse = new Fuse(notesItems, {
      keys: ["title", "content", "tags"],
      threshold: 0.3,
    });
}
  // Search results
  $: searchResults =
    searchQuery && fuse
      ? fuse.search(searchQuery).map((result) => result.item)
      : activeTab === "evidence"
        ? evidenceItems
        : notesItems;

  onMount(() => {
    // Initialize Loki store
    loki.init();

    // Load data
    refreshData();
  });

  function refreshData() {
    if (activeTab === "evidence") {
      loki.evidence.refreshStore();
    } else if (activeTab === "notes") {
      loki.notes.refreshStore();
}}
  function handleMouseEnter() {
    isHovered = true;
}
  function handleMouseLeave() {
    isHovered = false;
}
  function togglePin() {
    isPinned = !isPinned;
    sidebarStore.update((state) => ({ ...state, open: isPinned }));
}
  function handleSearch(event: CustomEvent) {
    searchQuery = event.detail.query;
}
  function handleItemClick(item: unknown) {
    console.log("Item clicked:", item);
    // Emit event or call parent function
}
  function handleTabChange(tab: string) {
    activeTab = tab;
    searchQuery = "";
    refreshData();
}
</script>

<div
  class="mx-auto px-4 max-w-7xl"
  class:open={sidebarOpen}
  bind:this={sidebarElement}
  role="complementary"
  aria-label="Content sidebar"
  onmouseenter={handleMouseEnter}
  onmouseleave={handleMouseLeave}
>
  <!-- Hover trigger area when closed -->
  {#if !sidebarOpen}
    <div class="mx-auto px-4 max-w-7xl" aria-hidden="true"></div>
  {/if}

  <!-- Sidebar content -->
  {#if sidebarOpen}
    <div
      class="mx-auto px-4 max-w-7xl"
      transition:slide={{ duration: 300, easing: quintOut, axis: "x"  "
    >
      <!-- Header -->
      <div class="mx-auto px-4 max-w-7xl">
        <h3>Content Library</h3>
        <div class="mx-auto px-4 max-w-7xl">
          <button
            class={`pin-button ${isPinned ? "pinned" : ""}`}
            onclick={() => togglePin()}
            aria-label={isPinned ? "Unpin sidebar" : "Pin sidebar"}
          >
            <Tag size={16} />
          </button>

          {#if !isPinned}
            <button
              class="mx-auto px-4 max-w-7xl"
              onclick={() => (isHovered = false)}
              aria-label="Close sidebar"
            >
              <X size={16} />
            </button>
          {/if}
        </div>
      </div>

      <!-- Search -->
      <div class="mx-auto px-4 max-w-7xl">
        <SearchBar
          placeholder="Search {activeTab}..."
          value={searchQuery}
          on:search={handleSearch}
        />
      </div>

      <!-- Tabs -->
      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">
          <button
            class="mx-auto px-4 max-w-7xl"
            class:active={activeTab === "evidence"}
            onclick={() => handleTabChange("evidence")}
          >
            <Folder size={16} />
            Evidence
          </button>
          <button
            class="mx-auto px-4 max-w-7xl"
            class:active={activeTab === "notes"}
            onclick={() => handleTabChange("notes")}
          >
            <FileText size={16} />
            Notes
          </button>
          <button
            class="mx-auto px-4 max-w-7xl"
            class:active={activeTab === "canvas"}
            onclick={() => handleTabChange("canvas")}
          >
            <Tag size={16} />
            Canvas
          </button>
        </div>

        <!-- Evidence Tab -->
        {#if activeTab === "evidence"}
          <div class="mx-auto px-4 max-w-7xl">
            <InfiniteScrollList
              items={searchResults}
              itemType="evidence"
              on:itemClick={handleItemClick}
              on:loadMore={refreshData}
            />
          </div>
        {/if}

        <!-- Notes Tab -->
        {#if activeTab === "notes"}
          <div class="mx-auto px-4 max-w-7xl">
            <InfiniteScrollList
              items={searchResults}
              itemType="notes"
              on:itemClick={handleItemClick}
              on:loadMore={refreshData}
            />
          </div>
        {/if}

        <!-- Canvas States Tab -->
        {#if activeTab === "canvas"}
          <div class="mx-auto px-4 max-w-7xl">
            <InfiniteScrollList
              items={canvasStates}
              itemType="canvas"
              on:itemClick={handleItemClick}
              on:loadMore={refreshData}
            />
          </div>
        {/if}
      </div>

      <!-- Tags -->
      <div class="mx-auto px-4 max-w-7xl">
        <TagList />
      </div>
    </div>
  {/if}
</div>

<style>
  .sidebar-container {
    position: fixed;
    top: 60px; /* Header height */
    left: 0;
    bottom: 0;
    width: 320px;
    z-index: 20;
    pointer-events: none;
    transition: transform 0.3s ease;
    transform: translateX(-100%);
}
  .sidebar-container.open {
    transform: translateX(0);
    pointer-events: all;
}
  .hover-trigger {
    position: absolute;
    top: 0;
    left: 0;
    width: 20px;
    height: 100%;
    background: transparent;
    pointer-events: all;
    z-index: 1;
}
  .sidebar-content {
    width: 100%;
    height: 100%;
    background: var(--pico-card-background-color);
    border-right: 1px solid var(--pico-muted-border-color);
    box-shadow: 2px 0 8px rgba(0, 0, 0, 0.1);
    display: flex;
    flex-direction: column;
    overflow: hidden;
}
  .sidebar-header {
    padding: 1rem;
    border-bottom: 1px solid var(--pico-muted-border-color);
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--pico-background-color);
}
  .sidebar-header h3 {
    margin: 0;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--pico-color);
}
  .header-actions {
    display: flex;
    gap: 0.5rem;
}
  .pin-button.pinned {
    background: var(--pico-primary-background);
    color: var(--pico-primary-inverse);
}
  .pin-button,
  .close-button {
    background: transparent;
    border: none;
    padding: 0.25rem;
    border-radius: 0.25rem;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--pico-color);
}
  .pin-button:hover,
  .close-button:hover {
    background: var(--pico-secondary-background);
}
  .search-section {
    padding: 1rem;
    border-bottom: 1px solid var(--pico-muted-border-color);
}
  .tab-list {
    display: flex;
    border-bottom: 1px solid var(--pico-muted-border-color);
    background: var(--pico-background-color);
}
  .tab-trigger {
    flex: 1;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem 1rem;
    background: transparent;
    border: none;
    color: var(--pico-muted-color);
    cursor: pointer;
    transition: all 0.2s ease;
}
  .tab-trigger:hover {
    background: var(--pico-secondary-background);
    color: var(--pico-color);
}
  .tab-trigger.active {
    background: var(--pico-primary-background);
    color: var(--pico-primary-inverse);
    border-bottom: 2px solid var(--pico-primary);
}
  .tab-content {
    flex: 1;
    overflow: hidden;
    display: flex;
    flex-direction: column;
}
  .tabs-container {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}
  .tags-section {
    padding: 1rem;
    border-top: 1px solid var(--pico-muted-border-color);
    background: var(--pico-background-color);
}
  /* Responsive */
  @media (max-width: 768px) {
    .sidebar-container {
      width: 280px;
}}
</style>