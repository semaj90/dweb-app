<script lang="ts">
  interface Props {
    onloadMore?: (event?: any) => void;
    onitemClick?: (event?: any) => void;
  }
  let {
    items = [],
    itemType = "evidence",
    loadMoreThreshold = 100,
    pageSize = 20,
    isLoading = false,
    selectedIndex = -1
  } = $props();



    import { quintOut } from "svelte/easing";
  import { fade, slide } from "svelte/transition";

  import {
    File,
    FileEdit,
    FileText,
    Image,
    Palette,
    Video,
  } from "lucide-svelte";

      let { loadMoreThreshold = $bindable() } = $props(); // 100; // pixels from bottom
      let { selectedIndex = $bindable() } = $props(); // number = -1; // Index of selected item

  
  let scrollContainer: HTMLElement
  let displayedItems: any[] = [];
  let currentPage = 0;
  let hasMore = true;

  $effect(() => { {
    // Reset when items change
    if (items !== displayedItems.slice(0, items.length)) {
      currentPage = 0;
      displayedItems = [];
      loadMore();
}}
  onMount(() => {
    loadMore();
  });

  function loadMore() {
    if (isLoading || !hasMore) return;

    const startIndex = currentPage * pageSize;
    const endIndex = Math.min(startIndex + pageSize, items.length);
    const newItems = items.slice(startIndex, endIndex);

    if (newItems.length === 0) {
      hasMore = false;
      return;
}
    displayedItems = [...displayedItems, ...newItems];
    currentPage++;

    // Check if we have more items to load
    hasMore = endIndex < items.length;

    // Emit event for loading more data from API
    if (!hasMore && items.length >= currentPage * pageSize) {
      onloadMore?.();
}}
  function handleScroll() {
    if (!scrollContainer) return;

    const { scrollTop, scrollHeight, clientHeight } = scrollContainer;
    const scrolledToBottom =
      scrollHeight - scrollTop - clientHeight < loadMoreThreshold;

    if (scrolledToBottom) {
      loadMore();
}}
  function handleItemClick(item: any) {
    onitemClick?.();
}
  function getItemIcon(item: any) {
    if (itemType === "notes") {
      return FileEdit;
    } else if (itemType === "canvas") {
      return Palette;
    } else {
      // Evidence
      const fileType = item.fileType || item.type || "";
      if (fileType.startsWith("image/")) return Image;
      if (fileType.startsWith("video/")) return Video;
      if (fileType.includes("text") || fileType.includes("pdf"))
        return FileText;
      return File;
}}
  function formatDate(dateString: string) {
    return new Date(dateString).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
}
  function truncateText(text: string, maxLength = 100) {
    if (!text) return "";
    return text.length > maxLength
      ? text.substring(0, maxLength) + "..."
      : text;
}
</script>

<div
  class="space-y-4"
  bind:this={scrollContainer}
  onscroll={handleScroll}
  role="listbox"
  aria-label="{itemType} list"
>
  {#if displayedItems.length === 0 && !isLoading}
    <div class="space-y-4" transitionfade={{ duration: 200 }}>
      <div class="space-y-4">
        <svelte:component this={getItemIcon({})} size={48} />
      </div>
      <p class="space-y-4">No {itemType} found</p>
    </div>
  {:else}
    <div class="space-y-4">
      {#each displayedItems as item, index (item.id || index)}
        <div
          class="space-y-4"
          transitionslide={{ duration: 300, easing: quintOut }}
          onclick={() => handleItemClick(item)}
          onkeydown={(e) => e.key === "Enter" && handleItemClick(item)}
          role="option"
          tabindex={0}
          aria-label="{itemType} item"
          aria-selected={index === selectedIndex}
        >
          <div class="space-y-4">
            <svelte:component this={getItemIcon(item)} size={20} />
          </div>

          <div class="space-y-4">
            <div class="space-y-4">
              <h4 class="space-y-4">
                {#if itemType === "evidence"}
                  {item.fileName || item.title || "Untitled Evidence"}
                {:else if itemType === "notes"}
                  {item.title || "Untitled Note"}
                {:else}
                  {item.name || `Canvas ${formatDate(item.lastModified)}`}
                {/if}
              </h4>
              <span class="space-y-4">
                {formatDate(
                  item.createdAt || item.lastModified || item.updatedAt
                )}
              </span>
            </div>

            <p class="space-y-4">
              {#if itemType === "evidence"}
                {truncateText(item.description)}
              {:else if itemType === "notes"}
                {truncateText(item.content)}
              {:else}
                Canvas state with {item.objectCount || 0} objects
              {/if}
            </p>

            {#if item.tags && item.tags.length > 0}
              <div class="space-y-4">
                {#each item.tags.slice(0, 3) as tag}
                  <span class="space-y-4">{tag}</span>
                {/each}
                {#if item.tags.length > 3}
                  <span class="space-y-4">+{item.tags.length - 3}</span>
                {/if}
              </div>
            {/if}
          </div>
        </div>
      {/each}
    </div>
  {/if}

  {#if isLoading}
    <div class="space-y-4" transitionfade={{ duration: 200 }}>
      <div class="space-y-4"></div>
      <p>Loading more {itemType}...</p>
    </div>
  {/if}

  {#if !hasMore && displayedItems.length > 0}
    <div class="space-y-4" transitionfade={{ duration: 200 }}>
      <p>No more {itemType} to load</p>
    </div>
  {/if}
</div>

<style>
  /* @unocss-include */
  .infinite-scroll-container {
    flex: 1;
    overflow-y: auto
    padding: 0;
    background: var(--bg-primary);
}
  .empty-state {
    display: flex
    flex-direction: column
    align-items: center
    justify-content: center
    padding: 2rem;
    text-align: center
    color: var(--text-muted);
    min-height: 200px;
}
  .empty-icon {
    margin-bottom: 1rem;
    opacity: 0.5;
}
  .empty-text {
    margin: 0;
    font-size: 0.875rem;
}
  .items-list {
    padding: 0.5rem;
}
  .list-item {
    display: flex
    align-items: flex-start;
    gap: 0.75rem;
    padding: 0.75rem;
    border-radius: 8px;
    cursor: pointer
    transition: all 0.2s ease;
    border: 1px solid transparent;
    margin-bottom: 0.5rem;
}
  .list-item:hover {
    background: var(--bg-tertiary);
    border-color: var(--border-light);
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}
  .list-item:focus {
    outline: 2px solid var(--harvard-crimson);
    outline-offset: 2px;
}
  .list-item:active {
    transform: translateY(0);
}
  .item-icon {
    display: flex
    align-items: center
    justify-content: center
    width: 40px;
    height: 40px;
    border-radius: 8px;
    background: var(--bg-secondary);
    color: var(--harvard-crimson);
    flex-shrink: 0;
}
  .item-content {
    flex: 1;
    min-width: 0;
}
  .item-header {
    display: flex
    align-items: center
    justify-content: space-between;
    gap: 0.5rem;
    margin-bottom: 0.25rem;
}
  .item-title {
    margin: 0;
    font-size: 0.875rem;
    font-weight: 600;
    color: var(--text-primary);
    overflow: hidden
    text-overflow: ellipsis
    white-space: nowrap
    flex: 1;
}
  .item-date {
    font-size: 0.75rem;
    color: var(--text-muted);
    flex-shrink: 0;
}
  .item-description {
    margin: 0 0 0.5rem;
    font-size: 0.8rem;
    color: var(--text-muted);
    line-height: 1.4;
    overflow: hidden
    text-overflow: ellipsis
    display: -webkit-box;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    -webkit-box-orient: vertical
}
  .item-tags {
    display: flex
    flex-wrap: wrap
    gap: 0.25rem;
    margin-top: 0.5rem;
}
  .tag {
    font-size: 0.7rem;
    padding: 0.15rem 0.5rem;
    background: var(--bg-secondary);
    color: var(--harvard-crimson);
    border-radius: 12px;
    border: 1px solid var(--harvard-crimson);
}
  .tag-more {
    font-size: 0.7rem;
    padding: 0.15rem 0.5rem;
    background: var(--muted-background);
    color: var(--text-muted);
    border-radius: 12px;
    border: 1px solid var(--border-light);
}
  .loading-indicator {
    display: flex
    flex-direction: column
    align-items: center
    padding: 1rem;
    color: var(--text-muted);
    gap: 0.5rem;
}
  .loading-spinner {
    width: 24px;
    height: 24px;
    border: 2px solid var(--border-light);
    border-top: 2px solid var(--harvard-crimson);
    border-radius: 50%;
    animation: spin 1s linear infinite;
}
  .end-indicator {
    text-align: center
    padding: 1rem;
    color: var(--text-muted);
    font-size: 0.875rem;
}
  @keyframes spin {
    0% {
      transform: rotate(0deg);
}
    100% {
      transform: rotate(360deg);
}}
  /* Custom scrollbar */
  .infinite-scroll-container::-webkit-scrollbar {
    width: 6px;
}
  .infinite-scroll-container::-webkit-scrollbar-track {
    background: var(--bg-primary);
}
  .infinite-scroll-container::-webkit-scrollbar-thumb {
    background: var(--border-light);
    border-radius: 3px;
}
  .infinite-scroll-container::-webkit-scrollbar-thumb:hover {
    background: var(--harvard-crimson);
}
</style>
