<script context="module" lang="ts">
// Svelte runes are declared globally in `src/types/svelte-helpers.d.ts`.
export {};
</script>

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
  let activeTab: "evidence" | "notes" | "canvas" = "evidence";
  let fuse: Fuse<any> | null = null;

  // Svelte store values (auto-unwrapped with $)
  // $sidebarStore and $lokiStore provided by imports
  $: sidebarOpen = $sidebarStore?.open || isHovered || isPinned;
  $: evidenceItems = $lokiStore?.evidence ?? [];
  $: notesItems = $lokiStore?.notes ?? [];
  $: canvasStates = $lokiStore?.canvasStates ?? [];

  // Create Fuse instance when relevant items change
  $: if (activeTab === "evidence" && evidenceItems.length > 0) {
    fuse = new Fuse(evidenceItems, { keys: ["fileName", "description", "tags"], threshold: 0.3 });
  } else if (activeTab === "notes" && notesItems.length > 0) {
    fuse = new Fuse(notesItems, { keys: ["title", "content", "tags"], threshold: 0.3 });
  } else {
    fuse = null;
  }

  // Compute search results reactively
  $: searchResults = (() => {
    if (searchQuery && fuse) {
      return fuse.search(searchQuery).map((r) => r.item);
    }
    if (activeTab === "evidence") return evidenceItems;
    if (activeTab === "notes") return notesItems;
    return canvasStates;
  })();

  onMount(() => {
    loki.init();
    refreshData();
  });

  function refreshData() {
    if (activeTab === "evidence") {
      loki.evidence.refreshStore();
    } else if (activeTab === "notes") {
      loki.notes.refreshStore();
    } else {
      // canvas states could be refreshed here if needed
    }
  }

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

  function handleSearch(event: CustomEvent<{ query: string }>) {
    searchQuery = event.detail.query;
  }

  function handleItemClick(item: any) {
    // Forward or handle item click
    console.log("Item clicked:", item);
  }

  function handleTabChange(tab: "evidence" | "notes" | "canvas") {
    activeTab = tab;
    searchQuery = "";
    refreshData();
  }
</script>

<div
  class="yorha-3d-panel nes-legal-container"
  class:open={sidebarOpen}
  bind:this={sidebarElement}
  role="complementary"
  aria-label="Content sidebar"
  on:on:mouseenter={handleMouseEnter}
  on:on:mouseleave={handleMouseLeave}
>
  {#if !sidebarOpen}
    <div class="nes-sidebar-trigger hover-trigger" aria-hidden="true"></div>
  {/if}

  {#if sidebarOpen}
    <div class="yorha-3d-panel-inner neural-sprite-active" transitislide={{ duration: 300, easing: quintOut, axis: "x" }}>
      <div class="nes-legal-header yorha-3d-button">
        <h3 class="nes-legal-title">CONTENT LIBRARY</h3>
        <div class="nes-header-actions">
          <button
            class={`nes-legal-priority-medium yorha-3d-button ${isPinned ? "nes-legal-priority-high" : ""}`}
            on:onclick={togglePin}
            aria-label={isPinned ? "Unpin sidebar" : "Pin sidebar"}
            type="button"
          >
            <Tag size={16} />
          </button>

          {#if !isPinned}
            <button class="nes-legal-priority-low yorha-3d-button" on:onclick={() => (isHovered = false)} aria-label="Close sidebar" type="button">
              <X size={16} />
            </button>
          {/if}
        </div>
      </div>

      <div class="nes-search-section neural-sprite-loading">
        <SearchBar placeholder={`Search ${activeTab}...`} value={searchQuery} search={handleSearch} />
      </div>

      <div class="nes-tabs-container yorha-3d-panel">
        <div class="nes-tab-list">
          <button class="nes-tab-trigger nes-legal-priority-medium" class:active={activeTab === "evidence"} on:onclick={() => handleTabChange("evidence")} type="button">
            <Folder size={16} /> EVIDENCE
          </button>
          <button class="nes-tab-trigger nes-legal-priority-medium" class:active={activeTab === "notes"} on:onclick={() => handleTabChange("notes")} type="button">
            <FileText size={16} /> NOTES
          </button>
          <button class="nes-tab-trigger nes-legal-priority-medium" class:active={activeTab === "canvas"} on:onclick={() => handleTabChange("canvas")} type="button">
            <Tag size={16} /> CANVAS
          </button>
        </div>

        <div class="nes-tab-content neural-sprite-active">
          {#if activeTab === "evidence"}
            <InfiniteScrollList items={searchResults} itemType="evidence" itemClick={handleItemClick} loadMore={refreshData} />
          {:else if activeTab === "notes"}
            <InfiniteScrollList items={searchResults} itemType="notes" itemClick={handleItemClick} loadMore={refreshData} />
          {:else}
            <InfiniteScrollList items={canvasStates} itemType="canvas" itemClick={handleItemClick} loadMore={refreshData} />
          {/if}
        </div>
      </div>

      <div class="nes-tags-section nes-legal-priority-low">
        <TagList />
      </div>
    </div>
  {/if}
</div>

<style>
  /* @unocss-include */
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
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-light);
    box-shadow: 2px 0 8px rgba(0, 0, 0, 0.1);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .sidebar-header {
    padding: 1rem;
    border-bottom: 1px solid var(--border-light);
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--bg-primary);
  }
  .sidebar-header h3 {
    margin: 0;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
  }
  .header-actions {
    display: flex;
    gap: 0.5rem;
  }
  .pin-button.pinned {
    background: var(--bg-secondary);
    color: var(--text-inverse);
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
    color: var(--text-primary);
  }
  .pin-button:hover,
  .close-button:hover {
    background: var(--bg-tertiary);
  }
  .search-section {
    padding: 1rem;
    border-bottom: 1px solid var(--border-light);
  }
  .tab-list {
    display: flex;
    border-bottom: 1px solid var(--border-light);
    background: var(--bg-primary);
  }
  .tab-trigger {
    flex: 1;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem 1rem;
    background: transparent;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.2s ease;
  }
  .tab-trigger:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }
  .tab-trigger.active {
    background: var(--bg-secondary);
    color: var(--text-inverse);
    border-bottom: 2px solid var(--harvard-crimson);
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
    border-top: 1px solid var(--border-light);
    background: var(--bg-primary);
  }
  /* Responsive */
  @media (max-width: 768px) {
    .sidebar-container {
      width: 280px;
    }
  }
  */*
I can do that — quick question before I modify the file:

- Which exact packages are installed (npm names)? e.g. is it "@melt-ui/core" / "@melt-ui/svelte" or "melt" / "melt-ui"? And what's the package name for -bits UI v2 (e.g. "@bits/ui" or "-bits")?
- Which parts do you want replaced with those libraries? (Tabs, buttons/pin, search input, list items, entire layout)
- Do you want the new UI to keep the same behavior (hover-to-open, pin, fuse search, infinite list) or simplify some behaviors?
// Packages to install (frontend)
- melt                       // `npm install melt`
- lucide-svelte              // `npm install lucide-svelte`
- fuse.js                    // `npm install fuse.js`
- xstate & @xstate/svelte    // `npm install xstate @xstate/svelte`

// Packages to install (backend / server-only)
- postgres (postgres-js)     // `npm install postgres`
- drizzle-orm (Postgres adapter) // `npm install drizzle-orm @drizzle-orm/postgres-js`
// Note: postgres-js + drizzle-orm must run on server endpoints / server-side routes (SvelteKit +server.ts), NOT bundled into the frontend.

Which components to swap (recommended)
- Tabs: Melt Tabs (TabList / Tab / TabPanel)
- Buttons: Melt Button / IconButton for pin/close/actions
- Search input: Melt TextField / Input (keep Fuse search wiring)
- List rows: keep your InfiniteScrollList but render items with Melt List/ListItem or simple Melt-styled row components
- TagList: keep or convert to Melt chips/buttons later

Behavior decisions
- Keep current behaviors: hover-to-open, pin/unpin, Fuse search, InfiniteScrollList.
- Use xstate on the frontend to represent auth/login state (so UI can show/hide or change behavior when logged in).
- Keep DB usage (postgres-js + drizzle) on backend endpoints; call them from the frontend via fetch.

Minimal examples

1) Simple auth machine (frontend)
Reply with the package names and which components to swap and I'll produce the exact Svelte 5-compatible code to drop into $SELECTION_PLACEHOLDER$.


/