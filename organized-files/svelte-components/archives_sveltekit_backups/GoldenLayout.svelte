<script lang="ts">
  import { createEventDispatcher } from "svelte";

  export let ratio: "golden" | "thirds" | "half" | "custom" = "golden";
  export let mainFlex = 1.618;
  export let sidebarFlex = 1;
  export let sidebarPosition: "left" | "right" = "right";
  export let collapsible = true;
  export let collapsed = false;
  export let minSidebarWidth = "200px";
  export let maxSidebarWidth = "400px";
  export let gap = "1rem";

  let className = "";
  export { className as class };

  const dispatch = createEventDispatcher();

  // Calculate flex values based on ratio
  let calculatedMainFlex: number;
  let calculatedSidebarFlex: number;

  $: {
    switch (ratio) {
      case "golden":
        calculatedMainFlex = 1.618;
        calculatedSidebarFlex = 1;
        break;
      case "thirds":
        calculatedMainFlex = 2;
        calculatedSidebarFlex = 1;
        break;
      case "half":
        calculatedMainFlex = 1;
        calculatedSidebarFlex = 1;
        break;
      case "custom":
        calculatedMainFlex = mainFlex;
        calculatedSidebarFlex = sidebarFlex;
        break;
    }
  }

  function toggleSidebar() {
    collapsed = !collapsed;
    dispatch("toggle", { collapsed });
  }

  function handleKeydown(e: KeyboardEvent) {
    if (collapsible && (e.ctrlKey || e.metaKey) && e.key === "\\") {
      e.preventDefault();
      toggleSidebar();
    }
  }
</script>

<svelte:window onkeydown={handleKeydown} />

<div
  class="mx-auto px-4 max-w-7xl"
  class:collapsed
  class:sidebar-left={sidebarPosition === "left"}
>
  {#if sidebarPosition === "left"}
    <aside
      class="mx-auto px-4 max-w-7xl"
      class:collapsed
      style="
        flex: {collapsed ? '0' : calculatedSidebarFlex}; 
        min-width: {collapsed ? '0' : minSidebarWidth};
        max-width: {collapsed ? '0' : maxSidebarWidth};
        margin-right: {collapsed ? '0' : gap};
      "
    >
      <div class="mx-auto px-4 max-w-7xl" class:hidden={collapsed}>
        <slot name="sidebar" />
      </div>

      {#if collapsible}
        <button
          class="mx-auto px-4 max-w-7xl"
          onclick={() => toggleSidebar()}
          title={collapsed
            ? "Expand sidebar (Ctrl+\\)"
            : "Collapse sidebar (Ctrl+\\)"}
        >
          {collapsed ? "▶" : "◀"}
        </button>
      {/if}
    </aside>
  {/if}

  <main class="mx-auto px-4 max-w-7xl" style="flex: {calculatedMainFlex};">
    <slot />
  </main>

  {#if sidebarPosition === "right"}
    <aside
      class="mx-auto px-4 max-w-7xl"
      class:collapsed
      style="
        flex: {collapsed ? '0' : calculatedSidebarFlex}; 
        min-width: {collapsed ? '0' : minSidebarWidth};
        max-width: {collapsed ? '0' : maxSidebarWidth};
        margin-left: {collapsed ? '0' : gap};
      "
    >
      <div class="mx-auto px-4 max-w-7xl" class:hidden={collapsed}>
        <slot name="sidebar" />
      </div>

      {#if collapsible}
        <button
          class="mx-auto px-4 max-w-7xl"
          onclick={() => toggleSidebar()}
          title={collapsed
            ? "Expand sidebar (Ctrl+\\)"
            : "Collapse sidebar (Ctrl+\\)"}
        >
          {collapsed ? "◀" : "▶"}
        </button>
      {/if}
    </aside>
  {/if}
</div>

<style>
  .golden-layout {
    display: flex;
    height: 100%;
    min-height: 0;
    transition: all 0.3s ease;
    gap: 0; /* We handle gap with margins for better control */
  }

  .main-content {
    min-width: 0;
    flex-shrink: 1;
    overflow: hidden;
    background: var(--pico-card-background-color, #ffffff);
    border-radius: 0.5rem;
    transition: all 0.3s ease;
  }

  .sidebar {
    position: relative;
    background: var(--pico-card-sectioning-background-color, #f8fafc);
    border: 1px solid var(--pico-border-color, #e2e8f0);
    border-radius: 0.5rem;
    overflow: hidden;
    transition: all 0.3s ease;
    min-width: 0;
  }

  .sidebar.collapsed {
    min-width: 0 !important;
    max-width: 0 !important;
    border-width: 0;
    margin: 0 !important;
  }

  .sidebar-content {
    height: 100%;
    overflow-y: auto;
    overflow-x: hidden;
    padding: 1rem;
    transition: opacity 0.3s ease;
  }

  .sidebar-content.hidden {
    opacity: 0;
    pointer-events: none;
  }

  .sidebar-toggle {
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    background: var(--pico-primary, #3b82f6);
    color: white;
    border: none;
    border-radius: 50%;
    width: 2rem;
    height: 2rem;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    font-size: 0.75rem;
    font-weight: bold;
    transition: all 0.2s ease;
    z-index: 10;
  }

  .sidebar-toggle:hover {
    background: var(--pico-primary-hover, #2563eb);
    transform: translateY(-50%) scale(1.1);
  }

  .sidebar-toggle.left {
    right: -1rem;
  }

  .sidebar-toggle.right {
    left: -1rem;
  }

  .collapsed .sidebar-toggle {
    background: var(--pico-card-background-color, #ffffff);
    color: var(--pico-primary, #3b82f6);
    border: 1px solid var(--pico-border-color, #e2e8f0);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  }

  .collapsed .sidebar-toggle:hover {
    background: var(--pico-primary-background, #f3f4f6);
  }

  /* Responsive design */
  @media (max-width: 768px) {
    .golden-layout {
      flex-direction: column;
    }

    .sidebar {
      order: 2;
      min-width: 100% !important;
      max-width: 100% !important;
      margin: 0 !important;
      margin-top: 1rem !important;
    }

    .sidebar.collapsed {
      margin-top: 0 !important;
      min-width: 0 !important;
      max-width: 0 !important;
    }

    .main-content {
      order: 1;
    }

    .sidebar-toggle {
      top: -1rem;
      left: 50%;
      transform: translateX(-50%);
    }

    .sidebar-toggle.left,
    .sidebar-toggle.right {
      left: 50%;
      right: auto;
      transform: translateX(-50%);
    }

    .collapsed .sidebar-toggle {
      top: 1rem;
      right: 1rem;
      left: auto;
      transform: none;
    }
  }

  /* Smooth scrollbar for sidebar */
  .sidebar-content::-webkit-scrollbar {
    width: 6px;
  }

  .sidebar-content::-webkit-scrollbar-track {
    background: transparent;
  }

  .sidebar-content::-webkit-scrollbar-thumb {
    background: var(--pico-border-color, #e2e8f0);
    border-radius: 3px;
  }

  .sidebar-content::-webkit-scrollbar-thumb:hover {
    background: var(--pico-muted-color, #6b7280);
  }
</style>
