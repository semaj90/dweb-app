<script lang="ts">

  import { browser } from "$app/environment";
  import { Button } from "$lib/components/ui/button";
  import Tooltip from "$lib/components/ui/Tooltip.svelte";
  import { Accessibility, Keyboard, Maximize2, Minimize2 } from "lucide-svelte";
  import { onMount } from "svelte";
  import AccessibilityPanel from "./AccessibilityPanel.svelte";

  // Keyboard shortcuts state
let showShortcuts = $state(false);
let showAccessibilityPanel = $state(false);
let isFullscreen = $state(false);

  // Keyboard shortcuts map
  const shortcuts = [
    { key: "Ctrl + K", description: "Quick search", action: "search" },
    { key: "Ctrl + N", description: "New evidence", action: "new" },
    { key: "Ctrl + S", description: "Save current work", action: "save" },
    { key: "Ctrl + E", description: "Export data", action: "export" },
    { key: "Ctrl + F", description: "Toggle filters", action: "filter" },
    { key: "Ctrl + H", description: "Show/hide shortcuts", action: "help" },
    {
      key: "Ctrl + Alt + A",
      description: "Accessibility panel",
      action: "accessibility",
    },
    { key: "F11", description: "Toggle fullscreen", action: "fullscreen" },
    { key: "Escape", description: "Close modals/exit", action: "escape" },
  ];

  onMount(() => {
    if (browser) {
      // Add keyboard event listeners
      document.addEventListener("keydown", handleKeyboardShortcut);
      document.addEventListener("fullscreenchange", handleFullscreenChange);

      return () => {
        document.removeEventListener("keydown", handleKeyboardShortcut);
        document.removeEventListener(
          "fullscreenchange",
          handleFullscreenChange
        );
      };
    }
  });

  function handleKeyboardShortcut(event: KeyboardEvent) {
    // Ignore if user is typing in an input/textarea
    if (
      event.target instanceof HTMLInputElement ||
      event.target instanceof HTMLTextAreaElement ||
      event.target instanceof HTMLSelectElement
    ) {
      return;
    }

    const { ctrlKey, metaKey, altKey, key } = event;
    const cmdOrCtrl = ctrlKey || metaKey;

    switch (true) {
      case cmdOrCtrl && key.toLowerCase() === "k":
        event.preventDefault();
        triggerSearch();
        break;
      case cmdOrCtrl && key.toLowerCase() === "n":
        event.preventDefault();
        triggerNewEvidence();
        break;
      case cmdOrCtrl && key.toLowerCase() === "s":
        event.preventDefault();
        triggerSave();
        break;
      case cmdOrCtrl && key.toLowerCase() === "e":
        event.preventDefault();
        triggerExport();
        break;
      case cmdOrCtrl && key.toLowerCase() === "f":
        event.preventDefault();
        triggerFilter();
        break;
      case cmdOrCtrl && key.toLowerCase() === "h":
        event.preventDefault();
        showShortcuts = !showShortcuts;
        break;
      case cmdOrCtrl && altKey && key.toLowerCase() === "a":
        event.preventDefault();
        showAccessibilityPanel = !showAccessibilityPanel;
        break;
      case key === "F11":
        event.preventDefault();
        toggleFullscreen();
        break;
      case key === "Escape":
        handleEscape();
        break;
    }
  }

  function handleFullscreenChange() {
    isFullscreen = !!document.fullscreenElement;
  }

  function triggerSearch() {
    // Focus search input if it exists
    const searchInput = document.querySelector(
      'input[type="search"], input[placeholder*="search" i]'
    ) as HTMLInputElement;
    if (searchInput) {
      searchInput.focus();
      searchInput.select();
    }
    console.log("🔍 Search triggered");
  }

  function triggerNewEvidence() {
    // Trigger new evidence creation
    console.log("➕ New evidence triggered");
    // You would dispatch an event or call a function here
    window.dispatchEvent(new CustomEvent("new-evidence"));
  }

  function triggerSave() {
    // Save current work
    console.log("💾 Save triggered");
    window.dispatchEvent(new CustomEvent("save-work"));
  }

  function triggerExport() {
    // Navigate to export or trigger export
    console.log("📤 Export triggered");
    window.location.href = "/export";
  }

  function triggerFilter() {
    // Toggle filter panel
    console.log("🔧 Filter toggle triggered");
    window.dispatchEvent(new CustomEvent("toggle-filters"));
  }

  function toggleFullscreen() {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
    } else {
      document.exitFullscreen();
    }
  }

  function handleEscape() {
    // Close modals, exit fullscreen, etc.
    if (showShortcuts) {
      showShortcuts = false;
    } else if (showAccessibilityPanel) {
      showAccessibilityPanel = false;
    } else if (document.fullscreenElement) {
      document.exitFullscreen();
    }
    console.log("⚡ Escape triggered");
  }

  export { showShortcuts, toggleFullscreen, isFullscreen };
</script>

<!-- Keyboard Shortcuts Modal -->
{#if showShortcuts}
  <div
    class="mx-auto px-4 max-w-7xl"
    on:onclick={() => (showShortcuts = false)}
    keydown={(e) => e.key === "Escape" && (showShortcuts = false)}
    role="dialog"
    aria-modal="true"
    aria-labelledby="shortcuts-title"
  >
    <div
      class="mx-auto px-4 max-w-7xl"
      on:click|stopPropagation
      role="document"
    >
      <div class="mx-auto px-4 max-w-7xl">
        <h3 id="shortcuts-title" class="mx-auto px-4 max-w-7xl">
          <Keyboard class="mx-auto px-4 max-w-7xl" />
          Keyboard Shortcuts
        </h3>
        <button
          class="mx-auto px-4 max-w-7xl"
          on:onclick={() => (showShortcuts = false)}
          aria-label="Close shortcuts dialog"
        >
          <svg class="mx-auto px-4 max-w-7xl" fill="currentColor" viewBox="0 0 20 20">
            <path
              fill-rule="evenodd"
              d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
              clip-rule="evenodd"
            />
          </svg>
        </button>
      </div>

      <div class="mx-auto px-4 max-w-7xl">
        {#each shortcuts as shortcut}
          <div
            class="mx-auto px-4 max-w-7xl"
          >
            <span class="mx-auto px-4 max-w-7xl">{shortcut.description}</span>
            <kbd
              class="mx-auto px-4 max-w-7xl"
            >
              {shortcut.key}
            </kbd>
          </div>
        {/each}
      </div>

      <div class="mx-auto px-4 max-w-7xl">
        <p class="mx-auto px-4 max-w-7xl">
          💡 Pro tip: These shortcuts work throughout the application to boost
          your productivity!
        </p>
      </div>
    </div>
  </div>
{/if}

<!-- Floating Action Buttons -->
<div class="mx-auto px-4 max-w-7xl">
  <!-- Accessibility Panel Toggle -->
  <Tooltip content="Accessibility panel (Ctrl+Alt+A)" placement="left">
    <Button
      variant="outline"
      size="sm"
      class="mx-auto px-4 max-w-7xl"
      on:on:click={() => (showAccessibilityPanel = !showAccessibilityPanel)}
      aria-label="Toggle accessibility panel"
    >
      <Accessibility class="mx-auto px-4 max-w-7xl" />
    </Button>
  </Tooltip>

  <!-- Keyboard Shortcuts Toggle -->
  <Tooltip content="Keyboard shortcuts (Ctrl+H)" placement="left">
    <Button
      variant="outline"
      size="sm"
      class="mx-auto px-4 max-w-7xl"
      on:on:click={() => (showShortcuts = !showShortcuts)}
      aria-label="Show keyboard shortcuts"
    >
      <Keyboard class="mx-auto px-4 max-w-7xl" />
    </Button>
  </Tooltip>

  <!-- Fullscreen Toggle -->
  <Tooltip
    content="{isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'} (F11)"
    placement="left"
  >
    <Button
      variant="outline"
      size="sm"
      class="mx-auto px-4 max-w-7xl"
      on:on:click={() => toggleFullscreen()}
      aria-label={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
    >
      {#if isFullscreen}
        <Minimize2 class="mx-auto px-4 max-w-7xl" />
      {:else}
        <Maximize2 class="mx-auto px-4 max-w-7xl" />
      {/if}
    </Button>
  </Tooltip>
</div>

<!-- Accessibility Panel -->
<AccessibilityPanel bind:showPanel={showAccessibilityPanel} />

<style>
  :global(.floating-actions) {
    transition: all 0.3s ease;
  }

  :global(.floating-actions:hover) {
    transform: scale(1.05);
  }

  kbd {
    box-shadow:
      0 1px 3px rgba(0, 0, 0, 0.12),
      0 1px 2px rgba(0, 0, 0, 0.24);
  }
</style>


