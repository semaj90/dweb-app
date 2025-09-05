<script lang="ts">
  import { browser } from "$app/environment";
  import { Button } from "$lib/components/ui/button";
  import Tooltip from "$lib/components/ui/Tooltip.svelte";
  import { Accessibility, Keyboard, Maximize2, Minimize2 } from "lucide-svelte";
  import { onMount } from "svelte";
  import AccessibilityPanel from "./AccessibilityPanel.svelte";

  // Keyboard shortcuts state
  let showShortcuts = false;
  let showAccessibilityPanel = false;
  let isFullscreen = false;

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
    class="shortcuts-dialog"
    onclick={() => (showShortcuts = false)}
    onkeydown={(e) => e.key === "Escape" && (showShortcuts = false)}
    role="dialog"
    aria-modal="true"
    aria-labelledby="shortcuts-title"
  >
    <div
      class="shortcuts-dialog__document"
      on:click|stopPropagation
      role="document"
    >
      <div class="shortcuts-dialog__header">
        <h3 id="shortcuts-title" class="shortcuts-dialog__title">
          <Keyboard class="shortcuts-dialog__icon" />
          Keyboard Shortcuts
        </h3>
        <button
          class="shortcuts-dialog__close"
          onclick={() => (showShortcuts = false)}
          aria-label="Close shortcuts dialog"
        >
          <svg class="shortcuts-dialog__close-icon" fill="currentColor" viewBox="0 0 20 20">
            <path
              fill-rule="evenodd"
              d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
              clip-rule="evenodd"
            />
          </svg>
        </button>
      </div>

      <div class="shortcuts-dialog__list">
        {#each shortcuts as shortcut}
          <div class="shortcuts-dialog__item">
            <span class="shortcuts-dialog__desc">{shortcut.description}</span>
            <kbd class="shortcuts-dialog__key">{shortcut.key}</kbd>
          </div>
        {/each}
      </div>

      <div class="shortcuts-dialog__footer">
        <p class="shortcuts-dialog__tip">
          💡 Pro tip: These shortcuts work throughout the application to boost your productivity!
        </p>
      </div>
    </div>
  </div>
{/if}

<!-- Floating Action Buttons -->
<div class="floating-actions">
  <!-- Accessibility Panel Toggle -->
  <Tooltip content="Accessibility panel (Ctrl+Alt+A)" placement="left">
    <Button
      variant="outline"
      size="sm"
      class="floating-actions__btn"
      onclick={() => (showAccessibilityPanel = !showAccessibilityPanel)}
      aria-label="Toggle accessibility panel"
    >
      <Accessibility class="floating-actions__icon" />
    </Button>
  </Tooltip>

  <!-- Keyboard Shortcuts Toggle -->
  <Tooltip content="Keyboard shortcuts (Ctrl+H)" placement="left">
    <Button
      variant="outline"
      size="sm"
      class="floating-actions__btn"
      onclick={() => (showShortcuts = !showShortcuts)}
      aria-label="Show keyboard shortcuts"
    >
      <Keyboard class="floating-actions__icon" />
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
      class="floating-actions__btn"
      onclick={() => toggleFullscreen()}
      aria-label={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
    >
      {#if isFullscreen}
        <Minimize2 class="floating-actions__icon" />
      {:else}
        <Maximize2 class="floating-actions__icon" />
      {/if}
    </Button>
  </Tooltip>
</div>

<!-- Accessibility Panel -->
<AccessibilityPanel bind:showPanel={showAccessibilityPanel} />

<style>
  /* @unocss-include */
  :global(.floating-actions) {
    display: flex
    gap: 0.5rem;
    align-items: center
    transition: all 0.3s ease;
  }
  :global(.floating-actions__btn) {
    transition: box-shadow 0.2s;
  }
  :global(.floating-actions__btn:hover) {
    box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    z-index: 2;
  }
  :global(.floating-actions__icon) {
    width: 1.25rem;
    height: 1.25rem;
    vertical-align: middle
  }
  kbd {
    box-shadow:
      0 1px 3px rgba(0, 0, 0, 0.12),
      0 1px 2px rgba(0, 0, 0, 0.24);
  }

  /* Modal styles for keyboard shortcuts dialog */
  :global(.shortcuts-dialog) {
    position: fixed
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: rgba(0,0,0,0.35);
    z-index: 1000;
    display: flex
    align-items: center
    justify-content: center
    transition: background 0.2s;
  }
  :global(.shortcuts-dialog__document) {
    background: #fff;
    border-radius: 0.75rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.18);
    max-width: 420px;
    width: 100%;
    padding: 1.5rem 1.25rem;
    outline: none
    display: flex
    flex-direction: column
    gap: 1.25rem;
  }
  :global(.shortcuts-dialog__header) {
    display: flex
    align-items: center
    justify-content: space-between;
    gap: 0.5rem;
  }
  :global(.shortcuts-dialog__title) {
    font-size: 1.25rem;
    font-weight: 600;
    display: flex
    align-items: center
    gap: 0.5rem;
    margin: 0;
  }
  :global(.shortcuts-dialog__icon) {
    width: 1.2em;
    height: 1.2em;
    color: #555;
  }
  :global(.shortcuts-dialog__close) {
    background: none
    border: none
    cursor: pointer
    padding: 0.25rem;
    border-radius: 0.25rem;
    transition: background 0.15s;
  }
  :global(.shortcuts-dialog__close:hover),
  :global(.shortcuts-dialog__close:focus) {
    background: #f2f2f2;
    outline: 2px solid #b3b3b3;
  }
  :global(.shortcuts-dialog__close-icon) {
    width: 1.1em;
    height: 1.1em;
    color: #888;
  }
  :global(.shortcuts-dialog__list) {
    display: flex
    flex-direction: column
    gap: 0.75rem;
  }
  :global(.shortcuts-dialog__item) {
    display: flex
    align-items: center
    justify-content: space-between;
    gap: 1rem;
    font-size: 1rem;
  }
  :global(.shortcuts-dialog__desc) {
    color: #333;
    font-size: 1rem;
  }
  :global(.shortcuts-dialog__key) {
    background: #f5f5f5;
    border-radius: 0.3em;
    padding: 0.2em 0.7em;
    font-family: inherit
    font-size: 0.98em;
    color: #222;
    border: 1px solid #e0e0e0;
  }
  :global(.shortcuts-dialog__footer) {
    margin-top: 0.5rem;
    text-align: center
  }
  :global(.shortcuts-dialog__tip) {
    color: #888;
    font-size: 0.98rem;
  }
</style>
