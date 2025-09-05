<script lang="ts">
  interface Props {

  }


  import type { User } from '$lib/types/user';
  import { browser } from "$app/environment";
  import { goto } from "$app/navigation";
  import Button from "$lib/components/ui/button/Button.svelte";
  import { notifications } from "$lib/stores/notification";
  import { FocusManager } from "$lib/utils/accessibility";
  import {
    Command,
    FileText,
    HelpCircle,
    Plus,
    Search,
    Settings,
    Users,
    X,
  } from "lucide-svelte";
  
  
  // Convert to Svelte 5 $props
  let {
    open = $bindable(false),
    shortcutsHelp
  } = $props();

  // Keyboard shortcuts configuration
  const shortcuts = [
    {
      category: "Navigation",
      items: [
        {
          key: "Ctrl+/",
          description: "Open command palette",
          action: () => (open = true),
        },
        {
          key: "Ctrl+H",
          description: "Go to dashboard",
          action: () => goto("/dashboard"),
        },
        {
          key: "Ctrl+Shift+C",
          description: "Go to cases",
          action: () => goto("/cases"),
        },
        {
          key: "Ctrl+Shift+E",
          description: "Go to evidence",
          action: () => goto("/evidence"),
        },
        {
          key: "Ctrl+Shift+P",
          description: "Go to persons",
          action: () => goto("/criminals"),
        },
        {
          key: "Ctrl+Shift+S",
          description: "Go to search",
          action: () => goto("/search"),
        },
        {
          key: "Ctrl+Shift+R",
          description: "Go to reports",
          action: () => goto("/reports"),
        },
        {
          key: "Ctrl+,",
          description: "Open settings",
          action: () => goto("/settings"),
        },
      ],
    },
    {
      category: "Creation",
      items: [
        {
          key: "Ctrl+N",
          description: "New case",
          action: () => goto("/cases/new"),
        },
        {
          key: "Ctrl+Shift+N",
          description: "New person",
          action: () => goto("/criminals/new"),
        },
        {
          key: "Ctrl+U",
          description: "Upload evidence",
          action: () => goto("/evidence"),
        },
        {
          key: "Ctrl+Shift+U",
          description: "Generate report",
          action: () => goto("/reports/new"),
        },
      ],
    },
    {
      category: "Search & Access",
      items: [
        {
          key: "Ctrl+K",
          description: "Quick search",
          action: () => focusSearch(),
        },
        {
          key: "Ctrl+F",
          description: "Find in page",
          action: () => triggerPageSearch(),
        },
        {
          key: "Ctrl+Shift+F",
          description: "Global search",
          action: () => goto("/search"),
        },
        {
          key: "F1",
          description: "Help & documentation",
          action: () => goto("/help"),
        },
        {
          key: "F11",
          description: "Toggle fullscreen",
          action: () => toggleFullscreen(),
        },
      ],
    },
    {
      category: "Interface",
      items: [
        {
          key: "Escape",
          description: "Close modals/overlays",
          action: () => closeModals(),
        },
        {
          key: "Ctrl+Shift+D",
          description: "Toggle dark mode",
          action: () => toggleDarkMode(),
        },
        {
          key: "Ctrl+Shift+L",
          description: "Toggle layout",
          action: () => toggleLayout(),
        },
        {
          key: "Ctrl+R",
          description: "Refresh page",
          action: () => window.location.reload(),
        },
      ],
    },
    {
      category: "Accessibility",
      items: [
        {
          key: "Alt+Shift+H",
          description: "Toggle heading navigation",
          action: () => toggleHeadingNav(),
        },
        {
          key: "Alt+Shift+L",
          description: "Toggle landmark navigation",
          action: () => toggleLandmarkNav(),
        },
        {
          key: "Alt+Shift+F",
          description: "Toggle focus indicators",
          action: () => toggleFocusIndicators(),
        },
        {
          key: "Ctrl+Alt+A",
          description: "Accessibility settings",
          action: () => goto("/settings?tab=accessibility"),
        },
      ],
    },
  ];

  import { keyboardShortcuts, loadShortcutsFromAI } from '$lib/stores';
// Duplicate import removed: onMount is already imported above
  import { get } from 'svelte/store';

  let searchQuery = $state("");
  let selectedIndex = $state(0);
  let filteredShortcuts = $state([]);
  let filteredCommands = $state([]);
  let commandInput: HTMLInputElement

  // Subscribe to keyboardShortcuts store for dynamic/AI-driven shortcuts
  let allShortcuts = $state(get(keyboardShortcuts));
  const unsubscribeShortcuts = keyboardShortcuts.subscribe((shortcuts) => {
    allShortcuts = shortcuts;
    filterShortcuts();
  });

  function filterShortcuts() {
    if (searchQuery.trim()) {
      filteredShortcuts = allShortcuts.filter(
        (s) =>
          s.key.toLowerCase().includes(searchQuery.toLowerCase()) ||
          s.description.toLowerCase().includes(searchQuery.toLowerCase())
      );
    } else {
      filteredShortcuts = allShortcuts;
    }
    selectedIndex = 0;
  }

  // Convert reactive statement to Svelte 5 $effect
  $effect(() => {
    filterShortcuts();
  });

  // Optionally, load AI-driven shortcuts on mount
  onMount(async () => {
    await loadShortcutsFromAI();
  });

  // Command palette items
  const commands = [
    {
      title: "Persons of interest",
      description: "Persons of interest",
      icon: Users,
      action: () => goto("/criminals"),
      keywords: ["people", "suspects"],
    },
    {
      title: "Search",
      description: "Global search",
      icon: Search,
      action: () => goto("/search"),
      keywords: ["find", "lookup"],
    },
    {
      title: "Reports",
      description: "Generate reports",
      icon: FileText,
      action: () => goto("/reports"),
      keywords: ["export", "print"],
    },
    {
      title: "Settings",
      description: "Application settings",
      icon: Settings,
      action: () => goto("/settings"),
      keywords: ["config", "preferences"],
    },
    {
      title: "Help",
      description: "Help & documentation",
      icon: HelpCircle,
      action: () => goto("/help"),
      keywords: ["support", "docs"],
    },
    {
      title: "New Case",
      description: "Create a new case",
      icon: Plus,
      action: () => goto("/cases/new"),
      keywords: ["create", "add"],
    },
    {
      title: "New Person",
      description: "Add person of interest",
      icon: Plus,
      action: () => goto("/criminals/new"),
      keywords: ["create", "add"],
    },
    {
      title: "Upload Evidence",
      description: "Upload evidence files",
      icon: Plus,
      action: () => goto("/evidence"),
      keywords: ["upload", "files"],
    },
    {
      title: "Hash Verification",
      description: "Verify evidence integrity",
      icon: Search,
      action: () => goto("/evidence/hash"),
      keywords: ["integrity", "verify"],
    },
    {
      title: "Analytics",
      description: "View analytics",
      icon: Search,
      action: () => goto("/analytics"),
      keywords: ["stats", "metrics"],
    },
    {
      title: "Canvas Board",
      description: "Evidence canvas",
      icon: Search,
      action: () => goto("/interactive-canvas"),
      keywords: ["board", "visual"],
    },
  ];

  // Convert reactive statement to Svelte 5 $effect
  $effect(() => {
    if (searchQuery.trim()) {
      filteredCommands = commands.filter(
        (cmd) =>
          cmd.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
          cmd.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
          cmd.keywords.some((keyword) =>
            keyword.toLowerCase().includes(searchQuery.toLowerCase())
          )
      );
    } else {
      filteredCommands = commands;
    }
    selectedIndex = 0;
  });
  onMount(() => {
    if (!browser) return;

    const handleKeydown = (event: KeyboardEvent) => {
      // Command palette toggle
      if ((event.ctrlKey || event.metaKey) && event.key === "/") {
        event.preventDefault();
        open = !open;
        return;
}
      // Handle command palette navigation
      if (open) {
        switch (event.key) {
          case "Escape":
            event.preventDefault();
            open = false;
            break;
          case "ArrowDown":
            event.preventDefault();
            selectedIndex = Math.min(
              selectedIndex + 1,
              filteredCommands.length - 1
            );
            break;
          case "ArrowUp":
            event.preventDefault();
            selectedIndex = Math.max(selectedIndex - 1, 0);
            break;
          case "Enter":
            event.preventDefault();
            executeCommand(filteredCommands[selectedIndex]);
            break;
}
        return;
}
      // Global shortcuts
      const shortcut = findShortcut(event);
      if (shortcut) {
        event.preventDefault();
        shortcut.action();
}
    };

    document.addEventListener("keydown", handleKeydown);

    return () => {
      document.removeEventListener("keydown", handleKeydown);
    };
  });

  function findShortcut(event: KeyboardEvent) {
    const key = formatKeyCombo(event);
    return shortcuts
      .flatMap((category) => category.items)
      .find((shortcut) => shortcut.key === key);
}
  function formatKeyCombo(event: KeyboardEvent): string {
    const parts: string[] = [];

    if (event.ctrlKey || event.metaKey) parts.push("Ctrl");
    if (event.altKey) parts.push("Alt");
    if (event.shiftKey) parts.push("Shift");

    // Handle special keys
    const specialKeys: Record<string, string> = {
      " ": "Space",
      Escape: "Escape",
      F1: "F1",
      F11: "F11",
      "/": "/",
      ",": ",",
      Enter: "Enter",
    };

    const key = specialKeys[event.key] || event.key.toUpperCase();
    parts.push(key);

    return parts.join("+");
}
  function executeCommand(command: any) {
    if (!command) return;

    open = false;
    searchQuery = "";

    try {
      command.action();
      notifications.add({
        type: "info",
        title: "Command Executed",
        message: command.title,
      });
    } catch (error) {
      notifications.add({
        type: "error",
        title: "Command Failed",
        message: `Failed to execute: ${command.title}`,
      });
}}
  function focusSearch() {
    const searchInput = document.querySelector(
      'input[type="search"]'
    ) as HTMLInputElement;
    if (searchInput) {
      searchInput.focus();
      searchInput.select();
}}
  function triggerPageSearch() {
    // Trigger browser's find in page
    if (document.execCommand) {
      document.execCommand("find");
}}
  function toggleFullscreen() {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
    } else {
      document.exitFullscreen();
}}
  function closeModals() {
    // Dispatch event to close any open modals
    document.dispatchEvent(new CustomEvent("close-modals"));
    open = false;
}
  function toggleDarkMode() {
    const html = document.documentElement;
    const isDark = html.classList.contains("dark");

    if (isDark) {
      html.classList.remove("dark");
      localStorage.setItem("theme", "light");
    } else {
      html.classList.add("dark");
      localStorage.setItem("theme", "dark");
}
    notifications.add({
      type: "info",
      title: "Theme Changed",
      message: `Switched to ${isDark ? "light" : "dark"} mode`,
    });
}
  function toggleLayout() {
    // This would toggle between different layout modes
    notifications.add({
      type: "info",
      title: "Layout Toggle",
      message: "Layout toggled (feature to be implemented)",
    });
}
  function toggleHeadingNav() {
    // Accessibility feature to navigate by headings
    const headings = Array.from(
      document.querySelectorAll("h1, h2, h3, h4, h5, h6")
    );
    if (headings.length > 0) {
      // Simple implementation - focus first heading
      (headings[0] as HTMLElement).focus();
      FocusManager.announceToScreenReader("Heading navigation enabled");
}}
  function toggleLandmarkNav() {
    // Accessibility feature to navigate by landmarks
    const landmarks = Array.from(
      document.querySelectorAll(
        '[role="main"], [role="navigation"], [role="banner"], [role="contentinfo"]'
      )
    );
    if (landmarks.length > 0) {
      (landmarks[0] as HTMLElement).focus();
      FocusManager.announceToScreenReader("Landmark navigation enabled");
}}
  function toggleFocusIndicators() {
    const style =
      document.getElementById("focus-indicators") ||
      document.createElement("style");
    style.id = "focus-indicators";

    if (style.textContent) {
      style.textContent = "";
      notifications.add({
        type: "info",
        title: "Focus Indicators",
        message: "Enhanced focus indicators disabled",
      });
    } else {
      style.textContent = `
        *:focus {
          outline: 3px solid #3b82f6 !important;
          outline-offset: 2px !important;
}
      `;
      notifications.add({
        type: "info",
        title: "Focus Indicators",
        message: "Enhanced focus indicators enabled",
      });
}
    if (!style.parentNode) {
      document.head.appendChild(style);
    }
  }

  // Focus management for command palette - convert to Svelte 5 $effect
  $effect(() => {
    if (open && commandInput) {
      commandInput.focus();
    }
  });
</script>

<!-- Command Palette Overlay -->
{#if open}
  <div
    class="command-palette-overlay"
    role="dialog"
    tabindex={0}
    aria-modal="true"
    aria-labelledby="command-palette-title"
    onclick={(e) => { if ((e.target as HTMLButtonElement) === e.currentTarget) open = false; }}
    onkeydown={(e) => (e.key === "Escape" ? (open = false) : null)}
  >
    <div class="command-palette">
      <div class="command-palette-header">
        <h2 id="command-palette-title" class="sr-only">Command Palette</h2>
        <div class="search-container">
          <Command class="search-icon" />
          <input
            bind:this={commandInput}
            bind:value={searchQuery}
            type="text"
            placeholder="Type a command or search..."
            class="search-input"
            autocomplete="off"
            spellcheck="false"
            aria-label="Command search"
          />
          <Button
            variant="ghost"
            size="sm"
            onclick={() => (open = false)}
            class="close-button"
            aria-label="Close command palette"
          >
            <X class="w-4 h-4" />
          </Button>
        </div>
      </div>

      <div class="command-palette-body">
        {#if filteredShortcuts.length > 0}
          <ul
            class="command-list"
            role="listbox"
            aria-label="Available commands"
          >
            {#each filteredShortcuts as shortcut, index}
              <li
                class="command-item"
                class:selected={index === selectedIndex}
                role="option"
                aria-selected={index === selectedIndex}
                tabindex={0}
                onclick={() => shortcut.action && shortcut.action()}
                onkeydown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    shortcut.action && shortcut.action();
                  }
                }}
                onmouseenter={() => (selectedIndex = index)}
              >
                <div class="command-content">
                  <div class="command-title flex items-center gap-2">
                    {shortcut.description}
                    {#if shortcut.aiScore !== undefined}
                      <span class="ml-2 px-2 py-0.5 rounded bg-blue-100 text-blue-700 text-xs font-semibold" title="AI Score">AI: {(shortcut.aiScore * 100).toFixed(0)}%</span>
                    {/if}
                  </div>
                  {#if shortcut.aiSummary}
                    <div class="command-description text-xs text-gray-500 mt-1">{shortcut.aiSummary}</div>
                  {/if}
                  <div class="command-key text-xs text-gray-400 mt-1">{shortcut.key}</div>
                </div>
              </li>
            {/each}
          </ul>
        {:else}
          <div class="empty-state">
            <Search class="w-8 h-8" />
            <p>No shortcuts found for "{searchQuery}"</p>
          </div>
        {/if}
      </div>

      <div class="command-palette-footer">
        <div class="footer-hint">
          <kbd>↑↓</kbd> to navigate
          <kbd>Enter</kbd> to select
          <kbd>Esc</kbd> to close
        </div>
      </div>
    </div>
  </div>
{/if}

<!-- Keyboard Shortcuts Help Modal -->
<div class="space-y-4">
  {@render shortcutsHelp?.()}
</div>

<!-- Shortcut definitions for screen readers -->
<div class="space-y-4" aria-live="polite" id="shortcuts-announcements"></div>

<style>
  /* @unocss-include */
  .command-palette-overlay {
    position: fixed
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.6);
    backdrop-filter: blur(4px);
    z-index: 9999;
    display: flex
    align-items: flex-start;
    justify-content: center
    padding-top: 10vh;
    animation: overlay-appear 0.2s ease-out;
}
  @keyframes overlay-appear {
    from {
      opacity: 0;
}
    to {
      opacity: 1;
}}
  .command-palette {
    background: white
    border-radius: 12px;
    box-shadow:
      0 25px 50px -12px rgba(0, 0, 0, 0.25),
      0 0 0 1px rgba(0, 0, 0, 0.05);
    width: 100%;
    max-width: 600px;
    max-height: 70vh;
    display: flex
    flex-direction: column
    animation: palette-appear 0.2s ease-out;
}
  @keyframes palette-appear {
    from {
      opacity: 0;
      transform: scale(0.95) translateY(-10px);
}
    to {
      opacity: 1;
      transform: scale(1) translateY(0);
}}
  .command-palette-header {
    padding: 1rem;
    border-bottom: 1px solid #e5e7eb;
}
  .search-container {
    position: relative
    display: flex
    align-items: center
}
  .search-input {
    width: 100%;
    padding: 0.75rem 1rem 0.75rem 3rem;
    border: none
    outline: none
    font-size: 1rem;
    background: transparent
    color: #111827;
}
  .search-input::placeholder {
    color: #9ca3af;
}
  .close-button {
    position: absolute
    right: 0.5rem;
}
  .command-palette-body {
    flex: 1;
    overflow-y: auto
    max-height: 400px;
}
  .commands-list {
    list-style: none
    padding: 0.5rem 0;
    margin: 0;
}
  .command-item {
    display: flex
    align-items: center
    gap: 0.75rem;
    padding: 0.75rem 1rem;
    cursor: pointer
    transition: background-color 0.1s ease;
}
  .command-item:hover,
  .command-item.selected {
    background: #f3f4f6;
}
  .command-icon {
    display: flex
    align-items: center
    justify-content: center
    width: 32px;
    height: 32px;
    background: #f9fafb;
    border-radius: 6px;
    color: #6b7280;
    flex-shrink: 0;
}
  .command-content {
    flex: 1;
    min-width: 0;
}
  .command-title {
    font-weight: 500;
    color: #111827;
    margin-bottom: 0.125rem;
}
  .command-description {
    font-size: 0.875rem;
    color: #6b7280;
}
  .no-results {
    display: flex
    flex-direction: column
    align-items: center
    justify-content: center
    padding: 3rem 2rem;
    text-align: center
    color: #6b7280;
}
  .no-results p {
    margin: 1rem 0 0 0;
    font-size: 0.875rem;
}
  .command-palette-footer {
    padding: 0.75rem 1rem;
    border-top: 1px solid #e5e7eb;
    background: #f9fafb;
    border-radius: 0 0 12px 12px;
}
  .shortcuts-hint {
    display: flex
    gap: 1rem;
    font-size: 0.75rem;
    color: #6b7280;
    align-items: center
}
  .shortcuts-hint kbd {
    background: #e5e7eb;
    color: #374151;
    padding: 0.125rem 0.375rem;
    border-radius: 0.25rem;
    font-family: inherit
    font-size: 0.75rem;
    font-weight: 500;
}
  .shortcuts-help.hidden {
    display: none
}
  /* Screen reader only content */
  .sr-only {
    position: absolute
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden
    clip: rect(0, 0, 0, 0);
    white-space: nowrap
    border: 0;
}
  /* Dark mode */
  @media (prefers-color-scheme: dark) {
    .command-palette {
      background: #1f2937;
      border: 1px solid #374151;
}
    .command-palette-header {
      border-bottom-color: #374151;
}
    .search-input {
      color: #f9fafb;
}
    .search-input::placeholder {
      color: #9ca3af;
}
    .command-item:hover,
    .command-item.selected {
      background: #374151;
}
    .command-icon {
      background: #374151;
      color: #d1d5db;
}
    .command-title {
      color: #f9fafb;
}
    .command-description {
      color: #d1d5db;
}
    .command-palette-footer {
      background: #374151;
      border-top-color: #4b5563;
}
    .shortcuts-hint kbd {
      background: #4b5563;
      color: #f9fafb;
}}
  /* Responsive design */
  @media (max-width: 640px) {
    .command-palette-overlay {
      padding: 1rem;
      padding-top: 5vh;
}
    .command-palette {
      max-height: 80vh;
}
    .command-item {
      padding: 1rem;
}
    .shortcuts-hint {
      flex-wrap: wrap
      gap: 0.5rem;
}}
  /* High contrast mode */
  @media (prefers-contrast: high) {
    .command-palette {
      border: 2px solid #000;
}
    .command-item.selected {
      background: #000;
      color: #fff;
}}
  /* Reduced motion */
  @media (prefers-reduced-motion: reduce) {
    .command-palette-overlay,
    .command-palette {
      animation: none
}
    .command-item {
      transition: none
}}
</style>
