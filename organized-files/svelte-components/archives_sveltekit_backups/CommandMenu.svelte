<script lang="ts">
  import { goto } from "$app/navigation";
  import { citationStore } from "$lib/stores/citations";
  import { createPopover, melt } from "@melt-ui/svelte";
  import {
    Calendar,
    FileText,
    Hash,
    Search,
    Settings,
    Zap,
  } from "lucide-svelte";
  import { tick } from "svelte";
  import { fly } from "svelte/transition";

  export let triggerText = "#";
  export let placeholder = "Type a command...";
  export let onInsert: (text: string) => void = () => {};
  export let textareaElement: HTMLTextAreaElement | undefined = undefined;

  // Command menu state
  let searchQuery = "";
  let selectedIndex = 0;
  let inputElement: HTMLInputElement;

  // Create popover
  const {
    elements: { trigger, content, arrow, close },
    states: { open },
  } = createPopover({
    positioning: { placement: "bottom-start" },
    forceVisible: true,
    preventScroll: true,
    escapeBehavior: "close",
    closeOnOutsideClick: true,
  });

  // Get recent citations
  $: recentCitations = citationStore.getRecentCitations($citationStore, 5);

  // Available commands
  $: commands = [
    {
      id: "search",
      label: "Search Cases",
      icon: Search,
      action: () => goto("/search"),
      category: "Navigation",
    },
    {
      id: "new-case",
      label: "New Case",
      icon: FileText,
      action: () => goto("/cases/new"),
      category: "Actions",
    },
    {
      id: "recent-cases",
      label: "Recent Cases",
      icon: Calendar,
      action: () => goto("/cases"),
      category: "Navigation",
    },
    {
      id: "evidence",
      label: "Evidence Library",
      icon: FileText,
      action: () => goto("/evidence"),
      category: "Navigation",
    },
    {
      id: "ai-assistant",
      label: "AI Assistant",
      icon: Zap,
      action: () => goto("/ai-assistant"),
      category: "AI",
    },
    {
      id: "settings",
      label: "Settings",
      icon: Settings,
      action: () => goto("/settings"),
      category: "System",
    },
    {
      id: "insert-date",
      label: "Insert Current Date",
      icon: Calendar,
      action: () => insertText(new Date().toLocaleDateString()),
      category: "Text",
    },
    {
      id: "insert-timestamp",
      label: "Insert Timestamp",
      icon: Calendar,
      action: () => insertText(new Date().toISOString()),
      category: "Text",
    },
    // Add recent citations as commands
    ...recentCitations.map((citation) => ({
      id: `citation-${citation.id}`,
      label: `Insert: ${citation.title}`,
      icon: Hash,
      action: () => insertCitation(citation),
      category: "Citations",
    })),
  ];

  // Filter commands based on search query
  $: filteredCommands = commands.filter(
    (cmd) =>
      cmd.label.toLowerCase().includes(searchQuery.toLowerCase()) ||
      cmd.category.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Group commands by category
  $: groupedCommands = filteredCommands.reduce(
    (acc, cmd) => {
      if (!acc[cmd.category]) {
        acc[cmd.category] = [];
}
      acc[cmd.category].push(cmd);
      return acc;
    },
    {} as Record<string, typeof commands>
  );

  // Handle keyboard navigation
  function handleKeydown(e: KeyboardEvent) {
    if (!$open) return;

    const totalCommands = filteredCommands.length;

    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        selectedIndex = (selectedIndex + 1) % totalCommands;
        break;
      case "ArrowUp":
        e.preventDefault();
        selectedIndex =
          selectedIndex === 0 ? totalCommands - 1 : selectedIndex - 1;
        break;
      case "Enter":
        e.preventDefault();
        if (filteredCommands[selectedIndex]) {
          executeCommand(filteredCommands[selectedIndex]);
}
        break;
      case "Escape":
        e.preventDefault();
        // Close the popover
        searchQuery = "";
        open.set(false);
        break;
}}
  function executeCommand(command: (typeof commands)[0]) {
    command.action();
    open.set(false);
    searchQuery = "";
    selectedIndex = 0;
}
  function insertText(text: string) {
    if (textareaElement) {
      const start = textareaElement.selectionStart;
      const end = textareaElement.selectionEnd;
      const currentValue = textareaElement.value;

      // Replace the trigger text with the new text
      const newValue =
        currentValue.substring(0, start - triggerText.length) +
        text +
        currentValue.substring(end);
      textareaElement.value = newValue;

      // Position cursor after inserted text
      const newCursorPos = start - triggerText.length + text.length;
      textareaElement.setSelectionRange(newCursorPos, newCursorPos);

      // Trigger input event
      textareaElement.dispatchEvent(new Event("input", { bubbles: true }));
}
    onInsert(text);
}
  function insertCitation(citation?: unknown) {
    if (citation) {
      // Format the citation properly
      const formattedCitation = `[${citation.title}${citation.source ? `, ${citation.source}` : ""}${citation.date ? ` (${citation.date})` : ""}]`;
      insertText(formattedCitation);

      // Mark as recently used
      citationStore.markAsRecentlyUsed(citation.id);
    } else {
      // Generic citation placeholder
      const citation = "[Citation: Document Title, Source (Year)]";
      insertText(citation);
}}
  // Open command menu
  export function openCommandMenu() {
    open.set(true);
    tick().then(() => {
      inputElement?.focus();
    });
}
  // Reset when closing
  $: if (!$open) {
    searchQuery = "";
    selectedIndex = 0;
}
</script>

<svelte:window onkeydown={handleKeydown} />

<!-- Hidden trigger (we'll open programmatically) -->
<button use:melt={$trigger} style="display: none;">Trigger</button>

{#if $open}
  <div
    use:melt={$content}
    class="mx-auto px-4 max-w-7xl"
    transition:fly={{ y: -10, duration: 150 "
  >
    <div class="mx-auto px-4 max-w-7xl">
      <Search size={16} />
      <input
        bind:this={inputElement}
        bind:value={searchQuery}
        {placeholder}
        class="mx-auto px-4 max-w-7xl"
        autocomplete="off"
        spellcheck="false"
      />
    </div>

    <div class="mx-auto px-4 max-w-7xl">
      {#each Object.entries(groupedCommands) as [category, categoryCommands], categoryIndex}
        <div class="mx-auto px-4 max-w-7xl">
          <div class="mx-auto px-4 max-w-7xl">{category}</div>
          {#each categoryCommands as command, commandIndex}
            {@const globalIndex = filteredCommands.findIndex(
              (c) => c.id === command.id
            )}
            <button
              class="mx-auto px-4 max-w-7xl"
              class:selected={globalIndex === selectedIndex}
              onclick={() => executeCommand(command)}
              onmouseenter={() => (selectedIndex = globalIndex)}
            >
              <svelte:component this={command.icon} size={16} />
              <span class="mx-auto px-4 max-w-7xl">{command.label}</span>
            </button>
          {/each}
        </div>
      {/each}

      {#if filteredCommands.length === 0}
        <div class="mx-auto px-4 max-w-7xl">
          <Search size={24} />
          <p>No commands found for "{searchQuery}"</p>
        </div>
      {/if}
    </div>

    <div class="mx-auto px-4 max-w-7xl">
      <div class="mx-auto px-4 max-w-7xl">
        <kbd>↑↓</kbd> Navigate
        <kbd>Enter</kbd> Select
        <kbd>Esc</kbd> Close
      </div>
    </div>
  </div>
{/if}

<style>
  .command-menu {
    background: var(--pico-card-background-color, #ffffff);
    border: 1px solid var(--pico-border-color, #e2e8f0);
    border-radius: 0.75rem;
    box-shadow:
      0 20px 25px -5px rgba(0, 0, 0, 0.1),
      0 10px 10px -5px rgba(0, 0, 0, 0.04);
    width: 400px;
    max-height: 400px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    z-index: 1000;
}
  .command-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 1rem;
    border-bottom: 1px solid var(--pico-border-color, #e2e8f0);
    background: var(--pico-card-sectioning-background-color, #f8fafc);
}
  .command-input {
    flex: 1;
    border: none;
    outline: none;
    background: transparent;
    font-size: 1rem;
    color: var(--pico-color, #111827);
}
  .command-input::placeholder {
    color: var(--pico-muted-color, #6b7280);
}
  .command-results {
    flex: 1;
    overflow-y: auto;
    padding: 0.5rem;
}
  .command-category {
    margin-bottom: 0.75rem;
}
  .command-category:last-child {
    margin-bottom: 0;
}
  .category-header {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--pico-muted-color, #6b7280);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0.5rem 0.75rem 0.25rem;
    margin-bottom: 0.25rem;
}
  .command-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    width: 100%;
    padding: 0.75rem;
    border: none;
    background: transparent;
    border-radius: 0.5rem;
    cursor: pointer;
    transition: all 0.15s ease;
    text-align: left;
}
  .command-item:hover,
  .command-item.selected {
    background: var(--pico-primary-background, #f3f4f6);
    color: var(--pico-primary, #3b82f6);
}
  .command-label {
    font-size: 0.875rem;
    font-weight: 500;
}
  .command-footer {
    border-top: 1px solid var(--pico-border-color, #e2e8f0);
    padding: 0.75rem;
    background: var(--pico-card-sectioning-background-color, #f8fafc);
}
  .command-shortcuts {
    display: flex;
    align-items: center;
    gap: 1rem;
    font-size: 0.75rem;
    color: var(--pico-muted-color, #6b7280);
}
  .command-shortcuts kbd {
    background: var(--pico-card-background-color, #ffffff);
    border: 1px solid var(--pico-border-color, #e2e8f0);
    border-radius: 0.25rem;
    padding: 0.125rem 0.375rem;
    font-size: 0.625rem;
    font-weight: 600;
    color: var(--pico-color, #111827);
}
  .no-results {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 2rem;
    text-align: center;
    color: var(--pico-muted-color, #6b7280);
}
  .no-results p {
    margin: 0.5rem 0 0;
    font-size: 0.875rem;
}
  /* Scrollbar styling */
  .command-results::-webkit-scrollbar {
    width: 6px;
}
  .command-results::-webkit-scrollbar-track {
    background: transparent;
}
  .command-results::-webkit-scrollbar-thumb {
    background: var(--pico-border-color, #e2e8f0);
    border-radius: 3px;
}
  .command-results::-webkit-scrollbar-thumb:hover {
    background: var(--pico-muted-color, #6b7280);
}
</style>