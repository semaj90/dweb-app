<!-- Production-level AI Dropdown using Melt UI with keyboard shortcuts -->
<script lang="ts">
  import { createDropdownMenu, melt, type CreateDropdownMenuProps } from '@melt-ui/svelte';
  import { fly } from 'svelte/transition';
  import { onMount } from 'svelte';
  import {
    Sparkles,
    FileText,
    Brain,
    Wand2,
    ChevronDown,
    Keyboard,
  } from 'lucide-svelte';

  // Props
  export let disabled = false;
  export let onReportGenerate: (reportType: string) => void = () => {};
  export let onSummarize: () => void = () => {};
  export let onAnalyze: () => void = () => {};
  export let hasContent = false;
  export let isGenerating = false;

  // Melt UI dropdown configuration
  const dropdownConfig: CreateDropdownMenuProps = {
    positioning: {
      placement: 'bottom-start',
      gutter: 8,
    },
    preventScroll: true,
    closeOnEscape: true,
    closeOnOutsideClick: true,
    portal: null, // Keep in document flow for better accessibility
  };

  const {
    elements: { trigger, menu, item, separator },
    states: { open },
    helpers: { isSelected },
  } = createDropdownMenu(dropdownConfig);

  // Report types configuration
  const reportTypes = [
    { 
      id: 'case-summary', 
      name: 'Case Summary Report', 
      icon: FileText, 
      shortcut: 'Ctrl+Shift+C',
      description: 'Comprehensive case overview and analysis'
    },
    { 
      id: 'evidence-analysis', 
      name: 'Evidence Analysis', 
      icon: Brain, 
      shortcut: 'Ctrl+Shift+E',
      description: 'Detailed evidence evaluation and admissibility'
    },
    { 
      id: 'legal-brief', 
      name: 'Legal Brief', 
      icon: Wand2, 
      shortcut: 'Ctrl+Shift+L',
      description: 'Structured legal arguments with precedents'
    },
    { 
      id: 'investigation-report', 
      name: 'Investigation Report', 
      icon: Sparkles, 
      shortcut: 'Ctrl+Shift+I',
      description: 'Investigation documentation and findings'
    },
  ];

  // AI tools configuration
  const aiTools = [
    {
      id: 'summarize',
      name: 'Summarize Content',
      icon: FileText,
      shortcut: 'Ctrl+Shift+S',
      description: 'Generate AI summary of current content',
      requiresContent: true,
    },
    {
      id: 'analyze',
      name: 'Analyze Report',
      icon: Brain,
      shortcut: 'Ctrl+Shift+A',
      description: 'Comprehensive AI analysis with insights',
      requiresContent: true,
    },
  ];

  // Keyboard shortcut handling
  function handleKeydown(event: KeyboardEvent) {
    if (!event.ctrlKey || !event.shiftKey) return;
    
    const key = event.key.toLowerCase();
    
    // Report generation shortcuts
    const reportShortcut = reportTypes.find(type => 
      type.shortcut.toLowerCase().endsWith(key)
    );
    if (reportShortcut && !disabled && !isGenerating) {
      event.preventDefault();
      onReportGenerate(reportShortcut.id);
      $open = false;
      return;
    }

    // AI tool shortcuts
    if (hasContent && !disabled && !isGenerating) {
      switch (key) {
        case 's':
          event.preventDefault();
          onSummarize();
          $open = false;
          break;
        case 'a':
          event.preventDefault();
          onAnalyze();
          $open = false;
          break;
      }
    }
  }

  // Handle item selection
  function handleItemSelect(action: string, requiresContent = false) {
    if (disabled || isGenerating) return;
    if (requiresContent && !hasContent) return;

    switch (action) {
      case 'summarize':
        onSummarize();
        break;
      case 'analyze':
        onAnalyze();
        break;
      default:
        // Report generation
        onReportGenerate(action);
    }
    $open = false;
  }

  onMount(() => {
    document.addEventListener('keydown', handleKeydown);
    return () => document.removeEventListener('keydown', handleKeydown);
  });
</script>

<!-- Trigger Button -->
<button
  use:melt={$trigger}
  class="ai-trigger"
  class:ai-trigger--active={$open}
  class:ai-trigger--disabled={disabled || isGenerating}
  {disabled}
  aria-label="AI Tools Menu"
  title="AI Tools (Press ? for shortcuts)"
>
  <Sparkles size="16" class="ai-trigger__icon" />
  <ChevronDown 
    size="12" 
    class="ai-trigger__chevron" 
    class:ai-trigger__chevron--rotated={$open}
  />
  
  {#if isGenerating}
    <div class="ai-trigger__spinner" aria-hidden="true"></div>
  {/if}
</button>

<!-- Dropdown Menu -->
{#if $open}
  <div
    use:melt={$menu}
    class="ai-menu"
    transition:fly={{ duration: 150, y: -8 }}
  >
    <!-- Report Generation Section -->
    <div class="ai-menu__section">
      <div class="ai-menu__header">
        <FileText size="14" />
        Generate Report
      </div>
      
      {#each reportTypes as reportType}
        <button
          use:melt={$item}
          class="ai-menu__item"
          class:ai-menu__item--selected={$isSelected(reportType.id)}
          on:click={() => handleItemSelect(reportType.id)}
          disabled={disabled || isGenerating}
          data-value={reportType.id}
        >
          <div class="ai-menu__item-content">
            <svelte:component this={reportType.icon} size="14" class="ai-menu__item-icon" />
            <div class="ai-menu__item-text">
              <span class="ai-menu__item-name">{reportType.name}</span>
              <span class="ai-menu__item-description">{reportType.description}</span>
            </div>
          </div>
          <kbd class="ai-menu__shortcut">{reportType.shortcut}</kbd>
        </button>
      {/each}
    </div>

    <!-- Separator -->
    <div use:melt={$separator} class="ai-menu__separator"></div>

    <!-- AI Tools Section -->
    <div class="ai-menu__section">
      <div class="ai-menu__header">
        <Brain size="14" />
        AI Analysis
      </div>
      
      {#each aiTools as tool}
        <button
          use:melt={$item}
          class="ai-menu__item"
          class:ai-menu__item--selected={$isSelected(tool.id)}
          class:ai-menu__item--disabled={tool.requiresContent && !hasContent}
          on:click={() => handleItemSelect(tool.id, tool.requiresContent)}
          disabled={disabled || isGenerating || (tool.requiresContent && !hasContent)}
          data-value={tool.id}
          title={tool.requiresContent && !hasContent ? 'Add content to enable this feature' : ''}
        >
          <div class="ai-menu__item-content">
            <svelte:component this={tool.icon} size="14" class="ai-menu__item-icon" />
            <div class="ai-menu__item-text">
              <span class="ai-menu__item-name">{tool.name}</span>
              <span class="ai-menu__item-description">{tool.description}</span>
            </div>
          </div>
          <kbd class="ai-menu__shortcut">{tool.shortcut}</kbd>
        </button>
      {/each}
    </div>

    <!-- Keyboard Shortcuts Help -->
    <div use:melt={$separator} class="ai-menu__separator"></div>
    <div class="ai-menu__footer">
      <Keyboard size="12" />
      <span class="ai-menu__footer-text">Use keyboard shortcuts or click items</span>
    </div>
  </div>
{/if}

<style>
  /* @unocss-include */
  
  /* Trigger Button */
  .ai-trigger {
    @apply relative flex items-center gap-1 px-3 py-2 text-sm font-medium;
    @apply border border-transparent rounded-md transition-all duration-200;
    @apply bg-gradient-to-r from-purple-50 to-indigo-50;
    @apply text-purple-700 hover:text-purple-800;
    @apply hover:bg-gradient-to-r hover:from-purple-100 hover:to-indigo-100;
    @apply focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-purple-500 focus-visible:ring-offset-2;
    @apply disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-gradient-to-r disabled:hover:from-purple-50 disabled:hover:to-indigo-50;
  }

  .ai-trigger--active {
    @apply bg-gradient-to-r from-purple-100 to-indigo-100;
    @apply border-purple-200 shadow-sm;
  }

  .ai-trigger--disabled {
    @apply opacity-60 cursor-not-allowed;
  }

  .ai-trigger__icon {
    @apply text-purple-600 transition-colors duration-200;
  }

  .ai-trigger__chevron {
    @apply text-purple-500 transition-transform duration-200;
  }

  .ai-trigger__chevron--rotated {
    @apply rotate-180;
  }

  .ai-trigger__spinner {
    @apply absolute inset-0 rounded-md;
    @apply bg-gradient-to-r from-purple-100/80 to-indigo-100/80;
    @apply animate-pulse;
  }

  /* Dropdown Menu */
  .ai-menu {
    @apply min-w-80 max-w-96 bg-white rounded-lg shadow-lg border border-gray-200;
    @apply p-2 z-50 backdrop-blur-sm;
    @apply ring-1 ring-black/5;
  }

  .ai-menu__section {
    @apply space-y-1;
  }

  .ai-menu__header {
    @apply flex items-center gap-2 px-3 py-2 text-xs font-semibold;
    @apply text-gray-500 uppercase tracking-wide;
    @apply border-b border-gray-100 mb-2;
  }

  .ai-menu__item {
    @apply w-full flex items-center justify-between px-3 py-2.5 text-left;
    @apply rounded-md transition-all duration-150;
    @apply hover:bg-gray-50 focus-visible:bg-gray-50;
    @apply focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-purple-500 focus-visible:ring-inset;
    @apply disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-transparent;
  }

  .ai-menu__item--selected {
    @apply bg-purple-50 text-purple-900;
  }

  .ai-menu__item--disabled {
    @apply opacity-40 cursor-not-allowed;
  }

  .ai-menu__item:not(.ai-menu__item--disabled):hover {
    @apply bg-gradient-to-r from-purple-25 to-indigo-25;
  }

  .ai-menu__item-content {
    @apply flex items-center gap-3 flex-1 min-w-0;
  }

  .ai-menu__item-icon {
    @apply text-gray-600 flex-shrink-0;
  }

  .ai-menu__item--selected .ai-menu__item-icon {
    @apply text-purple-600;
  }

  .ai-menu__item-text {
    @apply flex flex-col gap-0.5 min-w-0;
  }

  .ai-menu__item-name {
    @apply text-sm font-medium text-gray-900 truncate;
  }

  .ai-menu__item--selected .ai-menu__item-name {
    @apply text-purple-900;
  }

  .ai-menu__item-description {
    @apply text-xs text-gray-500 line-clamp-1;
  }

  .ai-menu__shortcut {
    @apply px-2 py-1 text-xs font-mono;
    @apply bg-gray-100 text-gray-600 rounded border;
    @apply flex-shrink-0 ml-2;
  }

  .ai-menu__item--selected .ai-menu__shortcut {
    @apply bg-purple-100 text-purple-700 border-purple-200;
  }

  .ai-menu__separator {
    @apply h-px bg-gray-200 my-2;
  }

  .ai-menu__footer {
    @apply flex items-center gap-2 px-3 py-2;
    @apply text-xs text-gray-500;
  }

  .ai-menu__footer-text {
    @apply flex-1;
  }

  /* Yorha Theme Integration */
  :global(.yorha-theme) .ai-trigger {
    @apply bg-gradient-to-r from-yorha-bg-secondary to-yorha-bg-tertiary;
    @apply text-yorha-text-primary border-yorha-border;
    @apply hover:border-yorha-primary hover:text-yorha-primary;
  }

  :global(.yorha-theme) .ai-menu {
    @apply bg-yorha-bg-secondary border-yorha-border;
    @apply shadow-lg shadow-yorha-primary/10;
  }

  :global(.yorha-theme) .ai-menu__item {
    @apply text-yorha-text-primary hover:bg-yorha-bg-tertiary;
  }

  :global(.yorha-theme) .ai-menu__item--selected {
    @apply bg-yorha-primary/20 text-yorha-primary;
  }

  :global(.yorha-theme) .ai-menu__shortcut {
    @apply bg-yorha-bg-tertiary text-yorha-text-secondary border-yorha-border;
  }

  /* Dark mode support */
  @media (prefers-color-scheme: dark) {
    .ai-menu {
      @apply bg-gray-900 border-gray-700;
    }

    .ai-menu__item {
      @apply text-gray-200 hover:bg-gray-800;
    }

    .ai-menu__header {
      @apply text-gray-400 border-gray-700;
    }

    .ai-menu__shortcut {
      @apply bg-gray-800 text-gray-400 border-gray-600;
    }
  }

  /* Reduced motion */
  @media (prefers-reduced-motion: reduce) {
    .ai-trigger,
    .ai-menu__item {
      @apply transition-none;
    }

    .ai-trigger__chevron {
      @apply transition-none;
    }
  }

  /* High contrast mode */
  @media (prefers-contrast: high) {
    .ai-trigger {
      @apply border-gray-800;
    }

    .ai-menu {
      @apply border-gray-800;
    }

    .ai-menu__item--selected {
      @apply bg-gray-900 text-white;
    }
  }
</style>