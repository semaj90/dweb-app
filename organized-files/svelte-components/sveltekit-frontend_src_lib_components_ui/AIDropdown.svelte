<!-- Production-level AI Dropdown using Melt UI with keyboard shortcuts -->
<script lang="ts">
  interface Props { disabled?: any;
    onReportGenerate: (reportType: string) => void;,
    onSummarize: () ;,
    onAnalyze: () ;,
    hasContent?: any;
    isGenerating?: any;
   }
  let { disabled = false,
    onReportGenerate = > void = () => { },
    onSummarize = > void = () => {},
    onAnalyze = > void = () => {},
    hasContent = false,
    isGenerating = false
  } = $props();



  import { createDropdownMenu,
    melt,
    type CreateDropdownMenuProps,
   } from "@melt-ui/svelte";
  import { Brain,
    ChevronDown,
    FileText,
    Keyboard,
    Sparkles,
    Wand2,
   } from "lucide-svelte";
  import { onMount  } from "svelte";
  import { fly  } from "svelte/transition";

  // Props
   // false;
  let { onReportGenerate = $bindable()  } = $props(); // (reportType: string) => void = () => {};
  let { onSummarize = $bindable()  } = $props(); // () => void = () => {};
  let { onAnalyze = $bindable()  } = $props(); // () => void = () => {};
  let { hasContent = $bindable()  } = $props(); // false;
  let { isGenerating = $bindable()  } = $props(); // false;

  // Melt UI dropdown configuration
  const dropdownConfig: CreateDropdownMenuProps = { positioning: {,
      placement: "bottom-start",
      gutter: 8,
     },
    preventScroll: true,
    portal: null, // Keep in document flow for better accessibility
  };

  const { elements: {, trigger, menu, item, separator  },
    states: { open  },
  } = createDropdownMenu(dropdownConfig);

  // Track selected item
  let selectedItem: string | null = null;

  // Report types configuration
  const reportTypes = [
    { id: "case-summary",
      name: "Case Summary, Report",
      icon: FileText,
      shortcut: "Ctrl+Shift+C",
      description: "Comprehensive case overview and, analysis",
     },
    { id: "evidence-analysis",
      name: "Evidence, Analysis",
      icon: Brain,
      shortcut: "Ctrl+Shift+E",
      description: "Detailed evidence evaluation and, admissibility",
     },
    { id: "legal-brief",
      name: "Legal, Brief",
      icon: Wand2,
      shortcut: "Ctrl+Shift+L",
      description: "Structured legal arguments with, precedents",
     },
    { id: "investigation-report",
      name: "Investigation, Report",
      icon: Sparkles,
      shortcut: "Ctrl+Shift+I",
      description: "Investigation documentation and, findings",
     },
  ];

  // AI tools configuration
  const aiTools = [
    { id: "summarize",
      name: "Summarize, Content",
      icon: FileText,
      shortcut: "Ctrl+Shift+S",
      description: "Generate AI summary of current, content",
      requiresContent: true,
     },
    { id: "analyze",
      name: "Analyze, Report",
      icon: Brain,
      shortcut: "Ctrl+Shift+A",
      description: "Comprehensive AI analysis with, insights",
      requiresContent: true,
     },
  ];

  // Keyboard shortcut handling
  function handleKeydown(event: KeyboardEvent) { if (!event.ctrlKey || !event.shiftKey) return;

    const key = event.key.toLowerCase();

    // Report generation shortcuts
    const reportShortcut = reportTypes.find((type) =>
      type.shortcut.toLowerCase().endsWith(key)
    );
    if (reportShortcut && !disabled && !isGenerating) {
      event.preventDefault();
      onReportGenerate(reportShortcut.id);
      $open = false;
      return;
     }

    // AI tool shortcuts
    if (hasContent && !disabled && !isGenerating) { switch (key) {
        case "s":
          event.preventDefault();
          onSummarize();
          $open = false;
          break;
        case "a":
          event.preventDefault();
          onAnalyze();
          $open = false;
          break;
       }
    }
  }

  // Handle item selection
  function handleItemSelect(action: string, requiresContent = false) { if (disabled || isGenerating) return;
    if (requiresContent && !hasContent) return;

    selectedItem = action;

    switch (action) {
      case "summarize":
        onSummarize();
        break;
      case "analyze":
        onAnalyze();
        break;
      default:
        // Report generation,
        onReportGenerate(action);
     }
    $open = false;
  }

  onMount(() => { document.addEventListener("keydown", handleKeydown);
    return () => document.removeEventListener("keydown", handleKeydown);
   });
</script>

<!-- Trigger Button -->
<button
  use:melt={ $trigger }
  class="ai-trigger"
  class:ai-trigger--active={ $open }
  class:ai-trigger--disabled={ disabled || isGenerating }
  { disabled }
  aria-label="AI Tools Menu"
  title="AI Tools (Press ? for shortcuts)"
>
  <Sparkles size={ 16 } class="ai-trigger__icon" />
  <ChevronDown
    size={ 12 }
    class="ai-trigger__chevron { $open ? 'ai-trigger__chevron--rotated' : '' }"
  />

  { #if isGenerating }
    <div class="ai-trigger__spinner" aria-hidden="true"></div>
  { /if }
</button>

<!-- Dropdown Menu -->
{ #if $open }
  <div
    use:melt={ $menu }
    class="ai-menu"
    transition:fly={ { duration: 150, y: -8  }}
  >
    <!-- Report Generation Section -->
    <div class="ai-menu__section">
      <div class="ai-menu__header">
        <FileText size={ 14 } />
        Generate Report
      </div>

      { #each reportTypes as reportType }
        <button
          use:melt={ $item }
          class="ai-menu__item"
          class:ai-menu__item--selected={ selectedItem === reportType.id }
          on:click={ () => handleItemSelect(reportType.id) }
          disabled={ disabled || isGenerating }
          data-value={ reportType.id }
        >
          <div class="ai-menu__item-content">
            <svelte:component
              this={ reportType.icon }
              size={ 14 }
              class="ai-menu__item-icon"
            />
            <div class="ai-menu__item-text">
              <span class="ai-menu__item-name">{ reportType.name }</span>
              <span class="ai-menu__item-description"
                >{ reportType.description }</span
              >
            </div>
          </div>
          <kbd class="ai-menu__shortcut">{ reportType.shortcut }</kbd>
        </button>
      { /each }
    </div>

    <!-- Separator -->
    <div use:melt={ $separator } class="ai-menu__separator"></div>

    <!-- AI Tools Section -->
    <div class="ai-menu__section">
      <div class="ai-menu__header">
        <Brain size={ 14 } />
        AI Analysis
      </div>

      { #each aiTools as tool }
        <button
          use:melt={ $item }
          class="ai-menu__item"
          class:ai-menu__item--selected={ selectedItem === tool.id }
          class:ai-menu__item--disabled={ tool.requiresContent && !hasContent }
          on:click={ () => handleItemSelect(tool.id, tool.requiresContent) }
          disabled={ disabled ||
            isGenerating ||
            (tool.requiresContent && !hasContent) }
          data-value={ tool.id }
          title={ tool.requiresContent && !hasContent
            ? "Add content to enable this feature"
            : "" }
        >
          <div class="ai-menu__item-content">
            <svelte:component
              this={ tool.icon }
              size={ 14 }
              class="ai-menu__item-icon"
            />
            <div class="ai-menu__item-text">
              <span class="ai-menu__item-name">{ tool.name }</span>
              <span class="ai-menu__item-description">{ tool.description }</span>
            </div>
          </div>
          <kbd class="ai-menu__shortcut">{ tool.shortcut }</kbd>
        </button>
      { /each }
    </div>

    <!-- Keyboard Shortcuts Help -->
    <div use:melt={ $separator } class="ai-menu__separator"></div>
    <div class="ai-menu__footer">
      <Keyboard size={ 12 } />
      <span class="ai-menu__footer-text"
        >Use keyboard shortcuts or click items</span
      >
    </div>
  </div>
{ /if }

<style>
  /* @unocss-include */

  /* Trigger Button */
  .ai-trigger { position: relative; display: flex,
    align-items: center; gap: 0.25rem;,
    padding: 0.5rem 0.75rem;,
    font-size: 0.875rem;,
    font-weight: 500;,
    border: 1px solid transparent;,
    border-radius: 0.375rem;,
    transition: all 0.2s;,
    background: linear-gradient(to, right, #faf5ff, #eef2ff);
    color: #7c3aed;
   }

  .ai-trigger:hover { color: #6b21a8;,
    background: linear-gradient(to, right, #f3e8ff, #e0e7ff);
   }

  .ai-trigger:focus-visible { outline: none; ring: 2px;,
    ring-color: #8b5cf6;,
    ring-offset: 2px;
   }

  .ai-trigger:disabled { opacity: 0.5;,
    cursor: not-allowed;
   }

  .ai-trigger:disabled:hover { background: linear-gradient(to, right, #faf5ff, #eef2ff);
   }

  .ai-trigger--active { background: linear-gradient(to, right, #f3e8ff, #e0e7ff);
    border-color: #d8b4fe;,
    box-shadow: 0 1px 2px 0, rgba(0, 0, 0, 0.05);
   }

  .ai-trigger--disabled { opacity: 0.6;,
    cursor: not-allowed;
   }

  .ai-trigger__icon { color: #9333ea;,
    transition: color 0.2s;
   }

  .ai-trigger__chevron { color: #a855f7;,
    transition: transform 0.2s;
   }

  .ai-trigger__chevron--rotated { transform: rotate(180deg);
   }

  .ai-trigger__spinner { position: absolute; inset: 0;,
    border-radius: 0.375rem;,
    background: linear-gradient(,
      to right,
      rgba(233, 213, 255, 0.8),
      rgba(224, 231, 255, 0.8)
    );
    animation: pulse 2s, cubic-bezier(0.4, 0, 0.6, 1) infinite;
   }

  /* Dropdown Menu */
  .ai-menu { min-width: 20rem;,
    max-width: 24rem;,
    background-color: white,
    border-radius: 0.5rem;,
    box-shadow:
      0 10px 15px -3px, rgba(0, 0, 0, 0.1),
      0 4px 6px -2px rgba(0, 0, 0, 0.05);
    border: 1px solid #e5e7eb;,
    padding: 0.5rem;,
    z-index: 50;,
    backdrop-filter: blur(4px);,
    ring: 1px;,
    ring-color: rgba(0, 0, 0, 0.05);
   }

  .ai-menu__section { display: flex,
    flex-direction: column;, gap: 0.25rem;
   }

  .ai-menu__header { display: flex,
    align-items: center; gap: 0.5rem;,
    padding: 0.5rem 0.75rem;,
    font-size: 0.75rem;,
    font-weight: 600;,
    color: #6b7280;,
    text-transform: uppercase,
    letter-spacing: 0.05em;,
    border-bottom: 1px solid #f3f4f6;,
    margin-bottom: 0.5rem;
   }

  .ai-menu__item { width: 100%;,
    display: flex,
    align-items: center,
    justify-content: space-between;,
    padding: 0.625rem 0.75rem;,
    text-align: left,
    border-radius: 0.375rem;,
    transition: all 0.15s;
   }

  .ai-menu__item:hover,
  .ai-menu__item:focus-visible { background-color: #f9fafb;
   }

  .ai-menu__item:focus-visible { outline: none; ring: 2px;,
    ring-color: #8b5cf6;,
    ring-inset: true }

  .ai-menu__item:disabled { opacity: 0.5;,
    cursor: not-allowed;
   }

  .ai-menu__item:disabled:hover { background-color: transparent }

  .ai-menu__item--selected { background-color: #faf5ff;,
    color: #581c87;
   }

  .ai-menu__item--disabled { opacity: 0.4;,
    cursor: not-allowed;
   }

  .ai-menu__item:not(.ai-menu__item--disabled):hover { background: linear-gradient(to, right, #fefcff, #fefcff);
   }

  .ai-menu__item-content { display: flex,
    align-items: center; gap: 0.75rem;,
    flex: 1;,
    min-width: 0;
   }

  .ai-menu__item-icon { color: #4b5563;,
    flex-shrink: 0;
   }

  .ai-menu__item--selected .ai-menu__item-icon { color: #9333ea;
   }

  .ai-menu__item-text { display: flex,
    flex-direction: column; gap: 0.125rem;,
    min-width: 0;
   }

  .ai-menu__item-name { font-size: 0.875rem;,
    font-weight: 500;,
    color: #111827;,
    overflow: hidden,
    text-overflow: ellipsis,
    white-space: nowrap }

  .ai-menu__item--selected .ai-menu__item-name { color: #581c87;
   }

  .ai-menu__item-description { font-size: 0.75rem;,
    color: #6b7280;,
    display: -webkit-box;
    -webkit-line-clamp: 1;
    -webkit-box-orient: vertical; overflow:, hidden
   }

  .ai-menu__shortcut { padding: 0.25rem 0.5rem;,
    font-size: 0.75rem;,
    font-family: ui-monospace, SFMono-Regular, monospace;
    background-color: #f3f4f6;,
    color: #4b5563;,
    border-radius: 0.25rem;,
    border: 1px solid #d1d5db;,
    flex-shrink: 0;,
    margin-left: 0.5rem;
   }

  .ai-menu__item--selected .ai-menu__shortcut { background-color: #f3e8ff;,
    color: #6b21a8;,
    border-color: #d8b4fe;
   }

  .ai-menu__separator { height: 1px;,
    background-color: #e5e7eb;,
    margin: 0.5rem 0;
   }

  .ai-menu__footer { display: flex,
    align-items: center; gap: 0.5rem;,
    padding: 0.5rem 0.75rem;,
    font-size: 0.75rem;,
    color: #6b7280;
   }

  .ai-menu__footer-text { flex: 1;
   }

  /* Yorha Theme Integration */
  :global(.yorha-theme) .ai-trigger { background: linear-gradient(,
      to right,
      var(--yorha-bg-secondary),
      var(--yorha-bg-tertiary)
    );
    color: var(--yorha-text-primary);,
    border-color: var(--yorha-border);
   }

  :global(.yorha-theme) .ai-trigger:hover { border-color: var(--yorha-primary);,
    color: var(--yorha-primary);
   }

  :global(.yorha-theme) .ai-menu { background-color: var(--yorha-bg-secondary);,
    border-color: var(--yorha-border);,
    box-shadow: 0 10px 15px -3px, rgba(var(--yorha-primary-rgb), 0.1);
   }

  :global(.yorha-theme) .ai-menu__item { color: var(--yorha-text-primary);
   }

  :global(.yorha-theme) .ai-menu__item:hover { background-color: var(--yorha-bg-tertiary);
   }

  :global(.yorha-theme) .ai-menu__item--selected { background-color: rgba(var(--yorha-primary-rgb), 0.2);
    color: var(--yorha-primary);
   }

  :global(.yorha-theme) .ai-menu__shortcut { background-color: var(--yorha-bg-tertiary);,
    color: var(--yorha-text-secondary);,
    border-color: var(--yorha-border);
   }

  /* Dark mode support */
  @media (prefers-color-scheme: dark) { .ai-menu {
      background-color: #111827;,
      border-color: #374151;
     }

    .ai-menu__item { color: #e5e7eb;
     }

    .ai-menu__item:hover { background-color: #1f2937;
     }

    .ai-menu__header { color: #9ca3af;,
      border-color: #374151;
     }

    .ai-menu__shortcut { background-color: #1f2937;,
      color: #9ca3af;,
      border-color: #4b5563;
     }
  }

  /* Reduced motion */
  @media (prefers-reduced-motion: reduce) { .ai-trigger,
    .ai-menu__item {
      transition: none }

    .ai-trigger__chevron { transition: none }
  }

  /* High contrast mode */
  @media (prefers-contrast: high) { .ai-trigger {
      border-color: #1f2937;
     }

    .ai-menu { border-color: #1f2937;
     }

    .ai-menu__item--selected { background-color: #111827;,
      color: white }
  }
</style>
