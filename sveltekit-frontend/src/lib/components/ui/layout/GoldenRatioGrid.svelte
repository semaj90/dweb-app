<script lang="ts">
  import { cn } from '$lib/utils/cn';

  interface GoldenRatioGridProps {
    /** Grid layout variant */
    variant?: 'sidebar' | 'dashboard' | 'split' | 'content' | 'legal-document';
    /** Golden ratio direction */
    direction?: 'horizontal' | 'vertical' | 'both';
    /** Container size */
    size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
    /** Enable responsive behavior */
    responsive?: boolean;
    /** Legal context styling */
    legal?: boolean;
    /** Evidence analysis layout */
    evidenceLayout?: boolean;
    /** Case management layout */
    caseLayout?: boolean;
    /** AI analysis panels */
    aiPanels?: boolean;
    /** Custom gap between grid items */
    gap?: 'none' | 'xs' | 'sm' | 'md' | 'lg' | 'xl';
    /** Custom class */
    class?: string;
    /** Main content slot */
    children?: import('svelte').Snippet;
    /** Sidebar content */
    sidebar?: import('svelte').Snippet;
    /** Header content */
    header?: import('svelte').Snippet;
    /** Footer content */
    footer?: import('svelte').Snippet;
    /** Secondary content */
    secondary?: import('svelte').Snippet;
  }

  let {
    variant = 'dashboard',
    direction = 'horizontal',
    size = 'full',
    responsive = true,
    legal = false,
    evidenceLayout = false,
    caseLayout = false,
    aiPanels = false,
    gap = 'md',
    class: className = '',
    children,
    sidebar,
    header,
    footer,
    secondary
  }: GoldenRatioGridProps = $props();

  // Golden ratio constant (φ ≈ 1.618)
  const PHI = 1.618033988749;
  const INVERSE_PHI = 0.618033988749; // 1/φ

  // Reactive grid classes using $derived
  const gridClasses = $derived(() => {
    const base = 'golden-ratio-grid';
    
    const variants = {
      sidebar: 'grid-template-areas-sidebar',
      dashboard: 'grid-template-areas-dashboard', 
      split: 'grid-template-areas-split',
      content: 'grid-template-areas-content',
      'legal-document': 'grid-template-areas-legal-document'
    };

    const directions = {
      horizontal: 'golden-horizontal',
      vertical: 'golden-vertical',
      both: 'golden-both'
    };

    const sizes = {
      sm: 'max-w-3xl',
      md: 'max-w-5xl', 
      lg: 'max-w-7xl',
      xl: 'max-w-full',
      full: 'w-full h-full'
    };

    const gaps = {
      none: 'gap-0',
      xs: 'gap-2',
      sm: 'gap-4',
      md: 'gap-6',
      lg: 'gap-8',
      xl: 'gap-12'
    };

    return cn(
      base,
      variants[variant],
      directions[direction],
      sizes[size],
      gaps[gap],
      {
        'golden-responsive': responsive,
        'nier-golden-grid': legal,
        'yorha-evidence-grid': evidenceLayout,
        'yorha-case-grid': caseLayout,
        'ai-analysis-grid': aiPanels,
        'min-h-screen': size === 'full'
      },
      className
    );
  });

  // Calculate golden ratio proportions
  const goldenProportions = $derived(() => {
    if (direction === 'horizontal') {
      return {
        primary: `${PHI}fr`,
        secondary: '1fr'
      };
    } else if (direction === 'vertical') {
      return {
        primary: `${PHI}fr`,
        secondary: '1fr'
      };
    } else {
      return {
        primary: `${PHI}fr`,
        secondary: '1fr',
        tertiary: `${INVERSE_PHI}fr`
      };
    }
  });
</script>

<div 
  class={gridClasses()}
  style="
    --golden-ratio: {PHI};
    --inverse-golden-ratio: {INVERSE_PHI};
    --primary-proportion: {goldenProportions.primary};
    --secondary-proportion: {goldenProportions.secondary};
    {goldenProportions.tertiary ? `--tertiary-proportion: ${goldenProportions.tertiary};` : ''}
  "
  data-variant={variant}
  data-direction={direction}
  data-legal={legal}
  data-evidence-layout={evidenceLayout}
  data-case-layout={caseLayout}
  data-ai-panels={aiPanels}
>
  {#if header}
    <header class="golden-header">
      {@render header()}
    </header>
  {/if}

  {#if sidebar}
    <aside class="golden-sidebar">
      {@render sidebar()}
    </aside>
  {/if}

  <main class="golden-main">
    {@render children?.()}
  </main>

  {#if secondary}
    <section class="golden-secondary">
      {@render secondary()}
    </section>
  {/if}

  {#if footer}
    <footer class="golden-footer">
      {@render footer()}
    </footer>
  {/if}
</div>

<style>
  /* @unocss-include */
  
  /* Base Golden Ratio Grid */
  :global(.golden-ratio-grid) {
    display: grid;
    width: 100%;
    height: 100%;
    position: relative;
  }

  /* Horizontal Golden Ratio Layout */
  :global(.golden-horizontal) {
    grid-template-columns: var(--primary-proportion) var(--secondary-proportion);
    grid-template-rows: auto 1fr auto;
  }

  /* Vertical Golden Ratio Layout */
  :global(.golden-vertical) {
    grid-template-columns: 1fr;
    grid-template-rows: auto var(--primary-proportion) var(--secondary-proportion) auto;
  }

  /* Both Directions Golden Ratio */
  :global(.golden-both) {
    grid-template-columns: var(--secondary-proportion) var(--primary-proportion) var(--tertiary-proportion);
    grid-template-rows: auto var(--primary-proportion) var(--secondary-proportion) auto;
  }

  /* Grid Template Areas - Sidebar Layout */
  :global(.grid-template-areas-sidebar) {
    grid-template-areas:
      "header header"
      "sidebar main"
      "footer footer";
  }

  /* Grid Template Areas - Dashboard Layout */
  :global(.grid-template-areas-dashboard) {
    grid-template-areas:
      "header header header"
      "sidebar main secondary"
      "footer footer footer";
  }

  /* Grid Template Areas - Split Layout */
  :global(.grid-template-areas-split) {
    grid-template-areas:
      "header header"
      "main secondary"
      "footer footer";
  }

  /* Grid Template Areas - Content Layout */
  :global(.grid-template-areas-content) {
    grid-template-areas:
      "header"
      "main"
      "footer";
  }

  /* Grid Template Areas - Legal Document Layout */
  :global(.grid-template-areas-legal-document) {
    grid-template-areas:
      "header header header"
      "sidebar main secondary"
      "sidebar footer footer";
  }

  /* Grid Area Assignments */
  :global(.golden-header) {
    grid-area: header;
    display: flex;
    align-items: center;
    min-height: calc(var(--inverse-golden-ratio) * 5rem);
  }

  :global(.golden-sidebar) {
    grid-area: sidebar;
    overflow-y: auto;
    min-width: 0;
  }

  :global(.golden-main) {
    grid-area: main;
    overflow-y: auto;
    min-width: 0;
    min-height: 0;
  }

  :global(.golden-secondary) {
    grid-area: secondary;
    overflow-y: auto;
    min-width: 0;
  }

  :global(.golden-footer) {
    grid-area: footer;
    display: flex;
    align-items: center;
    min-height: calc(var(--inverse-golden-ratio) * 3rem);
  }

  /* Legal AI Specific Styling */
  :global(.nier-golden-grid) {
    background: linear-gradient(
      135deg,
      var(--color-nier-bg-primary) 0%,
      var(--color-nier-bg-secondary) 100%
    );
    border: 1px solid var(--color-nier-border-primary);
  }

  :global(.nier-golden-grid .golden-header) {
    background: linear-gradient(
      90deg,
      var(--color-nier-accent-warm),
      var(--color-nier-accent-cool)
    );
    border-bottom: 2px solid var(--color-nier-border-primary);
  }

  :global(.nier-golden-grid .golden-sidebar) {
    background: rgba(var(--color-nier-bg-primary-rgb), 0.8);
    border-right: 1px solid var(--color-nier-border-secondary);
  }

  :global(.nier-golden-grid .golden-footer) {
    background: var(--color-nier-bg-tertiary);
    border-top: 1px solid var(--color-nier-border-secondary);
  }

  /* Evidence Analysis Layout */
  :global(.yorha-evidence-grid) {
    background-image: 
      linear-gradient(45deg, transparent 25%, rgba(0,0,0,0.02) 25%),
      linear-gradient(-45deg, transparent 25%, rgba(0,0,0,0.02) 25%),
      linear-gradient(45deg, rgba(0,0,0,0.02) 75%, transparent 75%),
      linear-gradient(-45deg, rgba(0,0,0,0.02) 75%, transparent 75%);
    background-size: 20px 20px;
    background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
  }

  :global(.yorha-evidence-grid .golden-main) {
    position: relative;
  }

  :global(.yorha-evidence-grid .golden-main::before) {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(
      90deg,
      var(--color-nier-accent-cool),
      var(--color-nier-accent-warm),
      var(--color-nier-accent-cool)
    );
    z-index: 1;
  }

  /* Case Management Layout */
  :global(.yorha-case-grid .golden-sidebar) {
    border-left: 4px solid var(--color-nier-accent-warm);
  }

  :global(.yorha-case-grid .golden-secondary) {
    border-right: 4px solid var(--color-nier-accent-cool);
  }

  /* AI Analysis Panels */
  :global(.ai-analysis-grid .golden-main) {
    background: linear-gradient(
      135deg,
      rgba(59, 130, 246, 0.03) 0%,
      rgba(16, 185, 129, 0.03) 100%
    );
    border: 1px solid rgba(59, 130, 246, 0.1);
  }

  :global(.ai-analysis-grid .golden-secondary) {
    background: linear-gradient(
      135deg,
      rgba(16, 185, 129, 0.03) 0%,
      rgba(245, 158, 11, 0.03) 100%
    );
    border: 1px solid rgba(16, 185, 129, 0.1);
  }

  /* Responsive Golden Ratio Grid */
  @media (max-width: 768px) {
    :global(.golden-responsive.grid-template-areas-sidebar) {
      grid-template-areas:
        "header"
        "main"
        "sidebar"
        "footer";
      grid-template-columns: 1fr;
      grid-template-rows: auto var(--primary-proportion) var(--secondary-proportion) auto;
    }

    :global(.golden-responsive.grid-template-areas-dashboard) {
      grid-template-areas:
        "header"
        "main"
        "secondary"
        "sidebar"
        "footer";
      grid-template-columns: 1fr;
      grid-template-rows: auto var(--primary-proportion) var(--secondary-proportion) var(--tertiary-proportion) auto;
    }

    :global(.golden-responsive.grid-template-areas-split) {
      grid-template-areas:
        "header"
        "main"
        "secondary"
        "footer";
      grid-template-columns: 1fr;
      grid-template-rows: auto var(--primary-proportion) var(--secondary-proportion) auto;
    }
  }

  @media (max-width: 480px) {
    :global(.golden-responsive) {
      gap: 0.5rem;
    }

    :global(.golden-responsive .golden-header),
    :global(.golden-responsive .golden-footer) {
      min-height: 3rem;
    }
  }

  /* Focus and Accessibility */
  :global(.golden-ratio-grid [tabindex]:focus-visible) {
    outline: 2px solid var(--color-nier-border-primary);
    outline-offset: 2px;
  }

  /* Print Styles */
  @media print {
    :global(.golden-ratio-grid) {
      background: white !important;
      color: black !important;
      grid-template-areas:
        "header"
        "main"
        "secondary"
        "footer" !important;
      grid-template-columns: 1fr !important;
    }

    :global(.golden-sidebar) {
      display: none !important;
    }
  }

  /* High Contrast Mode */
  @media (prefers-contrast: high) {
    :global(.nier-golden-grid) {
      border-width: 2px;
    }

    :global(.nier-golden-grid .golden-header),
    :global(.nier-golden-grid .golden-footer) {
      border-width: 2px;
    }
  }

  /* Reduced Motion */
  @media (prefers-reduced-motion: reduce) {
    :global(.golden-ratio-grid *) {
      transition: none !important;
      animation: none !important;
    }
  }
</style>