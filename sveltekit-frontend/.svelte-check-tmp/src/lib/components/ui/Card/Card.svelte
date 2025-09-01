<script lang="ts">
  import type { Snippet } from 'svelte';
  import { cn } from "$lib/utils";

  interface Props {
    class?: string;
    variant?: 'default' | 'evidence' | 'legal' | 'nes' | 'yorha';
    priority?: 'critical' | 'high' | 'medium' | 'low';
    loading?: boolean;
    interactive?: boolean;
    children?: Snippet;
  }

  let {
  // avoid using the reserved identifier `class` as a variable name
  class: className = '',
    variant = 'default',
    priority,
    loading = false,
    interactive = false,
    children,
    ...restProps
  }: Props = $props();

  let cardClass = $derived(() => {
    const baseClass = "border shadow-sm transition-all duration-200";

    // Variant-based styling with NES-inspired classes
    const variantClasses = {
      default: "card-nes-default",
      evidence: "card-nes-evidence legal-document-evidence",
      legal: "card-nes-legal legal-document-contract",
      nes: "yorha-3d-card bg-gray-900/90 border-yellow-400/50 text-yellow-100",
      yorha: "yorha-3d-panel text-yellow-400"
    };

    // Priority-based styling
    const priorityClasses = priority ? {
      critical: "nes-legal-priority-critical ring-2 ring-red-400",
      high: "nes-legal-priority-high ring-2 ring-yellow-400",
      medium: "nes-legal-priority-medium ring-2 ring-blue-400",
      low: "nes-legal-priority-low"
    }[priority] : '';

    // Loading state
    const loadingClass = loading ? "nes-loading opacity-75" : '';

    // Interactive state
    const interactiveClass = interactive ? "hover:scale-[1.01] cursor-pointer" : '';

    return cn(
      baseClass,
      variantClasses[variant],
      priorityClasses,
      loadingClass,
      interactiveClass,
  className
    );
  });
</script>

<div
  class={cardClass}
  role={interactive ? "button" : undefined}
  tabindex={interactive ? 0 : -1}
  {...restProps}
>
  {#if loading}
    <div class="neural-sprite-loading absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent"></div>
  {/if}
  {@render children?.()}
</div>
