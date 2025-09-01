<script lang="ts">
  import { cn } from '$lib/utils';

  interface Props {
    title?: string;
    subtitle?: string;
    variant?: 'default' | 'card' | 'panel' | 'grid';
    padding?: 'none' | 'sm' | 'md' | 'lg' | 'xl';
    gap?: 'none' | 'sm' | 'md' | 'lg' | 'xl';
    columns?: 1 | 2 | 3 | 4 | 6 | 12;
    class?: string;
  }

  let {
    title,
    subtitle,
    variant = 'default',
    padding = 'lg',
    gap = 'md',
    columns = 1,
    class: className = '',
    ...restProps
  }: Props = $props();

  const sectionClass = $derived(() => {
    const baseClass = "flex flex-col";
    
    const variantClasses = {
      default: "",
      card: "nes-legal-priority-medium yorha-3d-button rounded-lg",
      panel: "yorha-3d-panel neural-sprite-active border border-yellow-400/30",
      grid: "grid-responsive"
    };

    const paddingClasses = {
      none: "p-0",
      sm: "p-2",
      md: "p-4",
      lg: "p-6", 
      xl: "p-8"
    };

    const gapClasses = {
      none: "gap-0",
      sm: "gap-2",
      md: "gap-4",
      lg: "gap-6",
      xl: "gap-8"
    };

    const columnClasses = variant === 'grid' ? {
      1: "grid-cols-1",
      2: "grid-cols-1 md:grid-cols-2",
      3: "grid-cols-1 md:grid-cols-2 lg:grid-cols-3", 
      4: "grid-cols-1 md:grid-cols-2 lg:grid-cols-4",
      6: "grid-cols-2 md:grid-cols-3 lg:grid-cols-6",
      12: "grid-cols-4 md:grid-cols-6 lg:grid-cols-12"
    } : null;

    return cn(
      baseClass,
      variantClasses[variant],
      paddingClasses[padding],
      gapClasses[gap],
      variant === 'grid' && columnClasses?.[columns], className);
  });

  const headerClass = $derived(() => {
    return cn(
      "flex flex-col",
      gap === 'none' ? 'mb-0' :
      gap === 'sm' ? 'mb-2' :
      gap === 'md' ? 'mb-4' :
      gap === 'lg' ? 'mb-6' : 'mb-8'
    );
  });

  const contentClass = $derived(() => {
    if (variant === 'grid') {
      return cn(
        "grid",
        gap === 'none' ? 'gap-0' :
        gap === 'sm' ? 'gap-2' :
        gap === 'md' ? 'gap-4' :
        gap === 'lg' ? 'gap-6' : 'gap-8'
      );
    }

    return cn(
      "flex flex-col",
      gap === 'none' ? 'gap-0' :
      gap === 'sm' ? 'gap-2' :
      gap === 'md' ? 'gap-4' :
      gap === 'lg' ? 'gap-6' : 'gap-8'
    );
  });
</script>

<section class={sectionClass} {...restProps}>
  {#if title || subtitle}
    <header class={headerClass}>
      {#if title}
        <h2 class="nes-legal-title text-2xl md:text-3xl font-bold text-yellow-400">
          {title}
        </h2>
      {/if}
      
      {#if subtitle}
        <p class="nes-legal-subtitle text-gray-300 neural-sprite-active">
          {subtitle}
        </p>
      {/if}
    </header>
  {/if}

  <div class={contentClass}>
    <slot />
  </div>
</section>