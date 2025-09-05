<script lang="ts">
  import { cn } from '$lib/utils/cn';

  interface CardTitleProps {
    class?: string;
    children?: import('svelte').Snippet;
  }

  let {
    class: className = '',
    children
  } = $props();
</script>

<h3 class={cn('shadcn-card-title', className)}>
  {@render children?.()}
</h3>

<style>
  :global(.shadcn-card-title) {
    font-size: 1.25rem;
    font-weight: 600;
    line-height: 1.6;
    color: var(--color-foreground, hsl(var(--foreground)));
  }
</style>