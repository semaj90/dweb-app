<script lang="ts">
  import { cn } from '$lib/utils/cn';

  interface CardFooterProps {
    class?: string;
    children?: import('svelte').Snippet;
  }

  let {
    class: className = '',
    children
  } = $props();
</script>

<div class={cn('shadcn-card-footer', className)}>
  {@render children?.()}
</div>

<style>
  :global(.shadcn-card-footer) {
    padding: 0 1.5rem 1.5rem;
    display: flex
    align-items: center
    gap: 0.5rem;
  }
</style>