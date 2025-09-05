<script lang="ts">
  import { cn } from '$lib/utils/cn';

  interface CardContentProps {
    class?: string;
    children?: import('svelte').Snippet;
  }

  let {
    class: className = '',
    children
  } = $props();
</script>

<div class={cn('shadcn-card-content', className)}>
  {@render children?.()}
</div>

<style>
  :global(.shadcn-card-content) {
    padding: 1.5rem;
    padding-top: 0;
  }
</style>