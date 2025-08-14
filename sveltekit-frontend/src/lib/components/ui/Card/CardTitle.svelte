<script lang="ts">
  import { cn } from '$lib/utils';
  interface Props {
    children?: import('svelte').Snippet;
  }

  let { children }: Props = $props();

  let className: string = '';
</script>

<h3 class={cn('text-2xl font-semibold leading-none tracking-tight', className)}>
  {@render children?.()}
</h3>
