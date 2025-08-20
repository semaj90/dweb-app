<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';

  import { cn } from '$lib/utils';
  interface Props extends CommonProps {
    children?: import('svelte').Snippet;
  }

  let { children }: Props = $props();

  let className: string = '';
</script>

<h3 class={cn('text-2xl font-semibold leading-none tracking-tight', className)}>
  {@render children?.()}
</h3>
