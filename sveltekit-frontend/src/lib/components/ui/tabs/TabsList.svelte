<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';

  interface Props extends CommonProps {
    children?: import('svelte').Snippet;
  }

  let { children }: Props = $props();
  let className: string = '';
  ;
</script>

<div class="inline-flex h-10 items-center justify-center rounded-md bg-muted p-1 text-muted-foreground {className}">
  {@render children?.()}
</div>