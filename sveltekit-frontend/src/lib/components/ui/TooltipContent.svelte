<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';

  import { cn } from "$lib/utils";
  interface Props extends CommonProps {
    children?: import('svelte').Snippet;
  }

  let { children }: Props = $props();

  let className: string = "";
  ;
</script>

<div
  class={cn(
    "z-50 overflow-hidden rounded-md border bg-popover px-3 py-1.5 text-sm text-popover-foreground shadow-md",
    class
  )}
  
>
  {@render children?.()}
</div>
