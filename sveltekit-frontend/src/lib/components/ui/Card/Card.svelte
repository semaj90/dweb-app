<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';

  import { cn } from "$lib/utils";
  interface Props extends CommonProps {
    children?: import('svelte').Snippet;
  }

  let { children }: Props = $props();

  let className: string = "";
</script>

<div
  class={cn(
    "rounded-lg border bg-card text-card-foreground shadow-sm",
    className
  )}
>
  {@render children?.()}
</div>
