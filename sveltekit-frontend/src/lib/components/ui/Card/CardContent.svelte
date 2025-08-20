<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';

  import { cn } from "$lib/utils";
  interface Props extends CommonProps {
    children?: import('svelte').Snippet;
  }

  let { children }: Props = $props();

  let className: string = "";
</script>

<div class={cn("p-6 pt-0", className)} >
  {@render children?.()}
</div>
