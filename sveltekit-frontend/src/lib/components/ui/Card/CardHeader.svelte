<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';

  import { cn } from "$lib/utils";
  interface Props extends CommonProps {
    children?: import('svelte').Snippet;
  }

  let { children }: Props = $props();

  let className: string = "";
</script>

<div class={cn("flex flex-col space-y-1.5 p-6", className)} >
  {@render children?.()}
</div>
