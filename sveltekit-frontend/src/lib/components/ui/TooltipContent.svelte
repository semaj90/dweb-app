<script lang="ts">
  // runtime helper $props() is provided by the runes compiler; do not import it.

  import { cn } from "$lib/utils";
  interface Props {
    children?: import('svelte').Snippet;
  }

  let { children }: Props = $props();

  // `class` is a reserved identifier in runes mode; use `className` instead.
let className = $state<string >("");

</script>

<div
  class={cn(
    "z-50 overflow-hidden rounded-md border bg-popover px-3 py-1.5 text-sm text-popover-foreground shadow-md",
    className
  )}
>
  {@render children?.()}
</div>
