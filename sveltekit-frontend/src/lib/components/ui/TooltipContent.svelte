<script lang="ts">
  import { cn } from "$lib/utils";

  let className: string = "";
  export { className as class };
</script>

<div
  class={cn(
    "z-50 overflow-hidden rounded-md border bg-popover px-3 py-1.5 text-sm text-popover-foreground shadow-md",
    className
  )}
  {...$$restProps}
>
  <slot />
</div>
