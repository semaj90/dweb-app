<script lang="ts">
  import { cn } from "$lib/utils";

  let className: string = "";

</script>

<div
  class={cn(
    "z-50 overflow-hidden rounded-md border bg-popover px-3 py-1.5 text-sm text-popover-foreground shadow-md",
    className
  )}
  
>
  <slot></slot>
</div>
