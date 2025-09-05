<script lang="ts">
  import { cn } from "$lib/utils";

  let className: string = "";

</script>

<div
  class={cn(
    "rounded-lg border bg-card text-card-foreground shadow-sm",
    className
  )}
  
>
  <slot></slot>
</div>
