<script lang="ts">
  import { cn } from "$lib/utils";

  let className: string = "";

</script>

<p class={cn("text-sm text-muted-foreground", className)} >
  <slot></slot>
</p>
