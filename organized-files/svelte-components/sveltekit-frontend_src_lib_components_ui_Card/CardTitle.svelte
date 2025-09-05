<script lang="ts">
  import { cn } from "$lib/utils";

  let className: string = "";

</script>

<h3
  class={cn("text-2xl font-semibold leading-none tracking-tight", className)}
  
>
  <slot></slot>
</h3>
