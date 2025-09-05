<script lang="ts">
  import { cn } from "$lib/utils";

  let className: string = "";

</script>

<div class={cn("flex items-center p-6 pt-0", className)} >
  <slot></slot>
</div>
