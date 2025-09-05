<script lang="ts">
  import { cn } from "$lib/utils";

  let className: string = "";

</script>

<div class={cn("flex flex-col space-y-1.5 p-6", className)} >
  <slot></slot>
</div>
