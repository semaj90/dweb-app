<script lang="ts">
  import { cn } from "$lib/utils";
  interface Props {
    children?: import('svelte').Snippet;
  }

  let { children }: Props = $props();

  let className: string = "";
</script>

<div class={cn("p-6 pt-0", className)} >
  {@render children?.()}
</div>
