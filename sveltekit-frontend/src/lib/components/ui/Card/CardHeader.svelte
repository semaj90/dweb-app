<script lang="ts">
  import { cn } from "$lib/utils";
  interface Props {
    children?: import('svelte').Snippet;
  }

  let { children }: Props = $props();

  let className: string = "";
</script>

<div class={cn("flex flex-col space-y-1.5 p-6", className)} >
  {@render children?.()}
</div>
