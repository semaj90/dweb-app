<script lang="ts">
  import { $props } from 'svelte';

  interface Props {
    children?: import('svelte').Snippet;
  }

  let { children }: Props = $props();
  let className: string = '';
  ;
</script>

<div class="inline-flex h-10 items-center justify-center rounded-md bg-muted p-1 text-muted-foreground {className}">
  {@render children?.()}
</div>