<script lang="ts">
  import { $props } from 'svelte';

  interface Props {
    children?: import('svelte').Snippet;
  }

  let { children }: Props = $props();
let class = $state<string >('');
  ;
</script>

<div class="inline-flex h-10 items-center justify-center rounded-md bg-muted p-1 text-muted-foreground {class}">
  {@render children?.()}
</div>