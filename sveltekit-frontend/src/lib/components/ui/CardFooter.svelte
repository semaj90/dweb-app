<script lang="ts">
  import { $props, $derived } from 'svelte';

  interface Props {
    class?: string;
  }
  
  let {
    class = '',
    children
  }: Props & { children?: unknown } = $props();

  let classes = $derived(class);
</script>

<div class="space-y-4">
  {#if children}
    {@render children()}
  {/if}
</div>
