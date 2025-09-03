<script lang="ts">
  import { $props, $derived } from 'svelte';

  interface Props {
    class?: string;
  }
  
  let {
    class: className = '',
    children
  }: Props & { children?: unknown } = $props();

  let classes = $derived(className);
</script>

<div class="{classes} space-y-4">
  {#if children}
    {@render children()}
  {/if}
</div>
