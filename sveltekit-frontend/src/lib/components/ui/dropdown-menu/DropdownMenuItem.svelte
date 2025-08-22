<script lang="ts">
  import { $props } from 'svelte';

  interface Props {
    class?: string;
  }
  let {
    children
  }: { children?: unknown } = $props();
</script>

<button 
  class="space-y-4"
  
>
  {#if children}
    {@render children()}
  {/if}
</button>
