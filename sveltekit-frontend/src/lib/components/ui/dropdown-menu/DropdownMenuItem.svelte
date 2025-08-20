<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';

  interface Props extends CommonProps {
    class?: string;
  }
  let {
    children
  }: { children?: any } = $props();
</script>

<button 
  class="space-y-4"
  
>
  {#if children}
    {@render children()}
  {/if}
</button>
