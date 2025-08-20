<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';

  interface Props extends CommonProps {
    children?: import('svelte').Snippet;
  }

  let { children }: Props = $props();
  // Context menu item component
</script>

<button 
  class="space-y-4"
  
>
  {@render children?.()}
</button>
