<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';

  interface Props extends CommonProps {
    class?: string;
  }
  
  let {
    class = '',
    children
  }: Props & { children?: any } = $props();

  const classes = $derived(className);
</script>

<div class="space-y-4">
  {#if children}
    {@render children()}
  {/if}
</div>
