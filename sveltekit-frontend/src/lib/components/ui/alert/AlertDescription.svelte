<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';

  interface Props extends CommonProps {
    class?: string;
    children?: import('svelte').Snippet;
  }
  let { 
    class: className = '',
    children,
    ...restProps } = $props();
</script>

<div class="text-sm [&_p]:leading-relaxed {className}" {...restProps}>
  {@render children?.()}
</div>
