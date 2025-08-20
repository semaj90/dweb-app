<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';

  interface Props extends CommonProps {
    class?: string;
    children?: import('svelte').Snippet;
  }

  let { 
    children,
    class: className = '',
    ...restProps
  }: Props = $props();
</script>

<h5 class="mb-1 font-medium leading-none tracking-tight {className}" {...restProps}>
  {@render children?.()}
</h5>
