<script lang="ts">
  import { $props } from 'svelte';

  interface Props {
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
