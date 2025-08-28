<script lang="ts">
  import type { HTMLAttributes } from 'svelte/elements';
  import type { Snippet } from 'svelte';
  
  interface Props extends HTMLAttributes<HTMLHeadingElement> {
    className?: string;
    children?: Snippet;
  }
  
  let { className = '', children, ...restProps }: Props = $props();
</script>

<h3
  class="text-2xl font-semibold leading-none tracking-tight {className}"
  {...restProps}
>
  {#if children}
    {@render children()}
  {/if}
</h3>