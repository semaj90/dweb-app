<script lang="ts">
  import type { HTMLAttributes } from 'svelte/elements';
  import type { Snippet } from 'svelte';
  
  interface Props extends HTMLAttributes<HTMLLabelElement> {
    className?: string;
    for?: string;
    children?: Snippet;
  }
  
  let { className = '', htmlFor, children, ...restProps }: Props = $props();
</script>

<label
  for={htmlFor}
  class="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 {className}"
  {...restProps}
>
  {#if children}
    {@render children()}
  {/if}
</label>