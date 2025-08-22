<script lang="ts">
  import { $props } from 'svelte';

  interface Props {
    asChild?: unknown;
    builder: any ;
  }
  let {
    asChild = false,
    builder = null,
    children
  }: Props & { children?: unknown } = $props();




  let className = "";
</script>

{#if asChild}
  {#if children}
    {@render children({ builder })}
  {/if}
{:else}
  <button type="button"  use:builder>
    {#if children}
      {@render children()}
    {/if}
  </button>
{/if}
