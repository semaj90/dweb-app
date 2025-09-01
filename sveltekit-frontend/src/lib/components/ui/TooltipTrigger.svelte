<script lang="ts">
  // runtime helper $props() is provided by the runes compiler; do not import it.

  interface Props {
    asChild?: unknown;
    builder: any ;
  }
  let {
    asChild = false,
    builder = null,
    children
  }: Props & { children?: unknown } = $props();



  // `class` is reserved; use `className` instead
let className = $state("");
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
