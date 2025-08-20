<script>
import type { CommonProps } from '$lib/types/common-props';

  let {
    className = '',
    id = undefined,
    ref = null,
    role = undefined,
    ariaLabel = undefined,
    style = undefined,
    children,
    ...restProps
  } = $props();
</script>

<div {id} class={className} bind:this={ref} {role} aria-label={ariaLabel} {style} {...restProps}>
  {#if children}
    {@render children()}
  {/if}
</div>

<style>
  /* minimal fallback styling to avoid layout shifts during migration */
  :global(.bits-ui-fallback) { display: contents; }
</style>
