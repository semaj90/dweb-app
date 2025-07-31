<script lang="ts">
  import type { HTMLAttributes } from 'svelte/elements';

  interface Props extends HTMLAttributes<HTMLHeadingElement> {
    level?: 1 | 2 | 3 | 4 | 5 | 6;
  }

  let {
    level = 3,
    class: className = '',
    children,
    ...restProps
  }: Props & { children?: any } = $props();

  const classes = $derived([
    'nier-card-title',
    className
  ].filter(Boolean).join(' '));
</script>

{#if level === 1}
  <h1 class={classes} {...restProps}>
    {@render children?.()}
  </h1>
{:else if level === 2}
  <h2 class={classes} {...restProps}>
    {@render children?.()}
  </h2>
{:else if level === 3}
  <h3 class={classes} {...restProps}>
    {@render children?.()}
  </h3>
{:else if level === 4}
  <h4 class={classes} {...restProps}>
    {@render children?.()}
  </h4>
{:else if level === 5}
  <h5 class={classes} {...restProps}>
    {@render children?.()}
  </h5>
{:else}
  <h6 class={classes} {...restProps}>
    {@render children?.()}
  </h6>
{/if}

<style>
  :global(.nier-card-title) {
    font-family: 'Oswald', 'Montserrat', sans-serif;
    font-weight: 600;
    color: #f59e0b;
    margin: 0 0 0.5rem 0;
    line-height: 1.2;
  }
</style>