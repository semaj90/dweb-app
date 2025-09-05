<!-- Replace the Card component file -->
<script lang="ts">
  import type { HTMLAttributes } from 'svelte/elements';

  interface Props extends HTMLAttributes<HTMLDivElement> {
    variant?: 'default' | 'interactive' | 'outline';
    padding?: 'none' | 'sm' | 'md' | 'lg';
  }

  let {
    variant = 'default',
    padding = 'md',
    class: className = '',
    children,
    ...restProps
  }: Props & { children?: any } = $props();

  const classes = $derived([
    'nier-card',
    `nier-card-${variant}`,
    padding !== 'none' && `nier-card-padding-${padding}`,
    className
  ].filter(Boolean).join(' '));
</script>

<div class={classes} {...restProps} data-card-root>
  {@render children?.()}
</div>

<style>
  :global([data-card-root]) {
    background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
    border: 1px solid #404040;
    border-radius: 0.75rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    transition: all 0.2s ease;
  }

  :global(.nier-card-interactive:hover) {
    border-color: #f59e0b;
    box-shadow: 0 8px 25px -8px rgba(245, 158, 11, 0.3);
    transform: translateY(-2px);
  }

  :global(.nier-card-outline) {
    background: transparent
    border: 2px solid #404040;
  }

  :global(.nier-card-padding-sm) {
    padding: 0.75rem;
  }

  :global(.nier-card-padding-md) {
    padding: 1rem;
  }

  :global(.nier-card-padding-lg) {
    padding: 1.5rem;
  }
</style>
