<script lang="ts">
  import { $props } from 'svelte';
  
  interface ComponentFallbackProps {
    className?: string;
    id?: string;
    ref?: HTMLElement | null;
    role?: string;
    ariaLabel?: string;
    style?: string;
    variant?: string;
    size?: string;
    disabled?: boolean;
    children?: any;
    [key: string]: any;
  }

  let {
    className = '',
    id = undefined,
    ref = null,
    role = undefined,
    ariaLabel = undefined,
    style = undefined,
    variant = 'default',
    size = 'default',
    disabled = false,
    children,
    ...restProps
  }: ComponentFallbackProps = $props();

  // Generate fallback classes based on common patterns
  $: fallbackClasses = [
    className,
    variant && `variant-${variant}`,
    size && `size-${size}`,
    disabled && 'disabled',
    'bits-ui-fallback'
  ].filter(Boolean).join(' ');
</script>

<div 
  {id} 
  class={fallbackClasses} 
  bind:this={ref} 
  {role} 
  aria-label={ariaLabel} 
  {style} 
  {disabled}
  {...restProps}
>
  {#if children}
    {@render children()}
  {/if}
</div>

<style>
  /* Enhanced fallback styling for better compatibility */
  :global(.bits-ui-fallback) { 
    display: contents; 
  }
  
  :global(.bits-ui-fallback.variant-outline) {
    border: 1px solid rgba(0, 0, 0, 0.2);
    background: transparent;
  }
  
  :global(.bits-ui-fallback.variant-ghost) {
    border: none;
    background: transparent;
  }
  
  :global(.bits-ui-fallback.size-sm) {
    font-size: 0.875rem;
    padding: 0.25rem 0.5rem;
  }
  
  :global(.bits-ui-fallback.disabled) {
    opacity: 0.5;
    pointer-events: none;
  }
  
  /* Button-like fallback */
  :global(.bits-ui-fallback[role="button"]) {
    cursor: pointer;
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    transition: all 0.2s ease;
  }
  
  :global(.bits-ui-fallback[role="button"]:hover:not(.disabled)) {
    opacity: 0.8;
  }
</style>
