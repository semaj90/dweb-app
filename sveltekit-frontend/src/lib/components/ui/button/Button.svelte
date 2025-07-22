<script lang="ts">
  // Svelte 5 runes, Bits UI, UnoCSS, nier.css, shadcn-svelte, melt-ui, and context7 best practices
  import { createButton } from 'bits-ui';
  import { onMount } from 'svelte';
  import type { HTMLButtonAttributes } from 'svelte/elements';

  type ButtonProps = HTMLButtonAttributes & {
    variant?: 'default' | 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger' | 'success' | 'warning' | 'info' | 'nier';
    size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';
    loading?: boolean;
    icon?: string;
    iconPosition?: 'left' | 'right';
    fullWidth?: boolean;
  };

  let { variant = 'primary', size = 'md', loading = false, icon, iconPosition = 'left', fullWidth = false, class: className = '', children, ...restProps } = $props<ButtonProps>();
  const { builder } = createButton();

  // Svelte 5 runes: use $derived(() => ...) for computed values
  const classes = $derived(() => [
    'nier-btn',
    'btn',
    `btn-${variant}`,
    `btn-${size}`,
    fullWidth ? 'w-full' : '',
    loading ? 'btn-loading' : '',
    className
  ].filter(Boolean).join(' '));
</script>

<button use:builder {...restProps} class={classes} disabled={loading || restProps.disabled} data-button-root>
  {#if icon && iconPosition === 'left'}
    <i class={icon} aria-hidden="true"></i>
  {/if}
  {#if loading}
    <span class="loader mr-2"></span>
  {/if}
  {@render children()}
  {#if icon && iconPosition === 'right'}
    <i class={icon} aria-hidden="true"></i>
  {/if}
</button>

<style>
  /* @unocss-include */
  :global(.nier-btn) {
    font-family: 'Oswald', 'Montserrat', 'Inter', 'Segoe UI', 'Arial', 'Helvetica Neue', Arial, 'Liberation Sans', 'Noto Sans', 'sans-serif', 'Gothic A1', 'Gothic', 'sans-serif';
    font-weight: 600;
    letter-spacing: 0.01em;
    text-transform: uppercase;
    background: linear-gradient(90deg, #23272e 0%, #393e46 100%);
    color: #f3f3f3;
    border: none;
    border-radius: 0.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    transition: background 0.2s, color 0.2s, box-shadow 0.2s;
    cursor: pointer;
    min-width: 2.5rem;
    min-height: 2.5rem;
    outline: none;
  }
  :global(.nier-btn:hover) {
    background: linear-gradient(90deg, #393e46 0%, #23272e 100%);
    color: #fff;
    box-shadow: 0 4px 16px rgba(0,0,0,0.12);
  }
  :global(.nier-btn:active) {
    background: #181a1b;
    color: #e0e0e0;
  }
  :global(.nier-btn[disabled]),
  :global(.nier-btn.btn-loading) {
    opacity: 0.6;
    cursor: not-allowed;
    background: #23272e;
    color: #bcbcbc;
  }
  /* Variant styles (shadcn-svelte, melt-ui, bits-ui inspired) */
  :global(.btn-primary) { background: linear-gradient(90deg, #23272e 0%, #393e46 100%); color: #fff; }
  :global(.btn-secondary) { background: #f3f3f3; color: #23272e; border: 1px solid #393e46; }
  :global(.btn-outline) { background: transparent; color: #23272e; border: 1.5px solid #393e46; }
  :global(.btn-ghost) { background: transparent; color: #393e46; border: none; }
  :global(.btn-danger) { background: #e53935; color: #fff; }
  :global(.btn-success) { background: #43a047; color: #fff; }
  :global(.btn-warning) { background: #fbc02d; color: #23272e; }
  :global(.btn-info) { background: #1976d2; color: #fff; }
  :global(.btn-nier) { background: linear-gradient(90deg, #181a1b 0%, #393e46 100%); color: #e0e0e0; }
  /* Size styles */
  :global(.btn-xs) { font-size: 0.75rem; padding: 0.25rem 0.75rem; }
  :global(.btn-sm) { font-size: 0.875rem; padding: 0.375rem 1rem; }
  :global(.btn-md) { font-size: 1rem; padding: 0.5rem 1.25rem; }
  :global(.btn-lg) { font-size: 1.125rem; padding: 0.75rem 1.5rem; }
  :global(.btn-xl) { font-size: 1.25rem; padding: 1rem 2rem; }
  .loader {
    width: 1rem;
    height: 1rem;
    border: 2px solid currentColor;
    border-right-color: transparent;
    border-radius: 50%;
    animation: spin 0.75s linear infinite;
    display: inline-block;
    vertical-align: middle;
  }
  @keyframes spin {
    to { transform: rotate(360deg); }
  }
</style>
