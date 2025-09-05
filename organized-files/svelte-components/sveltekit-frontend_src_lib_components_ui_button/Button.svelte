<script lang="ts">
  import type { HTMLButtonAttributes } from 'svelte/elements';
  import type { ButtonVariant, ButtonSize } from '$lib/types';

  // Migrated to Svelte 5 runes
  let {
    variant = 'primary',
    size = 'md',
    loading = false,
    icon = undefined,
    iconPosition = 'left',
    fullWidth = false,
    class: className = '',
    children,
    ref = $bindable(undefined),
    ...rest
  }: {
    variant?: ButtonVariant;
    size?: ButtonSize;
    loading?: boolean;
    icon?: string;
    iconPosition?: 'left' | 'right';
    fullWidth?: boolean;
    class?: string;
    children?: any;
    ref?: HTMLButtonElement;
  } & HTMLButtonAttributes = $props();

  let classes = $derived([
    'nier-btn',
    `btn-${variant}`,
    `btn-${size}`,
    fullWidth && 'w-full',
    loading && 'btn-loading',
    className
  ].filter(Boolean).join(' '));
</script>

<button
  bind:this={ref}
  class={classes}
  disabled={loading || Boolean(rest.disabled)}
  {...rest}
  data-button-root
>
  {#if icon && iconPosition === 'left'}
    <i class={icon} aria-hidden="true"></i>
  {/if}
  {#if loading}
    <span class="loader mr-2"></span>
  {/if}
  {#if children}
    {@render children()}
  {/if}
  {#if icon && iconPosition === 'right'}
    <i class={icon} aria-hidden="true"></i>
  {/if}
</button>

<style>
  :global(.nier-btn) {
    font-family: 'Oswald', 'Montserrat', 'Inter', 'Segoe UI', 'Arial', 'Helvetica Neue', Arial, 'Liberation Sans', 'Noto Sans', 'Gothic A1', 'Gothic', 'sans-serif';
    font-weight: 600;
    letter-spacing: 0.01em;
    text-transform: uppercase
    background: linear-gradient(90deg, #23272e 0%, #393e46 100%);
    color: #f3f3f3;
    border: none
    border-radius: 0.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    transition: background 0.2s, color 0.2s, box-shadow 0.2s;
    cursor: pointer
    min-width: 2.5rem;
    min-height: 2.5rem;
    outline: none
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
  /* Variant styles */
  :global(.btn-default) { background: linear-gradient(90deg, #23272e 0%, #393e46 100%); color: #fff; }
  :global(.btn-primary) { background: linear-gradient(90deg, #23272e 0%, #393e46 100%); color: #fff; }
  :global(.btn-secondary) { background: #f3f3f3; color: #23272e; border: 1px solid #393e46; }
  :global(.btn-outline) { background: transparent color: #23272e; border: 1.5px solid #393e46; }
  :global(.btn-danger) { background: #e53935; color: #fff; }
  :global(.btn-destructive) { background: #e53935; color: #fff; }
  :global(.btn-success) { background: #43a047; color: #fff; }
  :global(.btn-warning) { background: #fbc02d; color: #23272e; }
  :global(.btn-info) { background: #1976d2; color: #fff; }
  :global(.btn-ghost) { background: transparent color: #23272e; border: none }
  :global(.btn-ghost:hover) { background: rgba(35, 39, 46, 0.1); }
  :global(.btn-nier) { background: linear-gradient(90deg, #181a1b 0%, #393e46 100%); color: #e0e0e0; }
  :global(.btn-crimson) { background: linear-gradient(90deg, #8B0000 0%, #DC143C 100%); color: #fff; }
  :global(.btn-gold) { background: linear-gradient(90deg, #B8860B 0%, #FFD700 100%); color: #000; }
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
    border-right-color: transparent
    border-radius: 50%;
    animation: spin 0.75s linear infinite;
    display: inline-block;
    vertical-align: middle
  }
  @keyframes spin {
    to { transform: rotate(360deg); }
  }
</style>
