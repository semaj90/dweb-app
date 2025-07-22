<script lang="ts">
  // Svelte 5 runes, Bits UI, UnoCSS, nier.css, and context7 best practices
  import { Button } from 'bits-ui';
  import type { HTMLButtonAttributes } from 'svelte/elements';

  type ButtonProps = HTMLButtonAttributes & {
    variant?: 'default' | 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger' | 'success' | 'warning' | 'info';
    size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';
    loading?: boolean;
    icon?: string;
    iconPosition?: 'left' | 'right';
    fullWidth?: boolean;
  };

let { variant = 'primary', size = 'md', loading = false, icon, iconPosition = 'left', fullWidth = false, class: className = '', children, ...restProps } = $props();
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

<Button.Root {...restProps} class={classes} disabled={loading || restProps.disabled}>
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
</Button.Root>

<style>
  /* @unocss-include */
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
