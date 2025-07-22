<script lang="ts">
  import { Button, type WithElementRef } from 'bits-ui';
  import type { HTMLButtonAttributes } from 'svelte/elements';

  type ButtonProps = WithElementRef<
    HTMLButtonAttributes & {
      variant?: 'default' | 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger' | 'success' | 'warning' | 'info';
      size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';
      loading?: boolean;
      icon?: string;
      iconPosition?: 'left' | 'right';
      fullWidth?: boolean;
      class?: string;
    },
    HTMLButtonElement
  >;

  let {
    variant = 'primary',
    size = 'md',
    loading = false,
    icon,
    iconPosition = 'left',
    fullWidth = false,
    class: className = '',
    ref = $bindable(),
    children,
    ...restProps
  }: ButtonProps = $props();

  const classes = $derived(() => {
    const arr = [
      'nier-btn',
      `nier-btn-${variant}`,
      `nier-btn-${size}`,
      fullWidth && 'w-full',
      loading && 'nier-btn-loading',
      className
    ];
    return arr.filter(Boolean).join(' ');
  });
</script>

<Button.Root
  bind:ref
  class={classes()}
  disabled={loading || Boolean(restProps.disabled)}
  {...restProps}
  {...restProps}
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
</Button.Root>

<style>
/* @unocss-include */
  :global([data-button-root]) {
    height: 2.5rem;
    min-width: 2.5rem;
    background: linear-gradient(90deg, #23272e 0%, #393e46 100%);
    color: #e5e5e5;
    border-radius: 0.5em;
    border: 1.5px solid #bcbcbc;
    font-weight: 600;
    padding: 0.5rem 1.2rem;
    transition: background 0.2s, color 0.2s, border 0.2s;
    box-shadow: 0 2px 8px 0 rgba(0,0,0,0.10);
    cursor: pointer;
    outline: none;
    display: inline-flex;
    align-items: center;
    gap: 0.5em;
  }
  :global([data-button-root]:hover) {
    background: linear-gradient(90deg, #393e46 0%, #23272e 100%);
    color: #a3e7fc;
    border-color: #a3e7fc;
  }
  :global([data-button-root][disabled]) {
    opacity: 0.6;
    cursor: not-allowed;
  }
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
