<!-- Replace the Button component file -->
<script lang="ts">
  import { Button } from 'bits-ui';
  import type { HTMLButtonAttributes } from 'svelte/elements';

  interface Props extends HTMLButtonAttributes {
    variant?: 'default' | 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger';
    size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';
    loading?: boolean;
    icon?: string;
    iconPosition?: 'left' | 'right';
    fullWidth?: boolean;
  }

  let {
    variant = 'primary',
    size = 'md',
    loading = false,
    icon = undefined,
    iconPosition = 'left',
    fullWidth = false,
    class: className = '',
    disabled,
    ...restProps
  }: Props = $props();

  const classes = $derived(() => {
    const base = [
      'nier-btn',
      `nier-btn-${variant}`,
      `nier-btn-${size}`,
      fullWidth && 'w-full',
      loading && 'nier-btn-loading',
      className
    ];
    return base.filter(Boolean).join(' ');
  });
</script>

<Button.Root
  class={classes()}
  disabled={loading || disabled}
  {...restProps}
  data-button-root
>
  {#if icon && iconPosition === 'left'}
    <i class={icon} aria-hidden="true"></i>
  {/if}
  {#if loading}
    <span class="loader mr-2"></span>
  {/if}
  <slot />
  {#if icon && iconPosition === 'right'}
    <i class={icon} aria-hidden="true"></i>
  {/if}
</Button.Root>

<style>
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
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }
</style>
