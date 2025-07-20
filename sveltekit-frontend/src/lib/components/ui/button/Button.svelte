<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import type { HTMLButtonAttributes } from 'svelte/elements';
  
  type $$Props = HTMLButtonAttributes & {
    variant?: 'default' | 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger';
    size?: 'sm' | 'md' | 'lg';
    disabled?: boolean;
    loading?: boolean;
  };
  
  export let variant: $$Props['variant'] = 'default';
  export let size: $$Props['size'] = 'md';
  export let disabled: $$Props['disabled'] = false;
  export let loading: $$Props['loading'] = false;
  
  const dispatch = createEventDispatcher();
  
  $: classes = [
    'button',
    `button--${variant}`,
    `button--${size}`,
    disabled && 'button--disabled',
    loading && 'button--loading'
  ].filter(Boolean).join(' ');
  
  function handleClick(event: MouseEvent) {
    if (!disabled && !loading) {
      dispatch('click', event);
}
}
</script>

<button
  {...$$restProps}
  class={`${classes} ${$$restProps.class || ''}`}
  {disabled}
  on:click={handleClick}
>
  {#if loading}
    <span class="container mx-auto px-4"></span>
  {/if}
  <slot />
</button>

<style>
  /* @unocss-include */
  .button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 0.375rem;
    font-weight: 500;
    transition: all 0.15s ease;
    cursor: pointer;
    outline: none;
    border: 1px solid transparent;
    gap: 0.5rem;
}
  /* Sizes */
  .button--sm {
    padding: 0.375rem 0.75rem;
    font-size: 0.875rem;
}
  .button--md {
    padding: 0.5rem 1rem;
    font-size: 0.875rem;
}
  .button--lg {
    padding: 0.625rem 1.25rem;
    font-size: 1rem;
}
  /* Variants */
  .button--default {
    background-color: #3b82f6;
    color: white;
    border-color: #3b82f6;
}
  .button--default:hover:not(.button--disabled) {
    background-color: #2563eb;
    border-color: #2563eb;
}
  .button--primary {
    background-color: #6366f1;
    color: white;
    border-color: #6366f1;
}
  .button--primary:hover:not(.button--disabled) {
    background-color: #4f46e5;
    border-color: #4f46e5;
}
  .button--secondary {
    background-color: #f3f4f6;
    color: #374151;
    border-color: #e5e7eb;
}
  .button--secondary:hover:not(.button--disabled) {
    background-color: #e5e7eb;
}
  .button--outline {
    background-color: transparent;
    color: #3b82f6;
    border-color: #3b82f6;
}
  .button--outline:hover:not(.button--disabled) {
    background-color: #3b82f6;
    color: white;
}
  .button--ghost {
    background-color: transparent;
    color: #6b7280;
}
  .button--ghost:hover:not(.button--disabled) {
    background-color: #f3f4f6;
    color: #374151;
}
  .button--danger {
    background-color: #ef4444;
    color: white;
    border-color: #ef4444;
}
  .button--danger:hover:not(.button--disabled) {
    background-color: #dc2626;
    border-color: #dc2626;
}
  /* States */
  .button--disabled {
    opacity: 0.5;
    cursor: not-allowed;
}
  .button--loading {
    cursor: wait;
}
  /* Loader */
  .loader {
    width: 1rem;
    height: 1rem;
    border: 2px solid currentColor;
    border-right-color: transparent;
    border-radius: 50%;
    animation: spin 0.75s linear infinite;
}
  @keyframes spin {
    to {
      transform: rotate(360deg);
}
}
</style>
