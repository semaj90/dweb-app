<script lang="ts">
  import type { HTMLButtonAttributes } from 'svelte/elements';
  import type { ButtonVariant, ButtonSize } from '$lib/types';
  import { createEventDispatcher } from 'svelte';

  interface Props extends HTMLButtonAttributes {
    variant?: ButtonVariant;
    size?: ButtonSize;
    children?: any;
  }

  let {
    variant = 'primary',
    size = 'md',
    class: className = '',
    children,
    ...rest
  } = $props();

  const dispatch = createEventDispatcher<{ click: MouseEvent }>();
  function handleClick(event: MouseEvent) {
    dispatch('click', event);
  }

  let classes = $derived([
    'inline-flex items-center justify-center rounded-md font-medium transition-colors',
    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2',
    'disabled:pointer-events-none disabled:opacity-50',
    variant === 'primary' && 'bg-blue-600 text-white hover:bg-blue-700',
    variant === 'secondary' && 'bg-gray-100 text-gray-900 hover:bg-gray-200',
    variant === 'outline' && 'border border-gray-300 bg-white text-gray-700 hover:bg-gray-50',
    variant === 'ghost' && 'text-gray-700 hover:bg-gray-100',
    variant === 'destructive' && 'bg-red-600 text-white hover:bg-red-700',
    variant === 'danger' && 'bg-red-600 text-white hover:bg-red-700',
    variant === 'success' && 'bg-green-600 text-white hover:bg-green-700',
    variant === 'warning' && 'bg-yellow-600 text-white hover:bg-yellow-700',
    variant === 'info' && 'bg-blue-500 text-white hover:bg-blue-600',
    variant === 'default' && 'bg-gray-600 text-white hover:bg-gray-700',
    variant === 'nier' && 'bg-gray-800 text-amber-400 hover:bg-gray-900',
    variant === 'crimson' && 'bg-red-800 text-white hover:bg-red-900',
    variant === 'gold' && 'bg-yellow-500 text-black hover:bg-yellow-600',
    size === 'sm' && 'h-8 px-3 text-sm',
    size === 'md' && 'h-10 px-4 text-sm',
    size === 'lg' && 'h-12 px-6 text-base',
    className
  ].filter(Boolean).join(' '));
 </script>

<button class={classes} on:click={handleClick} {...rest}>
  {#if children}
    {@render children()}
  {:else}
    <!-- Default content when no children provided -->
  {/if}
</button>