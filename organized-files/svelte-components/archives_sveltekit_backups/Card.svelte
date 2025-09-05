<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { quintOut } from 'svelte/easing';
  import { fly, scale } from 'svelte/transition';
  import { motion } from "../../../lib/stores/ui";

  export let variant: 'default' | 'elevated' | 'outlined' | 'filled' = 'default';
  export let padding: 'none' | 'sm' | 'md' | 'lg' = 'md';
  export let interactive = false;
  export let href: string | undefined = undefined;
  export let selected = false;
  export let loading = false;

  const dispatch = createEventDispatcher();

  let cardElement: HTMLElement;
  let isHovered = false;

  const handleClick = (event: MouseEvent) => {
    if (interactive || href) {
      dispatch('click', event);
}
  };

  const handleKeydown = (event: KeyboardEvent) => {
    if ((interactive || href) && (event.key === 'Enter' || event.key === ' ')) {
      event.preventDefault();
      dispatch('click', event);
}
  };

  // Dynamic classes
  $: baseClasses = `
    relative overflow-hidden rounded-xl border transition-all duration-200 ease-out
    ${interactive ? 'cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500' : ''}
    ${selected ? 'ring-2 ring-blue-500 border-blue-300' : ''}
  `;

  $: variantClasses = {
    default: 'bg-white border-gray-200 dark:bg-gray-900 dark:border-gray-800',
    elevated: 'bg-white border-gray-200 shadow-sm hover:shadow-md dark:bg-gray-900 dark:border-gray-800',
    outlined: 'bg-transparent border-gray-300 dark:border-gray-600',
    filled: 'bg-gray-50 border-gray-200 dark:bg-gray-800 dark:border-gray-700'
  }[variant];

  $: interactiveClasses = interactive ? 'hover:shadow-lg hover:scale-102 active:scale-98' : '';

  $: paddingClasses = {
    none: '',
    sm: 'p-3',
    md: 'p-4',
    lg: 'p-6'
  }[padding];

  $: classes = `${baseClasses} ${variantClasses} ${interactiveClasses} ${paddingClasses}`;
</script>

<svelte:element
  this={href ? 'a' : 'div'}
  bind:this={cardElement}
  {href}
  class={classes}
  role={interactive ? 'button' : undefined}
  tabindex={interactive ? 0 : undefined}
  onclick={handleClick}
  onkeydown={handleKeydown}
  onmouseenter={() => isHovered = true}
  onmouseleave={() => isHovered = false}
  in:scale={{
    duration: $motion.reduceMotion ? 0 : 300,
    easing: quintOut,
    start: 0.9
  "
  {...$$restProps}
>
  {#if loading}
    <div
      class="mx-auto px-4 max-w-7xl"
      in:fly={{ y: 20, duration: $motion.reduceMotion ? 0 : 200 "
    >
      <div class="mx-auto px-4 max-w-7xl"></div>
    </div>
  {/if}

  <!-- Glow effect for interactive cards -->
  {#if interactive && isHovered && !$motion.reduceMotion}
    <div
      class="mx-auto px-4 max-w-7xl"
      aria-hidden="true"
    ></div>
  {/if}

  <!-- Header slot -->
  {#if $$slots.header}
    <div class="mx-auto px-4 max-w-7xl">
      <slot name="header" />
    </div>
  {/if}

  <!-- Main content -->
  <div class="mx-auto px-4 max-w-7xl">
    <slot />
  </div>

  <!-- Footer slot -->
  {#if $$slots.footer}
    <div class="mx-auto px-4 max-w-7xl">
      <slot name="footer" />
    </div>
  {/if}

  <!-- Actions slot -->
  {#if $$slots.actions}
    <div class="mx-auto px-4 max-w-7xl">
      <slot name="actions" />
    </div>
  {/if}
</svelte:element>

<style>
  /* Custom hover scale using CSS transforms for better performance */
  .hover\:scale-102:hover {
    transform: scale(1.02);
}
  .active\:scale-98:active {
    transform: scale(0.98);
}
</style>