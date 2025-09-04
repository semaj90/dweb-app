<script lang="ts">
  import { fade, fly } from 'svelte/transition';
  
  export let overlay: unknown;
  export let content: unknown;
  export let openState: unknown;
  export let size: 'sm' | 'md' | 'lg' | 'xl' | 'full' = 'md';
  
  const sizeClasses = {
    sm: 'max-w-sm',
    md: 'max-w-md', 
    lg: 'max-w-lg',
    xl: 'max-w-xl',
    full: 'max-w-[95vw] max-h-[95vh]'
  };
</script>

{#if $openState}
  <!-- Overlay -->
  <div
    use:overlay
    class="mx-auto px-4 max-w-7xl"
    transition:fade={{ duration: 200 "
  ></div>
  
  <!-- Content -->
  <div
    use:content
    class="mx-auto px-4 max-w-7xl"
    transition:fly={{ y: -20, duration: 200  }}
    {...$$restProps}
  >
    <slot />
  </div>
{/if}
