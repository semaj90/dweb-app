<script lang="ts">
  import { fade, fly } from 'svelte/transition';
  
  export let overlay: any;
  export let content: any;
  export let openState: any;
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
    class="container mx-auto px-4"
    transition:fade={{ duration: 200 }}
  ></div>
  
  <!-- Content -->
  <div
    use:content
    class="container mx-auto px-4"
    transition:fly={{ y: -20, duration: 200  }}
    {...$$restProps}
  >
    <slot />
  </div>
{/if}
