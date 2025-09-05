<script lang="ts">
  interface Props {
    items: any[] ;
    columnWidth?: any;
    gutter?: any;
    itemSelector?: any;
    containerClass?: any;
    fitWidth?: any;
    horizontalOrder?: any;
    percentPosition?: any;
    resize?: any;
    initLayout?: any;
    transitionDuration?: any;
    dragDisabled?: any;
    dropTargetStyle?: any;
    dropFromOthersDisabled?: any;
  }
  let {
    items = [],
    columnWidth = 300,
    gutter = 16,
    itemSelector = '.masonry-item',
    containerClass = 'masonry-container',
    fitWidth = true,
    horizontalOrder = false,
    percentPosition = false,
    resize = true,
    initLayout = true,
    transitionDuration = '0.3s',
    dragDisabled = false,
    dropTargetStyle = {},
    dropFromOthersDisabled = false
  } = $props();



  import { onMount, onDestroy } from 'svelte';
  import { dndzone } from 'svelte-dnd-action';
  import { fly } from 'svelte/transition';
  import Masonry from 'masonry-layout';
  
                              
  let container: HTMLElement
  let masonry: any
  let isInitialized = false;
  
  // Masonry configuration
  let masonryOptions = $derived({})
    itemSelector,
    columnWidth,
    gutter,
    fitWidth,
    horizontalOrder,
    percentPosition,
    resize,
    initLayout,
    transitionDuration
  };
  
  // Initialize Masonry
  onMount(() => {
    if (container && items.length > 0) {
      setTimeout(() => {
        masonry = new Masonry(container, masonryOptions);
        isInitialized = true;
      }, 100);
}
  });
  
  // Update layout when items change
  $effect(() => { if (masonry && isInitialized) {
    setTimeout(() => {
      if (masonry) {
        masonry.reloadItems();
        masonry.layout();
}
    }, 50);
}
  // Cleanup
  onDestroy(() => {
    if (masonry) {
      masonry.destroy();
}
  });
  
  // Handle drag and drop
  const handleDndConsider = (e: CustomEvent) => {
    items = e.detail.items;
  };
  
  const handleDndFinalize = (e: CustomEvent) => {
    items = e.detail.items;
    // Trigger layout update after reordering
    setTimeout(() => {
      if (masonry) {
        masonry.layout();
}
    }, 100);
  };
  
  // Responsive breakpoints
  const getResponsiveColumns = () => {
    if (typeof window === 'undefined') return 3;
    const width = window.innerWidth;
    if (width < 640) return 1;
    if (width < 1024) return 2;
    if (width < 1280) return 3;
    return 4;
  };
  
  // Auto-resize functionality
  let resizeTimeout: NodeJS.Timeout;
  const handleResize = () => {
    if (!resize || !masonry) return;
    
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
      if (masonry) {
        masonry.layout();
}
    }, 150);
  };
  
  onMount(() => {
    if (resize) {
      window.addEventListener('resize', handleResize);
}
    return () => {
      window.removeEventListener('resize', handleResize);
      if (resizeTimeout) clearTimeout(resizeTimeout);
    };
  });
</script>

<div
  bind:this={container}
  class={`${containerClass} masonry-grid`}
  use:dndzone={{
    items,
    dragDisabled,
    dropTargetStyle,
    dropFromOthersDisabled
  }}
  onconsider={handleDndConsider}
  onfinalize={handleDndFinalize}
  style="--column-width: {columnWidth}px; --gutter: {gutter}px;"
>
  {#each items as item, index (item.id)}
    <div 
      class="space-y-4"
      transitionfly={{ y: 20, duration: 300, delay: index * 50 }}
    >
      <slot {item} {index} />
    </div>
  {/each}
</div>

<style>
  /* @unocss-include */
  .masonry-grid {
    position: relative
    margin: 0 auto;
}
  :global(.masonry-item) {
    width: var(--column-width);
    margin-bottom: var(--gutter);
    break-inside: avoid
    position: relative
    transition: transform 0.3s ease, opacity 0.3s ease;
}
  /* Responsive design */
  @media (max-width: 640px) {
    :global(.masonry-item) {
      width: calc(100% - var(--gutter));
}}
  @media (min-width: 641px) and (max-width: 1024px) {
    :global(.masonry-item) {
      width: calc(50% - var(--gutter));
}}
  @media (min-width: 1025px) and (max-width: 1280px) {
    :global(.masonry-item) {
      width: calc(33.333% - var(--gutter));
}}
  @media (min-width: 1281px) {
    :global(.masonry-item) {
      width: calc(25% - var(--gutter));
}}
  /* Drag and drop styling */
  :global(.masonry-item.drag-disabled) {
    cursor: default
}
  :global(.masonry-item:not(.drag-disabled)) {
    cursor: grab
}
  :global(.masonry-item:not(.drag-disabled):active) {
    cursor: grabbing
}
  :global(.masonry-item.drag-shadow) {
    opacity: 0.5;
    transform: scale(0.95);
}
  :global(.masonry-item.drag-ghost) {
    opacity: 0.3;
    transform: rotate(5deg);
}
  /* Loading state */
  .masonry-grid:empty::before {
    content: 'Loading...';
    display: block
    text-align: center
    color: var(--pico-muted-color, #6b7280);
    font-style: italic
    padding: 2rem;
}
  /* Animation for new items */
  :global(.masonry-item.newly-added) {
    animation: slideInUp 0.3s ease-out;
}
  @keyframes slideInUp {
    from {
      transform: translateY(20px);
      opacity: 0;
}
    to {
      transform: translateY(0);
      opacity: 1;
}}
  /* Hover effects */
  :global(.masonry-item:hover) {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}
  /* Focus styles for accessibility */
  :global(.masonry-item:focus-within) {
    outline: 2px solid var(--pico-primary, #3b82f6);
    outline-offset: 2px;
}
</style>
