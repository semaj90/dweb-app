<script lang="ts">
  interface Props {
    className: string
  }
  let {
    className = ''
  } = $props();



  import { getContext, onDestroy, onMount } from 'svelte';
  import type { Writable } from 'svelte/store';
  
    
  const { isOpen, position, close } = getContext<{
    isOpen: Writable<boolean>
    position: Writable<{ x: number y: number }>;
    close: () => void;
  }>('context-menu');
  
  let menuElement: HTMLDivElement
  
  function handleClickOutside(event: MouseEvent) {
    if (menuElement && !menuElement.contains(event.target as Node)) {
      close();
    }
  }
  function handleEscape(event: KeyboardEvent) {
    if (event.key === 'Escape') {
      close();
    }
  }
  onMount(() => {
    document.addEventListener('click', handleClickOutside);
    document.addEventListener('keydown', handleEscape);
  });
  
  onDestroy(() => {
    document.removeEventListener('click', handleClickOutside);
    document.removeEventListener('keydown', handleEscape);
  });
</script>

{#if $isOpen}
  <div
    bind:this={menuElement}
    class="space-y-4"
    style="left: {$position.x}px; top: {$position.y}px;"
    role="menu"
    tabindex={-1}
  >
    <slot></slot>
  </div>
{/if}

<style>
  /* @unocss-include */
  .context-menu-content {
    position: fixed
    z-index: 1000;
    min-width: 12rem;
    background-color: white
    border: 1px solid #e5e7eb;
    border-radius: 0.375rem;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    padding: 0.25rem;
}
</style>
