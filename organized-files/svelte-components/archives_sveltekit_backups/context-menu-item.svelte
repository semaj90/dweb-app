<script lang="ts">
  import { createEventDispatcher, getContext } from 'svelte';
  
  interface ContextMenuContext {
    close: () => void;
  }
  
  export let disabled = false;
  
  const dispatch = createEventDispatcher();
  const { close } = getContext<ContextMenuContext>('context-menu') || { close: () => {} };
  
  function handleClick() {
    if (!disabled) {
      dispatch('click');
      close();
    }
  }
</script>

<button
  class="mx-auto px-4 max-w-7xl"
  class:disabled
  role="menuitem"
  tabindex={disabled ? -1 : 0}
  onclick={() => handleClick()}
  {disabled}
>
  <slot />
</button>

<style>
  .context-menu-item {
    display: flex;
    align-items: center;
    width: 100%;
    padding: 0.375rem 0.5rem;
    font-size: 0.875rem;
    border: none;
    border-radius: 0.25rem;
    background: transparent;
    cursor: pointer;
    transition: background-color 0.15s;
    text-align: left;
  }
  
  .context-menu-item:hover:not(.disabled) {
    background-color: #f3f4f6;
  }
  
  .context-menu-item:focus {
    outline: 2px solid #3b82f6;
    outline-offset: -2px;
  }
  
  .context-menu-item.disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>
