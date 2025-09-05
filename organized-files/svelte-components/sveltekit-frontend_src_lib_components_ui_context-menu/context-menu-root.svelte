<script lang="ts">
  interface Props {
    onOpenChange?: (open: boolean) => void;
    children: any
  }
  let {
    onOpenChange = undefined,
    children
  } = $props();



  import { setContext } from 'svelte';
  import { writable } from 'svelte/store';
  
    
  const isOpen = writable(false);
  const position = writable({ x: 0, y: 0 });
  
  setContext('context-menu', {
    isOpen,
    position,
    close: () => {
      isOpen.set(false);
      onOpenChange?.(false);
    },
    open: (x: number, y: number) => {
      position.set({ x, y });
      isOpen.set(true);
      onOpenChange?.(true);
    }
  });
</script>

<div class="space-y-4">
  {@render children()}
</div>

<style>
  /* @unocss-include */
  .context-menu-root {
    position: relative
}
</style>
