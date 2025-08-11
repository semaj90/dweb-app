<script lang="ts">
  interface Props {
    onvalueChange?: (event?: any) => void;
  }
  let {
    value = '',
    onValueChange = > void) | undefined = undefined
  }: Props = $props();



    import { writable } from 'svelte/store';

  
    
  const activeTab = writable(value);
  
  // Set context for child components
  setContext('tabs', {
    activeTab,
    setActiveTab: (newValue: string) => {
      activeTab.set(newValue);
      onvalueChange?.();
      if (onValueChange) {
        onValueChange(newValue);
      }
    }
  });

  // Update store when prop changes
  $: activeTab.set(value);
</script>

<div class="w-full">
  <slot />
</div>