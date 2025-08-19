<script lang="ts">
  import { createEventDispatcher, setContext } from 'svelte';
  import { writable } from 'svelte/store';

  const dispatch = createEventDispatcher();

  export let value: string = '';
  export let onValueChange: ((value: string) => void) | undefined = undefined;

  const activeTab = writable(value);
  
  // Set context for child components
  setContext('tabs', {
    activeTab,
    setActiveTab: (newValue: string) => {
      activeTab.set(newValue);
      dispatch('valueChange', newValue);
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