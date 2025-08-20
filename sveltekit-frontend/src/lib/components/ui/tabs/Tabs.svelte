<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';

  import { setContext } from 'svelte';
  import { writable } from 'svelte/store';

  interface Props extends CommonProps {
    value: string;
    onValueChange?: (value: string) => void;
    children: any;
  }

  let { value, onValueChange, children }: Props = $props();

  const activeTab = writable(value);

  setContext('tabs', {
    activeTab,
    setActiveTab: (newValue: string) => {
      activeTab.set(newValue);
      if (onValueChange) {
        onValueChange(newValue);
      }
    }
  });

  $effect(() => {
    activeTab.set(value);
  });
</script>

<div class="w-full">
  {@render children()}
</div>
