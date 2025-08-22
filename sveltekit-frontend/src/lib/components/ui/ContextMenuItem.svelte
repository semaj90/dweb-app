<script lang="ts">
  import { $props } from 'svelte';

  interface Props {
    onselect?: (event?: unknown) => void;
  }
  let {
    class_ = "",
    disabled = false
  }: Props = $props();



  import { melt } from "melt";
  
    
    const contextMenu = (getContext("contextMenu") as any) || {
    elements: { item: { subscribe: () => {}, set: () => {} } },
  };

  const { elements } = contextMenu;
  const { item } = elements;

  function handleSelect() {
    onselect?.();
  }
</script>

<button
  use:melt={$item}
  class="flex items-center w-full px-3 py-2 text-sm text-left hover:bg-gray-100 dark:hover:bg-gray-700 focus:bg-gray-100 dark:focus:bg-gray-700 focus:outline-none transition-colors {class_}"
  {disabled}
  onclick={() => handleSelect()}
>
  {@render children?.()}
</button>
