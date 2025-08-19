<script lang="ts">
  import { melt } from "@melt-ui/svelte";
  import { createEventDispatcher, getContext } from "svelte";

  export let class_: string = "";
  export let disabled: boolean = false;

  const dispatch = createEventDispatcher();
  const contextMenu = (getContext("contextMenu") as any) || {
    elements: { item: { subscribe: () => {}, set: () => {} } },
  };

  const { elements } = contextMenu;
  const { item } = elements;

  function handleSelect() {
    dispatch("select");
  }
</script>

<button
  use:melt={$item}
  class="flex items-center w-full px-3 py-2 text-sm text-left hover:bg-gray-100 dark:hover:bg-gray-700 focus:bg-gray-100 dark:focus:bg-gray-700 focus:outline-none transition-colors {class_}"
  {disabled}
  on:click={() => handleSelect()}
>
  <slot />
</button>
