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
  class="mx-auto px-4 max-w-7xl"
  {disabled}
  onclick={() => handleSelect()}
>
  <slot />
</button>
