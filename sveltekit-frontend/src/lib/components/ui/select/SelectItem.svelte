<script lang="ts">
  interface Props {
    value: any;;
    class_: string ;
  }
  let {
    value,
    class_ = ""
  }: Props = $props();



  import { getContext } from "svelte";
  import { writable } from "svelte/store";
  import type { SelectContext } from "./types";

    export let class_: string = "";

  const context =
    getContext<SelectContext>("select") ||
    ({
      selected: writable(null),
      open: writable(false),
      onSelect: () => {},
      onToggle: () => {},
    } as SelectContext);
  const { selected, open, onSelect, onToggle } = context;

  $: isSelected = $selected === value;

  function handleClick() {
    onSelect(value);
    open.set(false);
}
</script>

<div
  class="space-y-4"
  role="option"
  aria-selected={isSelected ? "true" : "false"}
  on:click={() => handleClick()}
  on:keydown={(e) => e.key === "Enter" && handleClick()}
  tabindex={0}
>
  <slot />
</div>

<style>
  /* @unocss-include */
  .select-item {
    padding: 8px 12px;
    cursor: pointer;
    font-size: 14px;
    color: #374151;
    display: flex;
    align-items: center;
}
  .select-item:hover {
    background-color: #f3f4f6;
}
  .select-item:focus {
    outline: none;
    background-color: #e5e7eb;
}
</style>
