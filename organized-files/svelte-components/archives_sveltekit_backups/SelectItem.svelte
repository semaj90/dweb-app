<script lang="ts">
  import { getContext } from "svelte";
  import { writable } from "svelte/store";
  import type { SelectContext } from "./types";

  export let value: unknown;
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
  class="mx-auto px-4 max-w-7xl"
  role="option"
  aria-selected={isSelected ? "true" : "false"}
  onclick={() => handleClick()}
  onkeydown={(e) => e.key === "Enter" && handleClick()}
  tabindex={0}
>
  <slot />
</div>

<style>
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
