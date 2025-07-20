<script lang="ts">
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
  const { selected } = context;
</script>

<span class="container mx-auto px-4">
  {#if $selected}
    <slot value={$selected}>
      {$selected}
    </slot>
  {:else}
    <slot name="placeholder">Select an option...</slot>
  {/if}
</span>

<style>
  /* @unocss-include */
  .select-value {
    flex: 1;
    display: flex;
    align-items: center;}
</style>
