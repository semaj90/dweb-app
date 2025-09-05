<script lang="ts">
  import { getContext } from "svelte";
  import { writable } from "svelte/store";
  import type { SelectContext } from "./types";

  interface Props {
    class?: string;
  }
  let {
    class: class_ = ""
  } = $props();

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

<span class="space-y-4">
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
    display: flex
    align-items: center
}
</style>
