<script lang="ts">
  import { melt } from '@melt-ui/svelte';
  
  export let trigger: unknown;
  export let selectedLabel: unknown;
</script>

<button use:melt={trigger} class="mx-auto px-4 max-w-7xl">
  <slot {selectedLabel} />
</button>

<style>
  .select-trigger {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0.75rem;
    border: 1px solid #d1d5db;
    border-radius: 0.375rem;
    background: white;
    min-width: 8rem;
  }
</style>
