<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';

  import { melt } from '@melt-ui/svelte';
  
  interface Props extends CommonProps {
    trigger: any;
    selectedLabel: any;
  }
  let {
    trigger,
    selectedLabel
  }: Props = $props();
</script>

<button use:melt={trigger} class="space-y-4">
  <slot {selectedLabel} />
</button>

<style>
  /* @unocss-include */
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