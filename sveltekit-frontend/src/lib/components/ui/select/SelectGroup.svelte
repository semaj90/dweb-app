<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';

  interface Props extends CommonProps {
    class_: string ;
  }
  let {
    class_ = '',
    children
  }: Props & { children?: any } = $props();



  </script>

<div class="space-y-4">
  {#if children}
    {@render children()}
  {/if}
</div>

<style>
  /* @unocss-include */
  .select-group {
    padding: 4px 0;
}
</style>
