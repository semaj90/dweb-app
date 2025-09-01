<script lang="ts">
  import { $props } from 'svelte';

  interface Props {
    class_: string ;
  }
  let {
    class_ = '',
    children
  }: Props & { children?: unknown } = $props();



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
