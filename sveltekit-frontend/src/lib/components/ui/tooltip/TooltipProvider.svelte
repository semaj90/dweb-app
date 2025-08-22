<script lang="ts">
  import { $props } from 'svelte';

  import type { Props } from "$lib/types/global";


  let { children }: Props = $props();
</script>

<div class="tooltip-provider" style="display: contents">
  {@render children?.()}
</div>

<style>
  .tooltip-provider {
    position: relative;
    display: inline-block;
  }
</style>



