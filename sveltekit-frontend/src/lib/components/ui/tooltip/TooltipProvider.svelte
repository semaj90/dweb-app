<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';

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


<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';
interface Props extends CommonProps {}
</script>
