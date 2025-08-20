<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';

  interface Props extends CommonProps {
    menu: any;
  }
  let { children,
    menu
  }: Props = $props();



  </script>

<div 
  use:menu 
  class="space-y-4"
  
>
  {@render children?.()}
</div>
