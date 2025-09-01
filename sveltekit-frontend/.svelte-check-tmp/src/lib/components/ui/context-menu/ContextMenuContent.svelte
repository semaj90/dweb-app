<script lang="ts">
  import { $props } from 'svelte';

  interface Props {
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
