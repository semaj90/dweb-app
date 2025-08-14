<script lang="ts">
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
