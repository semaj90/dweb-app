<script lang="ts">
  interface Props {
    menu: any;;
  }
  let {
    menu
  }: Props = $props();



  </script>

<div 
  use:menu 
  class="space-y-4"
  {...$$restProps}
>
  <slot />
</div>
