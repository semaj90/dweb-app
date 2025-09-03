<script lang="ts">
  import { melt } from 'melt';

  interface Props {
    menu: any;
    children?: import('svelte').Snippet;
    class?: string;
  }
  
  let { children, menu, class: className = '' }: Props = $props();
</script>

<div
  <!-- use:melt={menu}
  class="min-w-32 rounded-md border bg-popover p-1 text-popover-foreground shadow-md focus:outline-none {className}"
>
  {@render children?.()}
</div>
