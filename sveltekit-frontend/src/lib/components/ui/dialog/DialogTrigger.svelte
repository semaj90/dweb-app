<script lang="ts">

  import { melt } from 'melt';
  import { $props } from 'svelte';
  
  interface Props {
    trigger: any;
  }
  let { children,
    trigger
  }: Props = $props();
</script>

<button use:melt={trigger}>
  {@render children?.()}
</button>
