<script lang="ts">

  import { melt } from '@melt-ui/svelte';
  import { $props } from 'svelte';
  
  interface Props {
    trigger: any;
  }
  let { children,
    trigger
  }: Props = $props();
</script>

<div use:melt={trigger}>
  {@render children?.()}
</div>
