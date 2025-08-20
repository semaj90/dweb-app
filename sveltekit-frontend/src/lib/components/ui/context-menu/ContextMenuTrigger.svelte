<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';

  import { melt } from '@melt-ui/svelte';
  
  interface Props extends CommonProps {
    trigger: any;
  }
  let { children,
    trigger
  }: Props = $props();
</script>

<div use:melt={trigger}>
  {@render children?.()}
</div>
