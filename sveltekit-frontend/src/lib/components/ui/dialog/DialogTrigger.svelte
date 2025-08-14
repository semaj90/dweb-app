<script lang="ts">
  import { melt } from '@melt-ui/svelte';
  
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
