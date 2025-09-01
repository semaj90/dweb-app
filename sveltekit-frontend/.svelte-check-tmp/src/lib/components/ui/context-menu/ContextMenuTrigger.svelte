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

<div data-melt-stripped>
  {@render children?.()}
</div>
