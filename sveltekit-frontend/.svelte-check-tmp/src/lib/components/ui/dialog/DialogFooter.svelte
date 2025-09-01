<script lang="ts">
  import { $props } from 'svelte';

  interface Props {
    children?: import('svelte').Snippet;
  }

  let { children }: Props = $props();
</script>

<div class="space-y-4">
  {@render children?.()}
</div>
