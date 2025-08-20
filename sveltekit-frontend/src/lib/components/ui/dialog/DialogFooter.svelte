<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';

  interface Props extends CommonProps {
    children?: import('svelte').Snippet;
  }

  let { children }: Props = $props();
</script>

<div class="space-y-4">
  {@render children?.()}
</div>
