<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';

  interface Props extends CommonProps {
    description: any;
  }
  let {
    description
  }: Props = $props();
</script>

<p use:description class="space-y-4">
  <slot></slot>
</p>