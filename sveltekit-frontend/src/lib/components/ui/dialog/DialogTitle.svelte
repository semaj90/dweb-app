<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';

  interface Props extends CommonProps {
    title: any;
  }
  let {
    title
  }: Props = $props();
</script>

<h2 use:title class="space-y-4">
  <slot></slot>
</h2>