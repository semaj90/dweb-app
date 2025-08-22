<script lang="ts">
  import { $props } from 'svelte';

  interface Props {
    description: any;
  }
  let {
    description
  }: Props = $props();
</script>

<p use:description class="space-y-4">
  <slot></slot>
</p>