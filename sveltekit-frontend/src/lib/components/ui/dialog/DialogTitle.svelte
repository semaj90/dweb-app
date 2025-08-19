<script lang="ts">
  interface Props {
    title: any;
  }
  let {
    title
  }: Props = $props();
</script>

<h2 use:title class="space-y-4">
  <slot></slot>
</h2>