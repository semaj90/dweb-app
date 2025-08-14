<!-- @migration-task Error while migrating Svelte code: Unexpected token
https://svelte.dev/e/js_parse_error -->
<script lang="ts">
  interface Props {
    title: any;;
  }
  let {
    title
  }: Props = $props();



  </script>

<h2 use:title class="space-y-4">
  <slot></slot>
</h2>
