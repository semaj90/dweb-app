<!-- @migration-task Error while migrating Svelte code: Unexpected token
https://svelte.dev/e/js_parse_error -->
<script lang="ts">
  interface Props {
    description: any;;
  }
  let {
    description
  }: Props = $props();



  </script>

<p use:description class="space-y-4">
  <slot></slot>
</p>
