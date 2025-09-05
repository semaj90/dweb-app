<script lang="ts">
  export let title: unknown;
</script>

<h2 use:title className="${1}">
  <slot />
</h2>
