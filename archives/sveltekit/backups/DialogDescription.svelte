<script lang="ts">
  export let description: unknown;
</script>

<p use:description className="${1}">
  <slot />
</p>
