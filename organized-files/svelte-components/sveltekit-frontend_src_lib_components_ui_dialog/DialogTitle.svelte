<script lang="ts">
  interface Props {
    title: any
    children?: import('svelte').Snippet;
  }
  let {
    title,
    children
  } = $props();
</script>

<h2 use:title class="text-lg font-semibold leading-none tracking-tight">
  {#if children}
    {@render children()}
  {/if}
</h2>