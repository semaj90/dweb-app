<script lang="ts">
  // Simple typed prop for dialog description
  export let description: any
</script>

<p class="space-y-4">
  {#if description}
    {@html description}
  {:else}
    <slot></slot>
  {/if}
</p>
