<script lang="ts">
  import { Button } from "$lib/components/ui/button";
  export let position:
    | "bottom-right"
    | "bottom-left"
    | "top-right"
    | "top-left" = "bottom-right";
  export let show = true;
</script>

{#if show}
  <div class="container mx-auto px-4" data-position={position}>
    <Button />
  </div>
{/if}

<style>
  /* @unocss-include */
  .ai-button-portal {
    position: fixed;
    z-index: 1000;
    pointer-events: auto;
}
  [data-position="bottom-right"] {
    right: 1.5rem;
    bottom: 1.5rem;
}
  [data-position="bottom-left"] {
    left: 1.5rem;
    bottom: 1.5rem;
}
  [data-position="top-right"] {
    right: 1.5rem;
    top: 1.5rem;
}
  [data-position="top-left"] {
    left: 1.5rem;
    top: 1.5rem;
}
</style>
