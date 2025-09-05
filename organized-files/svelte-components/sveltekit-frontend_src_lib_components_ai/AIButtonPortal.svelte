<script lang="ts">
  interface Props {
    position: | "bottom-right";
    show?: any;
  }
  let {
    position,
    show = true
  } = $props();



  import { Button } from "$lib/components/ui/button";
      | "bottom-left"
    | "top-right"
    | "top-left" = "bottom-right";
  </script>

{#if show}
  <div class="space-y-4" data-position={position}>
    <Button />
  </div>
{/if}

<style>
  /* @unocss-include */
  .ai-button-portal {
    position: fixed
    z-index: 1000;
    pointer-events: auto
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
