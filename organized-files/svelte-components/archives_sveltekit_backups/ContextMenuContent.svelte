<script lang="ts">
  import { melt } from "@melt-ui/svelte";
  import { getContext } from "svelte";
  import { fly } from "svelte/transition";

  export let class_: string = "";

  const contextMenu = (getContext("contextMenu") as any) || {
    elements: { menu: { subscribe: () => {} } },
  };

  const { elements } = contextMenu;
  const { menu } = elements;
</script>

{#if $menu}
  <div
    use:melt={$menu}
    class="mx-auto px-4 max-w-7xl"
    transition:fly={{ duration: 150, y: -10 "
  >
    <slot />
  </div>
{/if}

<style>
  .context-menu {
    animation: contextMenuFadeIn 150ms ease-out;
}
  @keyframes contextMenuFadeIn {
    from {
      opacity: 0;
      transform: scale(0.95);
}
    to {
      opacity: 1;
      transform: scale(1);
}}
</style>
