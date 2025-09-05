<script lang="ts">
  interface Props {
    class_: string
  }
  let {
    class_ = ""
  } = $props();



  
  import { getContext } from "svelte";
  import { fly } from "svelte/transition";

  
  const contextMenu = (getContext("contextMenu") as any) || {
    elements: { menu: { subscribe: () => {} } },
  };

  const { elements } = contextMenu;
  const { menu } = elements;
</script>

{#if $menu}
  <div
    use:melt={$menu}
    class="space-y-4"
    transitionfly={{ duration: 150, y: -10 }}
  >
    <slot></slot>
  </div>
{/if}

<style>
  /* @unocss-include */
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
