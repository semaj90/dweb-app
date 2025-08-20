<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';

  import { createContextMenu, melt } from "@melt-ui/svelte";
  import { setContext } from "svelte";
  interface Props extends CommonProps {
    children?: import('svelte').Snippet;
  }

  let { children }: Props = $props();

  // Create the context menu and set it in context for child components
  const contextMenu = createContextMenu();
  setContext("contextMenu", contextMenu);

  const { elements } = contextMenu;
  const { trigger } = elements;
</script>

<div use:melt={$trigger}>
  {@render children?.()}
</div>
