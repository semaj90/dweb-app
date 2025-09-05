// Backup of ScrollArea.svelte before any future changes
// Created on 2025-07-20

<script lang="ts">
  import { ScrollArea, type WithoutChild } from "bits-ui";
  type Props = WithoutChild<ScrollArea.RootProps> & {
    orientation?: "vertical" | "horizontal" | "both";
    viewportClasses?: string;
    type?: "hover" | "scroll" | "auto" | "always";
    scrollHideDelay?: number;
    ref?: HTMLDivElement | null;
    children?: unknown;
  };
  let {
    ref = $bindable(null),
    orientation = "vertical",
    viewportClasses = "",
    type = "hover",
    scrollHideDelay = 600,
    children,
    ...restProps
  }: Props = $props();

  function Scrollbar({ orientation }: { orientation: "vertical" | "horizontal" }) {
    return (
      <ScrollArea.Scrollbar orientation={orientation}>
        <ScrollArea.Thumb />
      </ScrollArea.Scrollbar>
    );
  }
</script>

<ScrollArea.Root bind:ref type={type} scrollHideDelay={scrollHideDelay} {...restProps}>
  <ScrollArea.Viewport class={viewportClasses}>
    {@render children?.()}
  </ScrollArea.Viewport>
  {#if orientation === "vertical" || orientation === "both"}
    {@render Scrollbar({ orientation: "vertical" })}
  {/if}
  {#if orientation === "horizontal" || orientation === "both"}
    {@render Scrollbar({ orientation: "horizontal" })}
  {/if}
  <ScrollArea.Corner />
</ScrollArea.Root>

<!-- UnoCSS handles styling via class names; no <style> block needed. -->
