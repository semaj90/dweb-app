<script lang="ts">
  import { $props } from "svelte";
  import { ScrollArea, type WithoutChild } from "bits-ui";
  // Svelte 5 runes, UnoCSS, nier.css, context7 best practices
  // UnoCSS handles styling via class names; see /src/styles/uno.css for class definitions.

  type Props = WithoutChild<ScrollArea.RootProps> & {
    orientation?: "vertical" | "horizontal" | "both";
    viewportClasses?: string;
    type?: "hover" | "scroll" | "auto" | "always";
    scrollHideDelay?: number;
    el?: HTMLDivElement | null;
    children?: any;
  };

  let {
    // Use Svelte 5's $bindable for two-way binding, e.g., bind:el={...}
    el = $bindable(null),
    orientation = "vertical",
    viewportClasses = "",
    type = "hover",
    scrollHideDelay = 600,
    // The `children` prop is part of the type but Svelte's <slot /> is used for content projection.
    // The `...restProps` gathers all other properties passed to this component.
    // These are then spread onto the underlying ScrollArea.Root component,
    // allowing you to pass any of its props directly through this wrapper.
    ...restProps
  }: Props = $props();
</script>

<ScrollArea.Root bind:el={el} type={type} scrollHideDelay={scrollHideDelay} {...restProps}>
  <ScrollArea.Viewport class={viewportClasses}>
    <slot />
  </ScrollArea.Viewport>
  {#if orientation === "vertical" || orientation === "both"}
    <ScrollArea.Scrollbar orientation="vertical">
      <ScrollArea.Thumb />
    </ScrollArea.Scrollbar>
  {/if}
  {#if orientation === "horizontal" || orientation === "both"}
    <ScrollArea.Scrollbar orientation="horizontal">
      <ScrollArea.Thumb />
    </ScrollArea.Scrollbar>
  {/if}
  <ScrollArea.Corner />
</ScrollArea.Root>
  <ScrollArea.Corner />
</ScrollArea.Root>

