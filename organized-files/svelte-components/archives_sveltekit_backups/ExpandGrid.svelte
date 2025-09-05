<script lang="ts">
  import { createEventDispatcher } from "svelte";

  export let columns = 1;
  export let expandedColumns = 3;
  export let gap = "1rem";
  export let expandDuration = "0.4s";
  export let easing = "ease";
  export let expandOnHover = true;
  export let expandOnFocus = true;

  let className = "";
  export { className as class };

  const dispatch = createEventDispatcher();

  let isExpanded = false;
  let containerElement: HTMLDivElement;

  function handleMouseEnter() {
    if (expandOnHover) {
      isExpanded = true;
      dispatch("expand", { expanded: true });
    }
  }

  function handleMouseLeave() {
    if (expandOnHover) {
      isExpanded = false;
      dispatch("expand", { expanded: false });
    }
  }

  function handleFocusIn() {
    if (expandOnFocus) {
      isExpanded = true;
      dispatch("expand", { expanded: true });
    }
  }

  function handleFocusOut() {
    if (expandOnFocus) {
      isExpanded = false;
      dispatch("expand", { expanded: false });
    }
  }

  $: currentColumns = isExpanded ? expandedColumns : columns;
</script>

<div
  bind:this={containerElement}
  class="mx-auto px-4 max-w-7xl"
  class:expanded={isExpanded}
  style="
    --columns: {currentColumns};
    --gap: {gap};
    --expand-duration: {expandDuration};
    --easing: {easing};
  "
  onmouseenter={handleMouseEnter}
  onmouseleave={handleMouseLeave}
  onfocusin={handleFocusIn}
  onfocusout={handleFocusOut}
  role="grid"
  tabindex={0}
>
  <slot />
</div>

<style>
  .expand-grid {
    display: grid;
    grid-template-columns: repeat(var(--columns), 1fr);
    gap: var(--gap);
    transition: grid-template-columns var(--expand-duration) var(--easing);
    outline: none;
    border-radius: 0.5rem;
    padding: 0.5rem;
    background: var(--pico-card-background-color, #ffffff);
    border: 1px solid transparent;
  }

  .expand-grid:focus-visible {
    border-color: var(--pico-primary, #3b82f6);
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
  }

  .expand-grid.expanded {
    background: var(--pico-card-sectioning-background-color, #f8fafc);
    border-color: var(--pico-border-color, #e2e8f0);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
  }

  /* Grid item styling */
  .expand-grid :global(.grid-item) {
    transition: all var(--expand-duration) var(--easing);
    border-radius: 0.375rem;
    overflow: hidden;
    position: relative;
  }

  .expand-grid.expanded :global(.grid-item) {
    transform: scale(1.02);
  }

  .expand-grid :global(.grid-item:hover) {
    transform: scale(1.05);
    z-index: 10;
  }

  /* Responsive design */
  @media (max-width: 768px) {
    .expand-grid {
      grid-template-columns: 1fr;
    }

    .expand-grid.expanded {
      grid-template-columns: repeat(2, 1fr);
    }
  }

  @media (max-width: 480px) {
    .expand-grid,
    .expand-grid.expanded {
      grid-template-columns: 1fr;
    }
  }
</style>