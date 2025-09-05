<script lang="ts">
  interface Props {
    columns?: number;
    expandedColumns?: number;
    gap?: string;
    expandDuration?: string;
    easing?: string;
    expandOnHover?: boolean;
    expandOnFocus?: boolean;
    onexpand?: (event?: any) => void;
  }
  
  let {
    columns = 1,
    expandedColumns = 3,
    gap = "1rem",
    expandDuration = "0.4s",
    easing = "ease",
    expandOnHover = true,
    expandOnFocus = true,
    onexpand
  } = $props();



  
              
  let className = "";

  let isExpanded = false;
  let containerElement: HTMLDivElement

  function handleMouseEnter() {
    if (expandOnHover) {
      isExpanded = true;
      onexpand?.();
}}
  function handleMouseLeave() {
    if (expandOnHover) {
      isExpanded = false;
      onexpand?.();
}}
  function handleFocusIn() {
    if (expandOnFocus) {
      isExpanded = true;
      onexpand?.();
}}
  function handleFocusOut() {
    if (expandOnFocus) {
      isExpanded = false;
      onexpand?.();
}}
  let currentColumns = $derived(isExpanded ? expandedColumns : columns)
</script>

<div
  bind:this={containerElement}
  class="space-y-4"
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
  <slot></slot>
</div>

<style>
  /* @unocss-include */
  .expand-grid {
    display: grid
    grid-template-columns: repeat(var(--columns), 1fr);
    gap: var(--gap);
    transition: grid-template-columns var(--expand-duration) var(--easing);
    outline: none
    border-radius: 0.5rem;
    padding: 0.5rem;
    background: #fff;
    border: 1px solid transparent;
}
  .expand-grid:focus-visible {
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}
  .expand-grid.expanded {
    background: #f8fafc;
    border-color: #e5e7eb;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
}
  /* Grid item styling */
  .expand-grid :global(.grid-item) {
    transition: all var(--expand-duration) var(--easing);
    border-radius: 0.375rem;
    overflow: hidden
    position: relative
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
}}
  @media (max-width: 480px) {
    .expand-grid,
    .expand-grid.expanded {
      grid-template-columns: 1fr;
}}
</style>
