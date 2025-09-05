<!-- Grid Item Component -->
<script lang="ts" context="module">
  interface Props {
    columns: number
    gap: "none" | "sm" | "md" | "lg" | "xl" ;
    responsive: boolean
    minHeight: string
    maxHeight: string
  }
  let {
    columns = 12,
    gap = "md",
    responsive = true,
    minHeight = "auto",
    maxHeight = "none"
  } = $props();



  export interface GridItemProps {
    colSpan?: number;
    rowSpan?: number;
    colStart?: number;
    rowStart?: number;
    responsive?: boolean;
    className?: string;
}
</script>

<!-- Grid Layout Component with svelte-brics inspired design -->
<script lang="ts">
  import { cn } from '$lib/utils';

        export const resizable: boolean = false;
    
  const gapClasses = {
    none: "gap-0",
    sm: "gap-2",
    md: "gap-4",
    lg: "gap-6",
    xl: "gap-8",
  };

  
  // Responsive breakpoints
  let gridClass = $derived(responsive)
    ? `grid-cols-1 sm:grid-cols-2 md:grid-cols-${Math.min(columns, 6)} lg:grid-cols-${Math.min(columns, 8)} xl:grid-cols-${columns}`
    : `grid-cols-${columns}`;
</script>

<div
  class={cn("grid w-full", gridClass, gapClasses[gap])}
  style:min-height={minHeight}
  style:max-height={maxHeight}
>
  <slot></slot>
</div>
