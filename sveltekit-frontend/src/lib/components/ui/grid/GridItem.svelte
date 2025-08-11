<!-- Grid Item Component -->
<script lang="ts">
  interface Props {
    colSpan: number ;
    rowSpan: number ;
    colStart: number | undefined ;
    rowStart: number | undefined ;
    responsive: boolean ;
    className: string ;
  }
  let {
    colSpan = 1,
    rowSpan = 1,
    colStart = undefined,
    rowStart = undefined,
    responsive = true,
    className = ''
  }: Props = $props();



  import { cn } from '$lib/utils';

            
  // Build grid classes dynamically
  $: spanClasses = responsive 
    ? `col-span-1 sm:col-span-${Math.min(colSpan, 2)} md:col-span-${Math.min(colSpan, 4)} lg:col-span-${Math.min(colSpan, 6)} xl:col-span-${colSpan}`
    : `col-span-${colSpan}`;

  $: rowSpanClass = rowSpan > 1 ? `row-span-${rowSpan}` : '';
  $: colStartClass = colStart ? `col-start-${colStart}` : '';
  $: rowStartClass = rowStart ? `row-start-${rowStart}` : '';
</script>

<div
  class={cn(
    'flex flex-col',
    spanClasses,
    rowSpanClass,
    colStartClass,
    rowStartClass,
    className
  )}
>
  <slot />
</div>
