<!-- Grid Item Component -->
<script lang="ts">
  import { cn } from '$lib/utils';

  export let colSpan: number = 1;
  export let rowSpan: number = 1;
  export let colStart: number | undefined = undefined;
  export let rowStart: number | undefined = undefined;
  export let responsive: boolean = true;
  export let className: string = '';

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
