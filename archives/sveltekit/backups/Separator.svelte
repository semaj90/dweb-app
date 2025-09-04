// Backup of Separator.svelte before any future changes
// Created on 2025-07-20

<script lang="ts">
  import { Separator } from "bits-ui";
  export let orientation: 'horizontal' | 'vertical' = 'horizontal';
  export let decorative: boolean = false;
  export let class: string = '';
  export let ref: HTMLDivElement | null = null;
  export let children: unknown;
</script>

<Separator.Root
  {orientation}
  {decorative}
  bind:ref
  class={`nier-separator bg-border my-4 shrink-0 data-[orientation=horizontal]:h-px data-[orientation=vertical]:h-full data-[orientation=horizontal]:w-full data-[orientation=vertical]:w-[1px] ${class}`}
>
  {@render children()}
</Separator.Root>

<style>
  .nier-separator {
    @apply bg-border;
  }
</style>
