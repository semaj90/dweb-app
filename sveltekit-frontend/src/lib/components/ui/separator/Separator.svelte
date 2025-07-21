
<script lang="ts">
  import { Separator } from "bits-ui";
  // Svelte 5 runes, UnoCSS, nier.css, context7 best practices
  export let orientation: 'horizontal' | 'vertical' = 'horizontal';
  export let decorative: boolean = false;
  export let className: string = '';
  export let ref: HTMLDivElement | null = null;
  export let children: any;
</script>


<Separator.Root
  {orientation}
  {decorative}
  bind:ref
  class={`nier-separator bg-border my-4 shrink-0 data-[orientation=horizontal]:h-px data-[orientation=vertical]:h-full data-[orientation=horizontal]:w-full data-[orientation=vertical]:w-[1px] ${className}`}
>
  {@render children?.()}
</Separator.Root>

<!-- UnoCSS handles styling via class names; no <style> block needed. -->
