<script lang="ts">
  import { Tooltip } from "bits-ui";
  import { type Snippet } from "svelte";
  // Svelte 5 runes, UnoCSS, nier.css, context7 best practices
  export let icon: Snippet;
  export let tooltip: string = "AI Assistant";
  export let triggerClass: string = "";
  export let contentClass: string = "";
  export let delayDuration: number = 200;
  export let sideOffset: number = 8;
  export let children: any;
</script>

<Tooltip.Provider>
  <Tooltip.Root {delayDuration}>
    <Tooltip.Trigger
      class={`border-border-input bg-background-alt shadow-btn ring-dark ring-offset-background hover:bg-muted focus-visible:ring-dark focus-visible:ring-offset-background focus-visible:outline-hidden inline-flex size-10 items-center justify-center rounded-full border focus-visible:ring-2 focus-visible:ring-offset-2 ${triggerClass}`}
    >
      {@render icon()}
    </Tooltip.Trigger>
    <Tooltip.Content {sideOffset} class={contentClass}>
      <div class="rounded-input border-dark-10 bg-background shadow-popover outline-hidden z-0 flex items-center justify-center border p-3 text-sm font-medium">
        {tooltip}
        {@render children?.()}
      </div>
      <Tooltip.Arrow />
    </Tooltip.Content>
  </Tooltip.Root>
</Tooltip.Provider>

<!-- UnoCSS handles styling via class names; no <style> block needed. -->
