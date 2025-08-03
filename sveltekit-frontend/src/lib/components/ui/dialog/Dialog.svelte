<!-- Enhanced Dialog component with custom implementation -->
<script lang="ts">
  import { X } from "lucide-svelte";
  import { createEventDispatcher } from "svelte";
  import { quadOut } from "svelte/easing";
  import { fade, fly } from "svelte/transition";
  import { cn } from '$lib/utils';

  const dispatch = createEventDispatcher();

  export let open: boolean = false;
  export let title: string = "";
  export let description: string = "";
  export let size: "sm" | "md" | "lg" | "xl" | "full" = "md";
  export let showClose: boolean = true;
  export let closeOnOutsideClick: boolean = true;
  export let closeOnEscape: boolean = true;

  // Custom dialog implementation
  function handleClose() {
    open = false;
    dispatch("close");
}
  function handleKeydown(event: KeyboardEvent) {
    if (closeOnEscape && event.key === "Escape") {
      handleClose();
}}
  function handleOutsideClick(event: MouseEvent) {
    if (closeOnOutsideClick && event.target === event.currentTarget) {
      handleClose();
}}
  const sizeClasses = {
    sm: "max-w-sm",
    md: "max-w-md",
    lg: "max-w-lg",
    xl: "max-w-xl",
    full: "max-w-[95vw] max-h-[95vh]",
  };
</script>

<!-- Trigger slot -->
<slot name="trigger" />

<!-- Overlay -->
{#if open}
  <div
    class="container mx-auto px-4"
    transition:fade={{ duration: 200, easing: quadOut }}
    on:click={handleOutsideClick}
    on:keydown={handleKeydown}
    role="dialog"
    tabindex={0}
    aria-modal="true"
  ></div>
{/if}

<!-- Content -->
{#if open}
  <div
    class={cn(
      "fixed left-1/2 top-1/2 z-50 grid w-full -translate-x-1/2 -translate-y-1/2 gap-4 border border-slate-200 bg-white p-6 shadow-lg duration-200 dark:border-slate-800 dark:bg-slate-950 sm:rounded-lg",
      sizeClasses[size]
    )}
    transition:fly={{ y: -20, duration: 200, easing: quadOut }}
    role="dialog"
    tabindex={0}
    aria-modal="true"
    aria-labelledby={title ? "dialog-title" : undefined}
    aria-describedby={description ? "dialog-description" : undefined}
  >
    <!-- Header -->
    <div class="container mx-auto px-4">
      {#if title}
        <h2
          id="dialog-title"
          class="container mx-auto px-4"
        >
          {title}
        </h2>
      {/if}

      {#if description}
        <p
          id="dialog-description"
          class="container mx-auto px-4"
        >
          {description}
        </p>
      {/if}
    </div>

    <!-- Close button -->
    {#if showClose}
      <button
        class="container mx-auto px-4"
        on:click={() => handleClose()}
        aria-label="Close dialog"
      >
        <X class="container mx-auto px-4" />
        <span class="container mx-auto px-4">Close</span>
      </button>
    {/if}

    <!-- Content slot -->
    <div class="container mx-auto px-4">
      <slot />
    </div>

    <!-- Footer slot -->
    <div class="container mx-auto px-4">
      <slot name="footer" {close} />
    </div>
  </div>
{/if}
